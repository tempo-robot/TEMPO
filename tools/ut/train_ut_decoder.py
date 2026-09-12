"""Stage 3 of the u_t pipeline: train the decoder the POLICY uses at inference.

Maps mu_v (ut_dim) -> action chunk (W, action_dim), plus an auxiliary mu_v -> SAM2 latent
window head that regularizes the shared task. The MVAE's own joint-PoE decoders cannot be
used here: they are conditioned on the fused latent, which sees the ground-truth action chunk
through the action encoder, so they are unusable on mu_v alone. These decoders are trained
from mu_v only -- exactly what the policy predicts at inference.

IMPORTANT: actions are normalized with the POLICY's norm stats (openpi assets
norm_stats.json, quantile for pi0.5), not the MVAE's own mean/std. The decoder therefore
emits actions in the space `sample_actions` returns, and openpi's standard Unnormalize output
transform recovers raw actions with no special-casing. The checkpoint records
action_space="policy_norm"; the model refuses to load a decoder trained in any other space.

  python tools/ut/train_ut_decoder.py \
      --mvae-ckpt <mvae_run>/checkpoints/last.pt \
      --sam-cache <sam2_cache> --ut-dir <u_window dir> \
      --norm-stats <openpi assets>/<repo_id> --out <decoder_run>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mvae import MVAEConfig, ActionDecoder, VideoLatentDecoder

from openpi.shared import normalize as _normalize


WINDOW = 16
SAM_DIM = 256


def policy_normalizer(norm_stats_dir: str, use_quantiles: bool = True):
    """The exact action normalization openpi applies, so the decoder trains in the policy's space."""
    stats = _normalize.load(norm_stats_dir)["actions"]
    if use_quantiles:
        if stats.q01 is None or stats.q99 is None:
            raise ValueError(
                f"{norm_stats_dir}/norm_stats.json has no quantiles, but pi0.5 normalizes actions "
                "with q01/q99. Recompute them with scripts/compute_norm_stats.py."
            )
        q01, q99 = np.asarray(stats.q01, np.float32), np.asarray(stats.q99, np.float32)

        def fn(a):  # matches transforms.Normalize._normalize_quantile
            d = a.shape[-1]
            return (a - q01[:d]) / (q99[:d] - q01[:d] + 1e-6) * 2.0 - 1.0

        return fn, {"q01": q01, "q99": q99, "use_quantiles": True}

    mean, std = np.asarray(stats.mean, np.float32), np.asarray(stats.std, np.float32)

    def fn(a):  # matches transforms.Normalize._normalize
        d = a.shape[-1]
        return (a - mean[:d]) / (std[:d] + 1e-6)

    return fn, {"mean": mean, "std": std, "use_quantiles": False}


class MuvWindowDataset(Dataset):
    """For each valid W-frame window in an episode, return:
       u   (ut_dim,)      : precomputed mu_v of that window (sidecar)
       act (W, action_dim): action chunk at the same window, in the POLICY's normalized space
       sam (W, 256)       : SAM2 latent pooled across spatial dims (auxiliary target)
    """
    def __init__(self, episodes: list[int], sam_cache: str, ut_dir: str, normalize_actions):
        self.episodes = sorted(episodes)
        sam_cache = Path(sam_cache); ut_dir = Path(ut_dir)
        self._u: dict[int, np.ndarray] = {}
        self._sam: dict[int, np.ndarray] = {}
        self._act_n: dict[int, np.ndarray] = {}
        for ep in self.episodes:
            with np.load(ut_dir / f"episode_{ep:06d}.npz") as z:
                u = z["u_window"]              # (N_win, ut_dim)
            with np.load(sam_cache / f"episode_{ep:06d}.npz") as z:
                sam = z["latent"].mean(axis=(2, 3))  # (T, 256)
                act = z["action"]                    # (T, action_dim)
            act_n = normalize_actions(act.astype(np.float32))
            self._u[ep] = u; self._sam[ep] = sam; self._act_n[ep] = act_n
        self.index: list[tuple[int, int]] = []
        for ep in self.episodes:
            N = self._u[ep].shape[0]
            for t in range(N):
                self.index.append((ep, t))

    def __len__(self): return len(self.index)

    def __getitem__(self, i):
        ep, t0 = self.index[i]
        u = self._u[ep][t0]                                  # (64,)
        sam = self._sam[ep][t0:t0 + WINDOW]                  # (16, 256)
        act = self._act_n[ep][t0:t0 + WINDOW]                # (16, 14)
        return {
            "u": torch.from_numpy(u.copy()).float(),
            "sam": torch.from_numpy(sam.copy()).float(),
            "act": torch.from_numpy(act.copy()).float(),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mvae-ckpt", required=True, help="MVAE checkpoint (for the decoder arch config)")
    ap.add_argument("--sam-cache", required=True, help="SAM2 token cache dir")
    ap.add_argument("--ut-dir", required=True, help="u_window sidecars from encode_ut_windows.py")
    ap.add_argument("--norm-stats", required=True,
                    help="openpi assets dir holding norm_stats.json for this repo_id")
    ap.add_argument("--no-quantile-norm", action="store_true",
                    help="z-score instead of quantile normalization (pi0 rather than pi0.5)")
    ap.add_argument("--split", default=None,
                    help="split JSON; defaults to the one saved by the MVAE run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    # Load the MVAE checkpoint for the decoder architecture (and the split it trained on).
    ck = torch.load(args.mvae_ckpt, map_location="cpu", weights_only=False)
    cfg = MVAEConfig(**ck["model_cfg"])

    normalize_actions, norm_meta = policy_normalizer(args.norm_stats, not args.no_quantile_norm)
    np.savez(out / "action_norm.npz", **{k: v for k, v in norm_meta.items() if k != "use_quantiles"})

    if args.split is not None:
        with open(args.split) as f:
            split = json.load(f)
    elif "split" in ck:
        split = ck["split"]
        print("using the split saved in the MVAE checkpoint")
    else:
        raise ValueError("no --split given and the MVAE checkpoint has none saved")
    with open(out / "split.json", "w") as f:
        json.dump(split, f, indent=2)

    print("loading train data...")
    train_ds = MuvWindowDataset(split["train_episodes"], args.sam_cache, args.ut_dir, normalize_actions)
    val_ds = MuvWindowDataset(split["val_episodes"], args.sam_cache, args.ut_dir, normalize_actions)
    print(f"  train windows: {len(train_ds)}  val windows: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=args.num_workers > 0)

    # Fresh decoders, mirroring MVAE arch but trained from μ_v alone.
    sam_dec = VideoLatentDecoder(cfg).to(device)
    act_dec = ActionDecoder(cfg).to(device)
    params = list(sam_dec.parameters()) + list(act_dec.parameters())
    print(f"params: {sum(p.numel() for p in params)/1e6:.2f}M")

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def run_epoch(loader, train: bool):
        sam_dec.train(train); act_dec.train(train)
        tot = {"sam": 0.0, "act": 0.0, "n": 0}
        for batch in loader:
            u = batch["u"].to(device, non_blocking=True)
            sam_gt = batch["sam"].to(device, non_blocking=True)  # (B, 16, 256)
            act_gt = batch["act"].to(device, non_blocking=True)  # (B, 16, 14)
            with torch.set_grad_enabled(train):
                sam_pred = sam_dec(u)                            # (B, 16, 256, 1, 1)
                sam_pred_2d = sam_pred.squeeze(-1).squeeze(-1)   # (B, 16, 256)
                act_pred = act_dec(u)                            # (B, 16, 14)
                loss_sam = F.mse_loss(sam_pred_2d, sam_gt)
                loss_act = F.mse_loss(act_pred, act_gt)
                loss = loss_sam + loss_act
            if train:
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
            B = u.shape[0]
            tot["sam"] += loss_sam.item() * B
            tot["act"] += loss_act.item() * B
            tot["n"] += B
        for k in ("sam", "act"): tot[k] /= max(tot["n"], 1)
        return tot

    history = []; best = float("inf")
    log_path = out / "metrics.jsonl"
    f = open(log_path, "a")
    try:
        for ep in range(args.epochs):
            t0 = time.time()
            tr = run_epoch(train_loader, True)
            with torch.no_grad():
                va = run_epoch(val_loader, False)
            sched.step()
            row = {
                "epoch": ep,
                "lr": sched.get_last_lr()[0],
                "train_mse_sam": tr["sam"], "train_mse_act": tr["act"],
                "val_mse_sam": va["sam"],   "val_mse_act": va["act"],
                "dt": time.time() - t0,
            }
            history.append(row); f.write(json.dumps(row) + "\n"); f.flush()
            print(
                f"ep {ep:3d} | tr sam {tr['sam']:.4f} act {tr['act']:.4f} | "
                f"va sam {va['sam']:.4f} act {va['act']:.4f} | dt {row['dt']:.1f}s"
            )
            if va["sam"] + va["act"] < best:
                best = va["sam"] + va["act"]
                torch.save({
                    "sam_dec": sam_dec.state_dict(),
                    "act_dec": act_dec.state_dict(),
                    "model_cfg": ck["model_cfg"],
                    "action_space": "policy_norm",
                    "action_norm": norm_meta,
                    "norm_stats_dir": args.norm_stats,
                    "epoch": ep, "val": va,
                }, out / "best.pt")
    finally:
        f.close()
    torch.save({
        "sam_dec": sam_dec.state_dict(), "act_dec": act_dec.state_dict(),
        "model_cfg": ck["model_cfg"],
        "action_space": "policy_norm",
        "action_norm": norm_meta,
        "norm_stats_dir": args.norm_stats,
        "epoch": args.epochs - 1, "val": va,
    }, out / "last.pt")
    print("\n=== final ===")
    print(json.dumps(history[-1], indent=2))


if __name__ == "__main__":
    main()
