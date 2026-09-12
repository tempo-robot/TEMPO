"""Train the PoE MVAE on cached SAM2 latents + actions for dyn_handover_v2.

Reports per-epoch:
  train_mse_latent, train_mse_action, train_kl
  val_mse_latent,   val_mse_action,   val_kl

Saves checkpoint at $OUT/checkpoints/best.pt by val_total = val_mse_latent +
action_loss_weight * val_mse_action.

Stage 1 of the u_t pipeline:
  python tools/ut/train_mvae.py --cache <sam2_cache> --out <run_dir>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import make_datasets, WINDOW, STRIDE
from mvae import MVAEConfig, PoEMVAE


@dataclass
class TrainCfg:
    cache: str
    out: str
    split: str | None = None
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    grad_clip: float = 1.0
    log_every: int = 50
    seed: int = 0
    # MVAE knobs.
    latent_dim: int = 256
    hidden_dim: int = 512
    enc_layers: int = 2
    dec_layers: int = 2
    nhead: int = 8
    beta: float = 1e-3
    action_loss_weight: float = 1.0
    # Action normalization.
    action_norm: str = "meanstd"   # "none" or "meanstd"
    # SAM2 input shape.
    pool_spatial: bool = False  # True -> (T, 256, 1, 1) global-avg-pool


def build_action_stats(train_ds) -> tuple[np.ndarray, np.ndarray]:
    actions = np.concatenate([
        train_ds._ep_data[ep]["action"] for ep in train_ds.episodes
    ], axis=0)  # (sum_T, 14)
    mean = actions.mean(axis=0)
    std = actions.std(axis=0) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default=None,
                    help="JSON with train_episodes/val_episodes; omit for a deterministic 80/20 split")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--latent-dim", type=int, default=256)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--enc-layers", type=int, default=2)
    ap.add_argument("--dec-layers", type=int, default=2)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--beta", type=float, default=1e-3)
    ap.add_argument("--action-loss-weight", type=float, default=1.0)
    ap.add_argument("--action-norm", default="meanstd", choices=["none", "meanstd"])
    ap.add_argument("--pool-spatial", action="store_true",
                    help="Use global-avg-pooled SAM2 latent (T,256) instead of (T,256,8,8).")
    args = ap.parse_args()

    cfg = TrainCfg(**{k: getattr(args, k) for k in TrainCfg.__dataclass_fields__})
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    with open(out / "train_cfg.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("loading datasets...")
    train_ds, val_ds, split = make_datasets(cfg.cache, cfg.split, WINDOW, STRIDE,
                                            pool_spatial=cfg.pool_spatial)
    print(f"  train: {len(train_ds)} windows from {len(split['train_episodes'])} eps")
    print(f"  val:   {len(val_ds)} windows from {len(split['val_episodes'])} eps")
    with open(out / "split.json", "w") as f:
        json.dump(split, f, indent=2)

    # Action normalization (computed on train only).
    if cfg.action_norm == "meanstd":
        a_mean, a_std = build_action_stats(train_ds)
    else:
        a_mean = np.zeros(14, dtype=np.float32)
        a_std = np.ones(14, dtype=np.float32)
    np.savez(out / "action_stats.npz", mean=a_mean, std=a_std)
    a_mean_t = torch.from_numpy(a_mean).to(device)
    a_std_t = torch.from_numpy(a_std).to(device)

    model_cfg = MVAEConfig(
        window=WINDOW, sam_dim=256, sam_grid=1 if cfg.pool_spatial else 8, action_dim=14,
        hidden_dim=cfg.hidden_dim, latent_dim=cfg.latent_dim,
        enc_layers=cfg.enc_layers, dec_layers=cfg.dec_layers, nhead=cfg.nhead,
        beta=cfg.beta, action_loss_weight=cfg.action_loss_weight,
    )
    model = PoEMVAE(model_cfg).to(device)
    print(f"  params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    def normalize_action(a):
        return (a - a_mean_t) / a_std_t if cfg.action_norm == "meanstd" else a

    def run_epoch(loader, train: bool):
        model.train(train)
        tot = {"loss": 0.0, "mse_latent": 0.0, "mse_action": 0.0, "kl": 0.0, "n": 0}
        for batch in loader:
            latent = batch["latent"].to(device, non_blocking=True)        # (B, W, 256, 8, 8)
            action = batch["action"].to(device, non_blocking=True)        # (B, W, 14)
            action_n = normalize_action(action)
            with torch.set_grad_enabled(train):
                out_ = model(latent, action_n)
                loss = out_["loss"]
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
            B = latent.shape[0]
            tot["loss"] += loss.item() * B
            tot["mse_latent"] += out_["mse_latent"].item() * B
            tot["mse_action"] += out_["mse_action"].item() * B
            tot["kl"] += out_["kl"].item() * B
            tot["n"] += B
        for k in ("loss", "mse_latent", "mse_action", "kl"):
            tot[k] /= max(tot["n"], 1)
        return tot

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True,
                              drop_last=True, persistent_workers=cfg.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True,
                            drop_last=False, persistent_workers=cfg.num_workers > 0)

    history = []
    best_val = float("inf")
    log_path = out / "metrics.jsonl"
    log_f = open(log_path, "a")
    try:
        for epoch in range(cfg.epochs):
            t0 = time.time()
            tr = run_epoch(train_loader, train=True)
            va = run_epoch(val_loader, train=False)
            sched.step()
            val_total = va["mse_latent"] + cfg.action_loss_weight * va["mse_action"]
            row = {
                "epoch": epoch,
                "lr": sched.get_last_lr()[0],
                "train_loss": tr["loss"],
                "train_mse_latent": tr["mse_latent"],
                "train_mse_action": tr["mse_action"],
                "train_kl": tr["kl"],
                "val_loss": va["loss"],
                "val_mse_latent": va["mse_latent"],
                "val_mse_action": va["mse_action"],
                "val_kl": va["kl"],
                "val_total": val_total,
                "dt": time.time() - t0,
            }
            history.append(row)
            log_f.write(json.dumps(row) + "\n"); log_f.flush()
            print(
                f"ep {epoch:3d} | tr_lat {tr['mse_latent']:.4f} tr_act {tr['mse_action']:.4f} "
                f"tr_kl {tr['kl']:.3f} | va_lat {va['mse_latent']:.4f} va_act {va['mse_action']:.4f} "
                f"va_kl {va['kl']:.3f} | dt {row['dt']:.1f}s"
            )
            if val_total < best_val:
                best_val = val_total
                torch.save({
                    "model": model.state_dict(),
                    "cfg": asdict(cfg),
                    "model_cfg": asdict(model_cfg),
                    "epoch": epoch,
                    "val": va,
                    "action_mean": a_mean,
                    "action_std": a_std,
                    "split": split,
                }, out / "checkpoints" / "best.pt")
        torch.save({
            "model": model.state_dict(),
            "cfg": asdict(cfg),
            "model_cfg": asdict(model_cfg),
            "epoch": cfg.epochs - 1,
            "val": va,
            "action_mean": a_mean,
            "action_std": a_std,
            "split": split,
        }, out / "checkpoints" / "last.pt")
    finally:
        log_f.close()

    # Final-epoch summary.
    print("\n=== final ===")
    print(json.dumps(history[-1], indent=2))

    # === Linear-probe of u → action_chunk ===
    # Use the last-epoch model; encode every train and val window, then fit a
    # Ridge regression on train (u -> flattened action chunk in normalized
    # space) and report R^2 on val. Mirrors how the prior SAM2 latent was
    # probed for ball velocity.
    try:
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score
    except Exception as e:
        print(f"sklearn unavailable, skipping probe: {e}")
        return

    @torch.no_grad()
    def encode_u(loader):
        model.eval()
        us, ys = [], []
        for batch in loader:
            latent = batch["latent"].to(device, non_blocking=True)
            action = batch["action"].to(device, non_blocking=True)
            action_n = normalize_action(action)
            mu_v, lv_v = model.video_enc(latent)
            mu_a, lv_a = model.action_enc(action_n)
            mu, _ = model.fuse_poe(mu_v, lv_v, mu_a, lv_a)
            # Probe should be based on video-only u, otherwise it trivially
            # uses the action info we want to predict. Use mu_v as the
            # representation. We also save mu (joint).
            us.append(mu_v.cpu().numpy())
            ys.append(action_n.cpu().numpy().reshape(action_n.shape[0], -1))
        return np.concatenate(us, 0), np.concatenate(ys, 0)

    train_loader_seq = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False,
                                  num_workers=cfg.num_workers, pin_memory=True)
    val_loader_seq = val_loader
    print("encoding u_v over train + val for probe...")
    u_tr, y_tr = encode_u(train_loader_seq)
    u_va, y_va = encode_u(val_loader_seq)
    np.savez(out / "probe_features.npz", u_train=u_tr, y_train=y_tr, u_val=u_va, y_val=y_va)

    print(f"  u_train {u_tr.shape}  y_train {y_tr.shape}  u_val {u_va.shape}  y_val {y_va.shape}")
    probe = Ridge(alpha=1.0)
    probe.fit(u_tr, y_tr)
    r2_tr = r2_score(y_tr, probe.predict(u_tr), multioutput="raw_values")
    r2_va = r2_score(y_va, probe.predict(u_va), multioutput="raw_values")
    mse_tr = float(((probe.predict(u_tr) - y_tr) ** 2).mean())
    mse_va = float(((probe.predict(u_va) - y_va) ** 2).mean())
    probe_row = {
        "latent_dim": cfg.latent_dim,
        "probe_target": "action_chunk_normalized",
        "probe_input": "mu_v (video-only)",
        "r2_train_mean": float(r2_tr.mean()),
        "r2_val_mean": float(r2_va.mean()),
        "r2_train_per_dim": r2_tr.tolist(),
        "r2_val_per_dim": r2_va.tolist(),
        "mse_train": mse_tr,
        "mse_val": mse_va,
    }
    with open(out / "probe.json", "w") as f:
        json.dump(probe_row, f, indent=2)
    print("PROBE: r2_train_mean={:.4f} r2_val_mean={:.4f}  (mse_tr={:.4f} mse_va={:.4f})".format(
        probe_row["r2_train_mean"], probe_row["r2_val_mean"], mse_tr, mse_va,
    ))


if __name__ == "__main__":
    main()
