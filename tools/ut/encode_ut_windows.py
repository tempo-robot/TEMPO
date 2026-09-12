"""Encode per-window μ_v from a trained MVAE (u=64, β=0) over the cached SAM2
latents of dynamic_handover_v2. Saves per-episode sidecar:

  $OUT/episode_NNNNNN.npz keys:
    u_window: (T - W + 1, 64) float32  — μ_v for window starting at frame t0.
    ep_len:   int

Loaded by the vlash_uvt dataset as: u_past(t) = u_window[t-W] (zero-pad if t<W)
                                     u_target(t) = u_window[t]   (skip if t>T-W)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import SAM2LatentWindowDataset, resolve_split
from mvae import MVAEConfig, PoEMVAE


WINDOW = 16


@torch.no_grad()
def encode_episodes(model, cache_dir: Path, out_dir: Path, episodes: list[int],
                    action_mean: np.ndarray, action_std: np.ndarray,
                    device: torch.device, batch_size: int = 256):
    out_dir.mkdir(parents=True, exist_ok=True)
    action_norm = action_mean is not None
    a_mu = torch.from_numpy(action_mean).to(device) if action_norm else None
    a_sd = torch.from_numpy(action_std).to(device) if action_norm else None
    t0_all = time.time()
    for ep in sorted(episodes):
        outf = out_dir / f"episode_{ep:06d}.npz"
        if outf.exists():
            print(f"  skip ep={ep} (exists)")
            continue
        # Build a single-episode dataset of all stride-1 windows, pool_spatial=True.
        ds = SAM2LatentWindowDataset(str(cache_dir), [ep], window=WINDOW,
                                     stride=1, pool_spatial=True)
        if len(ds) == 0:
            print(f"  ep={ep}: too short, skipping")
            continue
        # Iterate sequentially in t0 order — ds.index already sorted.
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
        us = []
        for batch in loader:
            latent = batch["latent"].to(device, non_blocking=True)
            # Note: μ_v only depends on `latent` — action is unused here.
            mu_v, _ = model.video_enc(latent)
            us.append(mu_v.float().cpu().numpy())
        u_window = np.concatenate(us, axis=0)
        T_total = ds._ep_data[ep]["latent"].shape[0]
        assert u_window.shape[0] == T_total - WINDOW + 1, (
            f"ep={ep}: expected {T_total - WINDOW + 1} windows, got {u_window.shape[0]}")
        np.savez(outf, u_window=u_window.astype(np.float32), ep_len=np.int32(T_total))
        print(f"  ep={ep:3d}: T={T_total} -> u_window={u_window.shape}  total={time.time()-t0_all:.1f}s")
    print(f"done in {time.time()-t0_all:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mvae", required=True, help="MVAE checkpoint from train_mvae.py")
    ap.add_argument("--cache", required=True, help="SAM2 token cache dir")
    ap.add_argument("--out", required=True, help="output dir for the u_window sidecars")
    ap.add_argument("--split", default=None,
                    help="optional split JSON; without it every episode in --cache is encoded")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.mvae, map_location=device, weights_only=False)
    model_cfg = MVAEConfig(**ckpt["model_cfg"])
    model = PoEMVAE(model_cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    print(f"loaded MVAE u={model_cfg.latent_dim}, sam_grid={model_cfg.sam_grid}, "
          f"window={model_cfg.window}, params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    split = resolve_split(args.cache, args.split)
    all_eps = sorted(set(split["train_episodes"]) | set(split["val_episodes"]))
    print(f"encoding {len(all_eps)} episodes")

    encode_episodes(model, Path(args.cache), Path(args.out), all_eps,
                    None, None, device, args.batch_size)

    # Also write a small manifest with the MVAE info.
    manifest = {
        "mvae_ckpt": args.mvae,
        "sam2_cache": args.cache,
        "split": args.split,
        "u_dim": int(model_cfg.latent_dim),
        "window": int(model_cfg.window),
        "pool_spatial": bool(model_cfg.sam_grid == 1),
    }
    with open(Path(args.out) / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("wrote manifest")


if __name__ == "__main__":
    main()
