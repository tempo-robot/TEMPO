"""Precompute SAM2.1-tiny post-memory-attention latents for every head-cam
frame in dynamic_handover_v2. One .npz per episode written to
$CACHE/episode_NNNNNN.npz containing:
  latent  : float32 (T, 256, 8, 8)
  action  : float32 (T, 14)
  state   : float32 (T, 14)
  ep_idx  : int32
  ep_len  : int32

Needs the SAM2 package importable (set SAM2_REPO / SAM2_CKPT).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sam2_encoder import (
    build_sam2_tiny,
    stream_extract,
    preprocess_video_to_sam2_input,
    LATENT_GRID,
    LATENT_DIM,
)

HEAD_KEY = "observation.images.head"


def read_video_rgb(mp4: Path) -> np.ndarray:
    """Decode an mp4 to numpy [T, H, W, 3] uint8 (RGB)."""
    cap = cv2.VideoCapture(str(mp4))
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not out:
        raise RuntimeError(f"empty video: {mp4}")
    return np.stack(out, axis=0)


def load_episode_meta(root: Path):
    with open(root / "meta" / "episodes.jsonl") as f:
        eps = [json.loads(line) for line in f]
    eps.sort(key=lambda x: x["episode_index"])
    return eps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="LeRobot dataset root")
    p.add_argument("--cache", required=True, help="output dir for episode_NNNNNN.npz")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=60)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    root = Path(args.root)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    print(f"build SAM2.1 tiny on {args.device}...")
    model = build_sam2_tiny(device=args.device)
    print(f"  image_size={model.image_size}, hidden_dim={model.hidden_dim}")

    eps = load_episode_meta(root)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)
    autocast.__enter__()
    try:
        t0 = time.time()
        for ep in eps:
            ep_idx = ep["episode_index"]
            if ep_idx < args.start or ep_idx >= args.end:
                continue
            outf = cache / f"episode_{ep_idx:06d}.npz"
            if outf.exists():
                print(f"  skip ep={ep_idx} (exists)")
                continue
            mp4 = root / "videos" / "chunk-000" / HEAD_KEY / f"episode_{ep_idx:06d}.mp4"
            pq_path = root / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"

            t_ep = time.time()
            video = read_video_rgb(mp4)
            T_video = video.shape[0]
            pq_table = pq.read_table(pq_path, columns=["frame_index", "action", "observation.state"])
            T_pq = pq_table.num_rows
            assert T_video == T_pq == ep["length"], (
                f"length mismatch ep={ep_idx}: video={T_video} pq={T_pq} meta={ep['length']}"
            )
            actions = np.stack(pq_table["action"].to_pylist()).astype(np.float32)  # (T, 14)
            states = np.stack(pq_table["observation.state"].to_pylist()).astype(np.float32)

            imgs = preprocess_video_to_sam2_input(video)
            lats = stream_extract(model, imgs, grid=LATENT_GRID, use_full_frame_mask=True)
            assert lats.shape == (T_video, LATENT_DIM, LATENT_GRID, LATENT_GRID)

            np.savez_compressed(
                outf,
                latent=lats.numpy().astype(np.float32),
                action=actions,
                state=states,
                ep_idx=np.int32(ep_idx),
                ep_len=np.int32(T_video),
            )
            dt = time.time() - t_ep
            print(f"  ep={ep_idx:3d} T={T_video} -> {outf.name} ({dt:.1f}s, "
                  f"{T_video/dt:.1f} fps) total={time.time()-t0:.1f}s")
    finally:
        autocast.__exit__(None, None, None)
    print("done.")


if __name__ == "__main__":
    main()
