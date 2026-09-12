"""Dataset over precomputed SAM2 latent caches.

Each episode is stored as one .npz at $CACHE/episode_NNNNNN.npz with keys:
  latent: (T, 256, 8, 8) float32
  action: (T, 14) float32

We sample 16-frame stride-1 windows. For an episode of length T we get
T - W + 1 windows; train/val split is determined by episode index per
split.json.

Returns dicts with:
  latent: (W, 256, 8, 8) float32 tensor
  action: (W, 14) float32 tensor
  ep_idx: int
  t0:     int  # absolute start frame
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


WINDOW = 16
STRIDE = 1


class SAM2LatentWindowDataset(Dataset):
    """Window dataset over precomputed SAM2 latents + actions + state.

    pool_spatial=True replaces the cached (T, 256, 8, 8) latent with the
    spatially-averaged (T, 256), the exact representation the prior
    velocity-probe used to get R²≈0.89. Downstream MVAE config is then run
    with sam_grid=1.
    """

    def __init__(self, cache_dir: str, episodes: list[int], window: int = WINDOW,
                 stride: int = STRIDE, pool_spatial: bool = False):
        self.cache_dir = Path(cache_dir)
        self.window = window
        self.stride = stride
        self.pool_spatial = pool_spatial
        self.episodes = sorted(episodes)
        # Preload episodes into memory (small: ~600 MB total across all 60).
        self._ep_data: dict[int, dict] = {}
        self.index: list[tuple[int, int]] = []  # list of (ep_idx, t0)
        for ep in self.episodes:
            f = self.cache_dir / f"episode_{ep:06d}.npz"
            with np.load(f) as z:
                latent = z["latent"]  # (T, 256, 8, 8)
                action = z["action"]  # (T, 14)
                state = z["state"] if "state" in z.files else action.copy()
            if pool_spatial:
                latent = latent.mean(axis=(2, 3), keepdims=False)  # (T, 256)
                latent = latent[:, :, None, None]                    # (T, 256, 1, 1)
            self._ep_data[ep] = {"latent": latent, "action": action, "state": state}
            T = latent.shape[0]
            for t0 in range(0, T - window + 1, stride):
                self.index.append((ep, t0))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        ep, t0 = self.index[i]
        d = self._ep_data[ep]
        latent = d["latent"][t0:t0 + self.window]  # (W, 256, H, W')
        action = d["action"][t0:t0 + self.window]
        state = d["state"][t0:t0 + self.window]
        return {
            "latent": torch.from_numpy(latent.copy()),
            "action": torch.from_numpy(action.copy()),
            "state": torch.from_numpy(state.copy()),
            "ep_idx": int(ep),
            "t0": int(t0),
        }


def resolve_split(cache_dir: str, split_path: str | None = None,
                  val_frac: float = 0.2, seed: int = 0) -> dict:
    """Episode-level train/val split.

    With `split_path`, read {"train_episodes": [...], "val_episodes": [...]} from that JSON.
    Without it, derive a deterministic split from the episodes present in `cache_dir`, so the
    pipeline runs on a fresh dataset with no extra setup. The split actually used is written
    next to every run's checkpoint, so it can be reused verbatim by the later stages.
    """
    if split_path is not None:
        with open(split_path) as f:
            return json.load(f)
    eps = sorted(int(p.stem.split("_")[-1]) for p in Path(cache_dir).glob("episode_*.npz"))
    if not eps:
        raise FileNotFoundError(f"no episode_*.npz under {cache_dir}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(eps))
    n_val = max(1, int(round(val_frac * len(eps))))
    val = sorted(eps[i] for i in perm[:n_val])
    train = sorted(eps[i] for i in perm[n_val:])
    return {"train_episodes": train, "val_episodes": val}


def make_datasets(cache_dir: str, split_path: str | None = None,
                  window: int = WINDOW, stride: int = STRIDE,
                  pool_spatial: bool = False):
    split = resolve_split(cache_dir, split_path)
    train_ds = SAM2LatentWindowDataset(cache_dir, split["train_episodes"], window,
                                       stride, pool_spatial=pool_spatial)
    val_ds = SAM2LatentWindowDataset(cache_dir, split["val_episodes"], window,
                                     stride, pool_spatial=pool_spatial)
    return train_ds, val_ds, split
