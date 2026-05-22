"""Wrap a VLASH dataset to inject motion_past + motion_future features per
sample from precomputed sidecars (see /home/dfeng8/sam2_uvt/precompute_motion_diff.py).

Per-episode sidecar layout: $MOTION_DIR/episode_NNNNNN.npz with arrays:
  motion_past, motion_future          : (T, 256) float32
  motion_past_is_pad, motion_future_is_pad : (T,) bool
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


OBS_MOTION_PAST = "observation.motion_past"
OBS_MOTION_FUTURE = "observation.motion_future"
OBS_MOTION_FUTURE_IS_PAD = "observation.motion_future_is_pad"


class MotionWrappedDataset(Dataset):
    def __init__(self, base_dataset, motion_dir: str, motion_dim: int = 256):
        self.base = base_dataset
        self.motion_dim = int(motion_dim)
        self.motion_dir = Path(motion_dir)
        self._past: dict[int, np.ndarray] = {}
        self._future: dict[int, np.ndarray] = {}
        self._past_pad: dict[int, np.ndarray] = {}
        self._future_pad: dict[int, np.ndarray] = {}
        for f in sorted(self.motion_dir.glob("episode_*.npz")):
            ep = int(f.stem.split("_")[-1])
            with np.load(f) as z:
                self._past[ep] = z["motion_past"]
                self._future[ep] = z["motion_future"]
                self._past_pad[ep] = z["motion_past_is_pad"].astype(bool)
                self._future_pad[ep] = z["motion_future_is_pad"].astype(bool)
        if not self._past:
            raise FileNotFoundError(f"no episode_*.npz under {self.motion_dir}")
        for attr in ("meta", "episodes", "num_episodes", "num_frames",
                     "delta_indices", "delta_timestamps"):
            if hasattr(self.base, attr):
                setattr(self, attr, getattr(self.base, attr))

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        ep = int(item["episode_index"].item()
                 if hasattr(item["episode_index"], "item") else item["episode_index"])
        frame_idx = int(item["frame_index"].item()
                        if hasattr(item["frame_index"], "item") else item["frame_index"])

        if ep in self._past and 0 <= frame_idx < self._past[ep].shape[0]:
            past = self._past[ep][frame_idx]
            future = self._future[ep][frame_idx]
            past_pad = bool(self._past_pad[ep][frame_idx])
            future_pad = bool(self._future_pad[ep][frame_idx])
        else:
            past = np.zeros(self.motion_dim, dtype=np.float32)
            future = np.zeros(self.motion_dim, dtype=np.float32)
            past_pad = True; future_pad = True

        item[OBS_MOTION_PAST] = torch.from_numpy(past.copy()).float()
        item[OBS_MOTION_FUTURE] = torch.from_numpy(future.copy()).float()
        item[OBS_MOTION_FUTURE_IS_PAD] = torch.tensor(future_pad, dtype=torch.bool)
        return item


class SharedObservationMotionWrappedDataset(Dataset):
    """Per-offset motion target wrapper around SharedObservationVLASHDataset.

    The shared-observation dataset returns one shared observation plus
    `num_offsets` action chunks (one per temporal offset k=0..max_delay_steps).
    For each sample we inject:
      - observation.motion_past: (motion_dim,) — single, anchored at the
        observation frame t (independent of offset, since the observation is
        what feeds adarms_cond).
      - observation.motion_future: (num_offsets, motion_dim) — per-offset
        target, anchored at frame t+k (matching where the k-th action chunk
        starts).
      - observation.motion_future_is_pad: (num_offsets,) bool.

    The episode-relative frame index t is reconstructed from the dataset
    index using `t = idx - ep_start` (which is also how the underlying
    LeRobotDataset numbers samples).
    """

    def __init__(self, base_dataset, motion_dir: str, motion_dim: int = 256):
        self.base = base_dataset
        self.motion_dim = int(motion_dim)
        self.motion_dir = Path(motion_dir)
        self._past: dict[int, np.ndarray] = {}
        self._future: dict[int, np.ndarray] = {}
        self._past_pad: dict[int, np.ndarray] = {}
        self._future_pad: dict[int, np.ndarray] = {}
        for f in sorted(self.motion_dir.glob("episode_*.npz")):
            ep = int(f.stem.split("_")[-1])
            with np.load(f) as z:
                self._past[ep] = z["motion_past"]
                self._future[ep] = z["motion_future"]
                self._past_pad[ep] = z["motion_past_is_pad"].astype(bool)
                self._future_pad[ep] = z["motion_future_is_pad"].astype(bool)
        if not self._past:
            raise FileNotFoundError(f"no episode_*.npz under {self.motion_dir}")
        for attr in ("meta", "episodes", "num_episodes", "num_frames",
                     "delta_indices", "delta_timestamps", "max_delay_steps"):
            if hasattr(self.base, attr):
                setattr(self, attr, getattr(self.base, attr))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        ep_idx = int(item["episode_index"].item()
                     if hasattr(item["episode_index"], "item") else item["episode_index"])
        ep = self.base.meta.episodes[ep_idx]
        ep_start = ep["dataset_from_index"]
        t = idx - ep_start
        n = int(item["num_offsets"])

        if ep_idx in self._past:
            T = self._past[ep_idx].shape[0]
        else:
            T = 0

        if 0 <= t < T:
            past = self._past[ep_idx][t].copy()
        else:
            past = np.zeros(self.motion_dim, dtype=np.float32)

        future = np.zeros((n, self.motion_dim), dtype=np.float32)
        future_pad = np.ones(n, dtype=bool)
        if ep_idx in self._future:
            for k in range(n):
                f = t + k
                if 0 <= f < T:
                    future[k] = self._future[ep_idx][f]
                    future_pad[k] = bool(self._future_pad[ep_idx][f])

        item[OBS_MOTION_PAST] = torch.from_numpy(past).float()
        item[OBS_MOTION_FUTURE] = torch.from_numpy(future).float()
        item[OBS_MOTION_FUTURE_IS_PAD] = torch.from_numpy(future_pad).bool()
        return item
