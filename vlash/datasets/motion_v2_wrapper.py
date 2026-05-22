"""Motion-V2 dataset wrappers.

On top of the underlying VLASHDataset / SharedObservationVLASHDataset, inject:
  - observation.sam2_tokens       : (sam2_num_tokens, sam2_token_dim) float32.
                                    SAM2 mem-attn latent at the observation
                                    frame t, 2x-avg-pooled spatially
                                    (256x8x8 -> 256x4x4) and reshaped into 16
                                    spatial tokens of 256-d each.
  - observation.action_history     : (steps, action_dim) float32 — mean-aggregated
                                    past actions in `steps` buckets ending at t.
  - observation.action_history_is_pad : (steps,) bool — True for buckets that
                                    fall before episode start.
  - observation.motion_future      : (motion_dim,) or (num_offsets, motion_dim).
  - observation.motion_future_is_pad

motion_past from v1 is NOT emitted (v2 replaces it with sam2_tokens).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


OBS_SAM2_TOKENS = "observation.sam2_tokens"
OBS_ACTION_HISTORY = "observation.action_history"
OBS_ACTION_HISTORY_IS_PAD = "observation.action_history_is_pad"
OBS_MOTION_FUTURE = "observation.motion_future"
OBS_MOTION_FUTURE_IS_PAD = "observation.motion_future_is_pad"


def _load_sam2_tokens(sam2_dir: Path) -> dict[int, np.ndarray]:
    """Load raw SAM2 cache, 2x-avg-pool spatially -> (T, H/2*W/2, C) per ep.

    For (T, 256, 8, 8) input this yields (T, 16, 256): 16 spatial tokens
    (each a 256-d feature vector) per frame, ready to feed into the VLM
    prefix as 16 tokens projected to vlm_hidden_size.
    """
    pooled: dict[int, np.ndarray] = {}
    for f in sorted(Path(sam2_dir).glob("episode_*.npz")):
        ep = int(f.stem.split("_")[-1])
        with np.load(f) as z:
            lat = z["latent"]  # (T, C, H, W) typically (T, 256, 8, 8)
        if lat.ndim != 4:
            raise ValueError(f"{f}: expected 4-D latent, got shape {lat.shape}")
        T, C, H, W = lat.shape
        if H % 2 or W % 2:
            raise ValueError(f"{f}: spatial dims {H}x{W} not divisible by 2")
        # 2x avg pool over spatial dims -> (T, C, H/2, W/2), then permute to
        # spatial-tokens layout (T, H/2*W/2, C).
        x = lat.reshape(T, C, H // 2, 2, W // 2, 2).mean(axis=(3, 5))
        x = np.transpose(x, (0, 2, 3, 1))            # (T, H/2, W/2, C)
        x = x.reshape(T, -1, C)                      # (T, n_tokens, C)
        pooled[ep] = x.astype(np.float32, copy=False)
    if not pooled:
        raise FileNotFoundError(f"no episode_*.npz under {sam2_dir}")
    return pooled


def _load_motion_future(motion_dir: Path):
    fut: dict[int, np.ndarray] = {}
    pad: dict[int, np.ndarray] = {}
    for f in sorted(Path(motion_dir).glob("episode_*.npz")):
        ep = int(f.stem.split("_")[-1])
        with np.load(f) as z:
            fut[ep] = z["motion_future"]
            pad[ep] = z["motion_future_is_pad"].astype(bool)
    if not fut:
        raise FileNotFoundError(f"no episode_*.npz under {motion_dir}")
    return fut, pad


def _aggregate_history(
    actions_window: np.ndarray,           # (frames_total, action_dim); frames before
                                          # episode start are zeros (pad).
    is_pad_window: np.ndarray,            # (frames_total,) bool
    steps: int,
    frames_per_bucket: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate (steps * frames_per_bucket) frames into (steps, action_dim) by
    plain mean over each bucket. Frames before episode start are treated as
    zero actions (their contribution = 0).

    `hist_pad[k]` is True iff EVERY frame in bucket k is a pre-episode pad —
    available to the model as auxiliary info (it's not used to renormalize).
    """
    expected = steps * frames_per_bucket
    if actions_window.shape[0] != expected:
        raise ValueError(
            f"actions_window has {actions_window.shape[0]} frames, expected {expected}"
        )
    A = actions_window.reshape(steps, frames_per_bucket, -1)
    hist = A.mean(axis=1).astype(np.float32, copy=False)  # (steps, action_dim)
    P = is_pad_window.reshape(steps, frames_per_bucket)
    hist_pad = P.all(axis=1)
    return hist, hist_pad


class _MotionV2Mixin:
    """Common: load SAM2 + motion sidecars; gather per-sample tensors."""

    def _init_caches(self, sam2_dir: str, motion_dir: str,
                     action_history_steps: int, frames_per_bucket: int):
        self._sam2 = _load_sam2_tokens(Path(sam2_dir))
        self._motion_future, self._motion_future_pad = _load_motion_future(Path(motion_dir))
        self._history_steps = int(action_history_steps)
        self._frames_per_bucket = int(frames_per_bucket)
        # Infer action_dim from any sample of the base hf_dataset.
        try:
            self._action_dim = int(self.base.hf_dataset[0]["action"].shape[-1])
        except Exception:
            self._action_dim = None

    def _gather_sam2(self, ep_idx: int, frame_idx: int) -> torch.Tensor:
        if ep_idx in self._sam2:
            T = self._sam2[ep_idx].shape[0]
            if 0 <= frame_idx < T:
                arr = self._sam2[ep_idx][frame_idx]
            else:
                arr = self._sam2[ep_idx][max(0, min(T - 1, frame_idx))]
            return torch.from_numpy(arr.copy()).float()
        # Missing episode: zeros matching configured shape (n_tokens, C).
        if self._sam2:
            shape = next(iter(self._sam2.values())).shape[1:]
        else:
            shape = (0, 0)
        return torch.zeros(shape, dtype=torch.float32)

    def _gather_action_history(self, ep_idx: int, frame_idx: int, ep_start: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Past `steps*frames_per_bucket` actions ending at t-1, then bucket-mean."""
        window_size = self._history_steps * self._frames_per_bucket
        start = frame_idx - window_size  # may be negative
        # Gather raw frames.
        action_dim = self._action_dim or 0
        actions = np.zeros((window_size, action_dim), dtype=np.float32)
        is_pad = np.ones(window_size, dtype=bool)
        for j in range(window_size):
            f = start + j
            ds_idx = ep_start + f
            if f < 0:
                continue
            try:
                a = self.base.hf_dataset[int(ds_idx)]["action"]
                if hasattr(a, "numpy"):
                    a = a.numpy()
                actions[j] = a
                is_pad[j] = False
            except Exception:
                # treat as pad
                continue
        hist, hist_pad = _aggregate_history(
            actions, is_pad, self._history_steps, self._frames_per_bucket
        )
        return (
            torch.from_numpy(hist).float(),
            torch.from_numpy(hist_pad).bool(),
        )


class MotionV2WrappedDataset(_MotionV2Mixin, Dataset):
    """Wrap a regular (non-shared) VLASHDataset."""

    def __init__(
        self,
        base_dataset,
        motion_dir: str,
        sam2_dir: str,
        action_history_steps: int = 10,
        frames_per_bucket: int = 30,
    ):
        self.base = base_dataset
        self._init_caches(sam2_dir, motion_dir, action_history_steps, frames_per_bucket)
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

        item[OBS_SAM2_TOKENS] = self._gather_sam2(ep_idx, t)

        if ep_idx in self._motion_future:
            T = self._motion_future[ep_idx].shape[0]
        else:
            T = 0
        if 0 <= t < T:
            mf = self._motion_future[ep_idx][t]
            mf_pad = bool(self._motion_future_pad[ep_idx][t])
        else:
            dim = next(iter(self._motion_future.values())).shape[1] if self._motion_future else 0
            mf = np.zeros(dim, dtype=np.float32)
            mf_pad = True
        item[OBS_MOTION_FUTURE] = torch.from_numpy(mf.copy()).float()
        item[OBS_MOTION_FUTURE_IS_PAD] = torch.tensor(mf_pad, dtype=torch.bool)

        hist, hist_pad = self._gather_action_history(ep_idx, t, ep_start)
        item[OBS_ACTION_HISTORY] = hist
        item[OBS_ACTION_HISTORY_IS_PAD] = hist_pad
        return item


class SharedObservationMotionV2WrappedDataset(_MotionV2Mixin, Dataset):
    """Wrap a SharedObservationVLASHDataset; motion_future is per-offset."""

    def __init__(
        self,
        base_dataset,
        motion_dir: str,
        sam2_dir: str,
        action_history_steps: int = 10,
        frames_per_bucket: int = 30,
    ):
        self.base = base_dataset
        self._init_caches(sam2_dir, motion_dir, action_history_steps, frames_per_bucket)
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

        item[OBS_SAM2_TOKENS] = self._gather_sam2(ep_idx, t)

        # Per-offset motion_future at frames t..t+n-1
        T = self._motion_future[ep_idx].shape[0] if ep_idx in self._motion_future else 0
        if self._motion_future:
            mf_dim = next(iter(self._motion_future.values())).shape[1]
        else:
            mf_dim = 0
        future = np.zeros((n, mf_dim), dtype=np.float32)
        future_pad = np.ones(n, dtype=bool)
        if ep_idx in self._motion_future:
            for k in range(n):
                f = t + k
                if 0 <= f < T:
                    future[k] = self._motion_future[ep_idx][f]
                    future_pad[k] = bool(self._motion_future_pad[ep_idx][f])
        item[OBS_MOTION_FUTURE] = torch.from_numpy(future).float()
        item[OBS_MOTION_FUTURE_IS_PAD] = torch.from_numpy(future_pad).bool()

        hist, hist_pad = self._gather_action_history(ep_idx, t, ep_start)
        item[OBS_ACTION_HISTORY] = hist
        item[OBS_ACTION_HISTORY_IS_PAD] = hist_pad
        return item
