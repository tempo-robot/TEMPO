"""Policy transforms for YAM (yam_dual_arm) LeRobot datasets.

Supports both bimanual (14D = 7 left + 7 right) and single-arm (7D = 6 joints + 1 gripper)
variants, with up to three cameras: head (third-person), left_wrist, right_wrist.
Missing wrist cameras are zero-padded and masked off for PI0/PI05.
"""

import dataclasses
import pathlib
from typing import ClassVar

import einops
import numpy as np
from collections.abc import Sequence

from openpi import transforms
from openpi.models import model as _model


# Module-level per-cache-dir store of SAM2 memory-attention tokens, loaded lazily once per
# worker process. Maps cache_dir -> {episode_index: (T, n_tokens, dim) float32}.
_SAM2_CACHES: dict[str, dict[int, np.ndarray]] = {}


def _get_sam2_cache(cache_dir: str) -> dict[int, np.ndarray]:
    cache = _SAM2_CACHES.get(cache_dir)
    if cache is None:
        cache = {}
        for f in sorted(pathlib.Path(cache_dir).glob("episode_*.npz")):
            ep = int(f.stem.split("_")[-1])
            with np.load(f) as z:
                lat = z["latent"]  # (T, C, H, W) raw SAM2 memory-attn feature
            if lat.ndim == 4:
                t, c, h, w = lat.shape
                # (T, C, H, W) -> (T, H*W, C): spatial tokens straight out of memory
                # attention (no diff, no pool).
                lat = np.transpose(lat, (0, 2, 3, 1)).reshape(t, h * w, c)
            cache[ep] = np.ascontiguousarray(lat, dtype=np.float32)
        if not cache:
            raise FileNotFoundError(f"no episode_*.npz under {cache_dir}")
        _SAM2_CACHES[cache_dir] = cache
    return cache


@dataclasses.dataclass(frozen=True)
class LoadSam2Tokens(transforms.DataTransformFn):
    """Attach the current frame's SAM2 memory-attention tokens to a raw LeRobot sample.

    Prepended to the YAM repack pipeline so it runs while `episode_index`/`frame_index`
    are still present (RepackTransform / YamInputs would otherwise drop them). Adds
    `data["sam2_tokens"]` of shape (n_tokens, dim); the model fuses it into the head cam.
    """

    cache_dir: str

    def __call__(self, data: dict) -> dict:
        cache = _get_sam2_cache(self.cache_dir)
        ep = int(np.asarray(data["episode_index"]).item())
        frame = int(np.asarray(data["frame_index"]).item())
        arr = cache.get(ep)
        if arr is None:
            tok = np.zeros(next(iter(cache.values())).shape[1:], dtype=np.float32)
        else:
            t = arr.shape[0]
            tok = arr[max(0, min(t - 1, frame))]
        return {**data, "sam2_tokens": np.asarray(tok, dtype=np.float32)}


# ---------------------------------------------------------------------------------------------
# TEMPO-ACT: compact past-action history. Reuses the per-episode SAM2 cache npz, which already
# carries the raw "action" (T, A) column straight from the LeRobot parquet. Loaded lazily once
# per worker process.
_ACTION_HISTORY_CACHES: dict[str, dict[int, np.ndarray]] = {}


def _get_action_history_cache(cache_dir: str) -> dict[int, np.ndarray]:
    cache = _ACTION_HISTORY_CACHES.get(cache_dir)
    if cache is None:
        cache = {}
        for f in sorted(pathlib.Path(cache_dir).glob("episode_*.npz")):
            ep = int(f.stem.split("_")[-1])
            with np.load(f) as z:
                cache[ep] = np.ascontiguousarray(z["action"], dtype=np.float32)  # (T, A) raw
        if not cache:
            raise FileNotFoundError(f"no episode_*.npz under {cache_dir}")
        _ACTION_HISTORY_CACHES[cache_dir] = cache
    return cache


@dataclasses.dataclass(frozen=True)
class LoadActionHistory(transforms.DataTransformFn):
    """Attach bucket-mean past actions to a raw LeRobot sample (TEMPO-ACT).

    Take the `steps * frames_per_bucket` raw frames ending at t-1, mean each bucket, and flag
    buckets that are entirely outside the episode. Frames before the episode start contribute
    zero. Actions stay RAW (unnormalized): the history is a conditioning signal with its own
    learned projection, not something the action-norm stats apply to.

    Prepended to the YAM repack pipeline so it runs while `episode_index`/`frame_index` are
    still present. Adds `action_history` (steps, A) and `action_history_is_pad` (steps,).
    """

    cache_dir: str
    steps: int = 10
    frames_per_bucket: int = 30

    def __call__(self, data: dict) -> dict:
        cache = _get_action_history_cache(self.cache_dir)
        ep = int(np.asarray(data["episode_index"]).item())
        frame = int(np.asarray(data["frame_index"]).item())
        arr = cache.get(ep)
        if arr is None:  # episode missing from the cache: all-pad history
            arr = np.zeros((0, next(iter(cache.values())).shape[1]), dtype=np.float32)
        window = self.steps * self.frames_per_bucket
        idx = np.arange(frame - window, frame)                 # may run negative
        valid = (idx >= 0) & (idx < arr.shape[0])
        actions = np.zeros((window, arr.shape[1]), dtype=np.float32)
        actions[valid] = arr[idx[valid]]
        hist = actions.reshape(self.steps, self.frames_per_bucket, -1).mean(axis=1)
        hist_pad = (~valid).reshape(self.steps, self.frames_per_bucket).all(axis=1)
        return {
            **data,
            "action_history": hist.astype(np.float32),
            "action_history_is_pad": hist_pad,
        }


# ---------------------------------------------------------------------------------------------
# u_t targets: per-episode mu_v sidecars from tools/ut/encode_ut_windows.py. One vector per
# window START frame, so entry t is the latent of the observation window [t, t+W) -- the same
# window the action chunk at t covers. Loaded lazily once per worker process.
_UT_CACHES: dict[str, dict[int, np.ndarray]] = {}


def _get_ut_cache(cache_dir: str) -> dict[int, np.ndarray]:
    cache = _UT_CACHES.get(cache_dir)
    if cache is None:
        cache = {}
        for f in sorted(pathlib.Path(cache_dir).glob("episode_*.npz")):
            ep = int(f.stem.split("_")[-1])
            with np.load(f) as z:
                cache[ep] = np.ascontiguousarray(z["u_window"], dtype=np.float32)  # (N_win, ut_dim)
        if not cache:
            raise FileNotFoundError(f"no episode_*.npz under {cache_dir}")
        _UT_CACHES[cache_dir] = cache
    return cache


@dataclasses.dataclass(frozen=True)
class LoadUtTargets(transforms.DataTransformFn):
    """Attach the frame's u_t target (the MVAE latent of its observation window).

    Prepended to the YAM repack pipeline so it runs while `episode_index`/`frame_index` are
    still present. Adds `ut_target` (ut_dim,) and `ut_target_is_pad` (): the last W-1 frames of
    an episode have no full window, so they get a zero target and pad=True, and the model masks
    their flow loss rather than regressing them to zero.
    """

    cache_dir: str

    def __call__(self, data: dict) -> dict:
        cache = _get_ut_cache(self.cache_dir)
        ep = int(np.asarray(data["episode_index"]).item())
        frame = int(np.asarray(data["frame_index"]).item())
        arr = cache.get(ep)
        dim = next(iter(cache.values())).shape[1]
        if arr is None or frame >= arr.shape[0]:
            return {**data, "ut_target": np.zeros(dim, np.float32), "ut_target_is_pad": np.bool_(True)}
        return {
            **data,
            "ut_target": np.asarray(arr[frame], dtype=np.float32),
            "ut_target_is_pad": np.bool_(False),
        }



def _undistort_crop_hwc_uint8(img, k1, l, r, t, b):
    """Fisheye (Brown radial, focal==width) undistort + margin crop on HWC uint8.

    The same math at train, eval and deploy time so all three match
    across codebases.
    """
    import cv2

    H, W = img.shape[:2]
    K = np.array([[W, 0, W / 2.0], [0, W, H / 2.0], [0, 0, 1.0]], dtype=np.float64)
    D = np.array([k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    und = cv2.undistort(np.ascontiguousarray(img), K, D, None, K)
    y0, y1 = int(round(t * H)), H - int(round(b * H))
    x0, x1 = int(round(l * W)), W - int(round(r * W))
    return und[y0:y1, x0:x1]


def _apply_fisheye(arr, k1, lrtb):
    """Apply undistort+crop to an image array, preserving layout/dtype/range.

    Accepts a single frame ([C,H,W] or [H,W,C]) or a K-stacked history
    ([K,C,H,W] or [K,H,W,C]), float in [0,1] or uint8. Routes every frame through
    the HWC-uint8 op (round-tripping floats through uint8) for exact parity.
    """
    a = np.asarray(arr)
    is_float = np.issubdtype(a.dtype, np.floating)

    def one(frame):
        chw = frame.ndim == 3 and frame.shape[0] == 3
        hwc = np.transpose(frame, (1, 2, 0)) if chw else frame
        if is_float:
            hwc = (np.clip(hwc, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        out = _undistort_crop_hwc_uint8(hwc, k1, *lrtb)  # HWC uint8
        if chw:
            out = np.transpose(out, (2, 0, 1))
        return (out.astype(np.float32) / 255.0) if is_float else out

    if a.ndim == 4:  # K-stacked history
        return np.stack([one(a[i]) for i in range(a.shape[0])], axis=0)
    return one(a)


@dataclasses.dataclass(frozen=True)
class FisheyeUndistortCrop(transforms.DataTransformFn):
    """Undistort + center-bottom crop the raw fisheye camera(s) on a LeRobot sample.

    Prepended to the YAM repack pipeline (runs on `observation.images.{cam}` before
    RepackTransform renames them). Handles TEMPO-MOT's K-stacked history frames too.
    """

    cameras: tuple[str, ...] = ("head",)
    k1: float = -0.6
    lrtb: tuple[float, float, float, float] = (0.18, 0.18, 0.34, 0.0)

    def __call__(self, data: dict) -> dict:
        out = dict(data)
        for cam in self.cameras:
            key = f"observation.images.{cam}"
            if key in out:
                out[key] = _apply_fisheye(out[key], self.k1, self.lrtb)
        return out


def make_yam_example(state_dim: int = 14, cameras: tuple[str, ...] = ("head", "left_wrist", "right_wrist")) -> dict:
    images = {c: np.random.randint(256, size=(224, 224, 3), dtype=np.uint8) for c in cameras}
    return {
        "state": np.zeros((state_dim,), dtype=np.float32),
        "images": images,
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    # TEMPO-MOT visual memory: a stack of K history frames arrives as (K, C, H, W) (LeRobot
    # delta_timestamps) -> convert each frame to H W C, keeping the leading time axis.
    elif image.ndim == 4 and image.shape[1] == 3:
        image = einops.rearrange(image, "k c h w -> k h w c")
    return image


@dataclasses.dataclass(frozen=True)
class YamInputs(transforms.DataTransformFn):
    """Pack YAM state/images/actions into the canonical openpi model inputs.

    State + actions get passed through as-is (padding to the model action_dim is handled
    downstream by PadStatesAndActions). Cameras get mapped to base/left_wrist/right_wrist
    slots; any missing camera is zero-filled with image_mask=False (PI0/PI05).
    """

    model_type: _model.ModelType

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("head", "left_wrist", "right_wrist")

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]
        unexpected = set(in_images) - set(self.EXPECTED_CAMERAS)
        if unexpected:
            raise ValueError(f"Unexpected cameras {unexpected}, expected subset of {self.EXPECTED_CAMERAS}")
        if "head" not in in_images:
            raise ValueError("YAM datasets must have a 'head' camera as the base view")

        head_image = _parse_image(in_images["head"])

        images = {"base_0_rgb": head_image}
        image_masks = {"base_0_rgb": np.True_}

        for dest, source in (("left_wrist_0_rgb", "left_wrist"), ("right_wrist_0_rgb", "right_wrist")):
            if source in in_images:
                images[dest] = _parse_image(in_images[source])
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(head_image)
                image_masks[dest] = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": np.asarray(data["state"], dtype=np.float32),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        # SAM2 motion-fusion tokens (attached upstream by LoadSam2Tokens); forward as-is.
        if "sam2_tokens" in data:
            inputs["sam2_tokens"] = np.asarray(data["sam2_tokens"], dtype=np.float32)
        # TEMPO-ACT past-action history (attached upstream by LoadActionHistory); raw units.
        if "action_history" in data:
            inputs["action_history"] = np.asarray(data["action_history"], dtype=np.float32)
            inputs["action_history_is_pad"] = np.asarray(data["action_history_is_pad"], dtype=bool)
        # u_t prediction target (attached upstream by LoadUtTargets); already in MVAE space.
        if "ut_target" in data:
            inputs["ut_target"] = np.asarray(data["ut_target"], dtype=np.float32)
            inputs["ut_target_is_pad"] = np.asarray(data["ut_target_is_pad"], dtype=bool)

        return inputs


@dataclasses.dataclass(frozen=True)
class YamOutputs(transforms.DataTransformFn):
    """Slice off action padding to recover the dataset's native action dimension."""

    action_dim: int = 14

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
