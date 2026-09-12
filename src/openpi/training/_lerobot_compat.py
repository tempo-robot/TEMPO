"""Compatibility shims for loading our locally-recorded LeRobot v2.1 YAM datasets.

1. Alias the HF `datasets` `List` feature to `Sequence` (the v3.x equivalent),
   because newer `datasets` versions write `"_type": "List"` into parquet metadata.
2. When loading `episodes_stats.jsonl`, wrap any scalar min/max/mean/std for
   auxiliary fields (timestamp, frame_index, ...) in a 1-element array, since
   lerobot's `_assert_type_and_shape` requires `ndim >= 1`. Some of our datasets
   recorded these as bare floats instead of `[float]`.
"""

import numpy as np
from datasets.features.features import _FEATURE_TYPES, Sequence

if "List" not in _FEATURE_TYPES:
    _FEATURE_TYPES["List"] = Sequence


def _patch_lerobot_episodes_stats() -> None:
    from lerobot.common.datasets import utils as _ds_utils

    _orig_load = _ds_utils.load_episodes_stats

    def _patched_load(local_dir):
        eps = _orig_load(local_dir)
        for _ep_idx, ep in eps.items():
            stats = ep.get("stats", ep)
            for _fkey, fstats in stats.items():
                for k, v in list(fstats.items()):
                    if isinstance(v, np.ndarray) and v.ndim == 0:
                        fstats[k] = v.reshape(1)
        return eps

    _ds_utils.load_episodes_stats = _patched_load


_patch_lerobot_episodes_stats()
del _patch_lerobot_episodes_stats
