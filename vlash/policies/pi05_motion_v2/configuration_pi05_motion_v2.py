"""PI05-Motion-V2 config.

Same flow-matching architecture as pi05_motion v1 (action chunk extended by
motion_steps positions), with three new conditioning inputs:

- sam2_feature_dim:        SAM2 mem-attn latent flattened size at the
                           observation frame (default 4096 from 256x4x4 after
                           2x avg-pool of the raw 256x8x8 cache).
- action_history_steps:    number of aggregated past-action vectors (default 10).
- action_history_seconds:  span of the past-action window in seconds (default 10).
- dataset_fps:             dataset frame rate, used for the n-frames-per-bucket
                           aggregation (default 30).

Aggregation rate is derived: frames_per_bucket = dataset_fps * (seconds / steps).
"""

from __future__ import annotations

from dataclasses import dataclass

from vlash.policies.pi05_motion.configuration_pi05_motion import PI05MotionConfig


@dataclass
class PI05MotionV2Config(PI05MotionConfig):
    # SAM2 mem-attn latent (after 2x avg-pool spatially) is fed to the VLM
    # prefix as `sam2_num_tokens` tokens of `sam2_token_dim` channels each.
    # For raw 8x8 spatial latents at 256 channels this gives 16 tokens of 256-d.
    sam2_num_tokens: int = 16
    sam2_token_dim: int = 256

    # Past-action history sequence length and span.
    action_history_steps: int = 10
    action_history_seconds: int = 10

    # Dataset frame rate for the past-action aggregator (mean over fps frames).
    dataset_fps: int = 30

    @property
    def frames_per_history_bucket(self) -> int:
        return int(round(self.dataset_fps * self.action_history_seconds / self.action_history_steps))
