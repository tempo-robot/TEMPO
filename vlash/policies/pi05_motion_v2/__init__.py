"""VLASH PI05-Motion-V2 policy.

Extends pi05_motion with richer conditioning:
- 4096-d SAM2 mem-attn latent at the observation frame (2x-pooled, flat)
  as adaRMS residual on the action expert (replaces v1 motion_past path).
- Past 10 seconds of actions aggregated to 1/sec (=10 vectors), fed in two
  ways: (a) projected per-step into the VLM prefix as 10 extra tokens,
  (b) flattened + projected as adaRMS residual on the action expert.

The output (256-d motion_future = 8 suffix positions x 32-d) is unchanged.
"""

from .configuration_pi05_motion_v2 import PI05MotionV2Config
from .modeling_pi05_motion_v2 import PI05MotionV2Policy

__all__ = ["PI05MotionV2Config", "PI05MotionV2Policy"]
