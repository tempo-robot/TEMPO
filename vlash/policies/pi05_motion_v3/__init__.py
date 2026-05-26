"""pi05_motion_v3: like pi05_motion_v2 but without the motion-target output.

Keeps the V2 conditioning paths (SAM2 spatial tokens to VLM prefix, action
history to VLM prefix + adaRMS) and drops the 256-d motion_future
flow-matching auxiliary target. The action expert predicts actions only.
"""

from .configuration_pi05_motion_v3 import PI05MotionV3Config
from .modeling_pi05_motion_v3 import PI05MotionV3Policy

__all__ = ["PI05MotionV3Config", "PI05MotionV3Policy"]
