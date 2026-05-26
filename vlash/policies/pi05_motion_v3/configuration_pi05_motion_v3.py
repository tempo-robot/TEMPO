"""PI05-Motion-V3 config: v2 minus the motion-target output.

`motion_steps=0` and `motion_dim=0` disable the auxiliary motion-future
flow-matching target. The model's chunk size stays equal to n_action_steps
(no suffix extension). All other v2 conditioning paths (SAM2 spatial tokens
to VLM prefix, action history to VLM prefix + adaRMS residual) are kept.
"""

from __future__ import annotations

from dataclasses import dataclass

from vlash.policies.pi05_motion_v2.configuration_pi05_motion_v2 import PI05MotionV2Config


@dataclass
class PI05MotionV3Config(PI05MotionV2Config):
    # No motion-future output target.
    motion_steps: int = 0
    motion_dim: int = 0
    motion_loss_weight: float = 0.0
