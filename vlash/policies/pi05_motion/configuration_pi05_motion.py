"""PI05-Motion config: pi05 + motion auxiliary input/output.

Key knob: the action chunk is extended from `n_action_steps` action steps to
`n_action_steps + n_motion_steps` total chunk positions. The motion vector
(256-d) is reshaped into n_motion_steps × max_action_dim. With pi05's default
max_action_dim=32 and motion_dim=256, we get n_motion_steps = 8.
"""

from __future__ import annotations

from dataclasses import dataclass

from vlash.policies.pi05.configuration_pi05 import PI05Config


@dataclass
class PI05MotionConfig(PI05Config):
    # Motion target dim (= MVAE-free "diff_first_last" SAM2 feature dim).
    motion_dim: int = 256
    # Number of suffix steps reserved for motion (motion_dim must equal
    # motion_steps × max_action_dim so the reshape is exact).
    motion_steps: int = 8
    # Weight on the motion flow-matching MSE relative to the action FM MSE
    # (both contribute as the unweighted mean of per-position MSE).
    motion_loss_weight: float = 1.0

    def __post_init__(self):
        # PI05Config has no __post_init__; we just guard config sanity here.
        assert self.motion_dim == self.motion_steps * self.max_action_dim, (
            f"motion_dim ({self.motion_dim}) must equal motion_steps "
            f"({self.motion_steps}) * max_action_dim ({self.max_action_dim})"
        )

    @property
    def action_delta_indices(self) -> list:
        # The dataset must query only the n_action_steps real action frames.
        # The extra motion_steps positions are appended in-model, not via the
        # action loader. Even after the model mutates self.chunk_size to the
        # extended length, this property must still return the action-only
        # range; using n_action_steps (not chunk_size) makes that explicit.
        return list(range(self.n_action_steps))
