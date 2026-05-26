"""pi05_motion_v3: model + policy.

Reuses every v2 conditioning module (SAM2 prefix tokens, action-history
prefix tokens, action-history adaRMS MLP). The only difference is that the
auxiliary motion-future flow-matching target is removed:
  * chunk_size == n_action_steps (no suffix extension)
  * forward/forward_shared_observation do not require observation.motion_future
  * loss = action-MSE only, no motion_loss

This lets us cleanly ablate whether the v1/v2 motion-prediction signal was
the part driving any gain.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from lerobot.utils.constants import ACTION, OBS_STATE

from vlash.policies.pi05.modeling_pi05 import (
    PI05Policy,
    pad_vector,
)
from vlash.policies.pi05_motion_v2.modeling_pi05_motion_v2 import PI05MotionV2Model
from vlash.policies.pi05_motion_v3.configuration_pi05_motion_v3 import PI05MotionV3Config


# Re-export the v2 obs keys so callers can identify the same conditioning inputs.
from vlash.policies.pi05_motion_v2.modeling_pi05_motion_v2 import (  # noqa: F401
    OBS_SAM2_TOKENS,
    OBS_ACTION_HISTORY,
    OBS_ACTION_HISTORY_IS_PAD,
)


class PI05MotionV3Model(PI05MotionV2Model):
    """Same module set + masking helpers as V2. With motion_steps=0 the
    parent's __init__ does not extend chunk_size (ext = chunk_size + 0)."""

    pass


class PI05MotionV3Policy(PI05Policy):
    """V2-style conditioning, V1/V2's motion-future auxiliary target removed."""

    config_class = PI05MotionV3Config
    name = "pi05_motion_v3"

    def __init__(self, config: PI05MotionV3Config, dataset_stats=None):
        super().__init__(config, dataset_stats=dataset_stats)
        self.model = PI05MotionV3Model(config)
        self._n_action_steps = config.n_action_steps

    @staticmethod
    def _extract_v2(batch: dict):
        return (
            batch.get(OBS_SAM2_TOKENS, None),
            batch.get(OBS_ACTION_HISTORY, None),
            batch.get(OBS_ACTION_HISTORY_IS_PAD, None),
        )

    def forward(self, batch: dict, noise=None, time=None):
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")

        sam2_tokens, action_history, action_history_is_pad = self._extract_v2(batch)

        losses = self.model(
            images, img_masks, lang_tokens, lang_masks,
            state, actions, noise=noise, time=time,
            sam2_tokens=sam2_tokens,
            action_history=action_history,
            action_history_is_pad=action_history_is_pad,
        )

        cfg = self.config
        action_dim = cfg.output_features[ACTION].shape[0]
        action_losses = losses[:, : cfg.n_action_steps, : action_dim]
        if actions_is_pad is not None:
            action_losses = action_losses * (~actions_is_pad).unsqueeze(-1)
        loss = action_losses.mean()
        return loss, {"loss": loss.item(), "action_loss": loss.item()}

    def forward_shared_observation(self, batch: dict, noise=None, time=None):
        cfg = self.config
        offset_mask = batch["offset_mask"]
        batch_size, num_offsets = offset_mask.shape

        states = batch[OBS_STATE]
        states_flat = states.view(batch_size * num_offsets, -1)
        state_batch = self.normalize_inputs({OBS_STATE: states_flat})
        states_normalized = state_batch[OBS_STATE].view(batch_size, num_offsets, -1)
        states_normalized = pad_vector(states_normalized, cfg.max_state_dim)

        actions = batch[ACTION]
        orig_shape = actions.shape
        actions_flat = actions.view(batch_size * num_offsets * orig_shape[2], -1)
        action_batch = self.normalize_targets({ACTION: actions_flat})
        actions_normalized = action_batch[ACTION].view(orig_shape)
        actions_normalized = pad_vector(actions_normalized, cfg.max_action_dim)

        sam2_tokens, action_history, action_history_is_pad = self._extract_v2(batch)
        images, img_masks = self.prepare_images(batch)
        if not cfg.state_cond:
            raise ValueError("state_cond must be True for shared observation training")
        lang_tokens, lang_masks = self.prepare_language(batch)

        losses = self.model.forward_shared_observation(
            images, img_masks, lang_tokens, lang_masks,
            states_normalized, actions_normalized, offset_mask,
            noise=noise, time=time,
            sam2_tokens=sam2_tokens,
            action_history=action_history,
            action_history_is_pad=action_history_is_pad,
        )

        n_action = cfg.n_action_steps
        action_dim = cfg.output_features[ACTION].shape[0]

        action_losses = losses[:, :, :n_action, :action_dim]
        actions_is_pad = batch.get("action_is_pad")
        if actions_is_pad is not None:
            action_losses = action_losses * (~actions_is_pad).unsqueeze(-1)
        action_losses = action_losses * offset_mask[:, :, None, None]
        num_valid = offset_mask.sum().clamp(min=1)
        denom = (num_valid * n_action * action_dim).to(action_losses.dtype)
        loss = action_losses.sum() / denom

        return loss, {
            "loss": loss.item(),
            "action_loss": loss.item(),
            "num_offsets": num_offsets,
            "avg_valid_offsets": offset_mask.float().sum(dim=1).mean().item(),
        }

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict, noise: Optional[Tensor] = None) -> Tensor:
        self.eval()
        batch = self.normalize_inputs(batch)
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch, pad_to_max_length=False)
        sam2_tokens, action_history, action_history_is_pad = self._extract_v2(batch)
        actions = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise,
            sam2_tokens=sam2_tokens,
            action_history=action_history,
            action_history_is_pad=action_history_is_pad,
        )
        cfg = self.config
        original_action_dim = cfg.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]
        actions = self.unnormalize_outputs({ACTION: actions})[ACTION]
        return actions[:, : cfg.n_action_steps]
