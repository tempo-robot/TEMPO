"""PI05-Motion: pi05 + motion auxiliary input + joint action/motion target.

Design:
- The action expert's suffix is extended by `motion_steps` extra "steps".
  The motion target (256-d) is reshaped to (motion_steps, max_action_dim) =
  (8, 32) and concatenated after the 50 action steps. So flow matching runs
  end-to-end on a 58-step chunk with the existing action_in_proj /
  action_out_proj — no per-stream projections required.
- Loss has a position mask: action positions [0..49] use action_dim=14, motion
  positions [50..57] use all 32 dims.
- Motion past (256-d) is fed in as a zero-init residual on adarms_cond, same
  pattern as PI05-UVT.
- At inference, predict_action_chunk runs the same sample_actions loop and
  returns the action portion as a (B, n_action_steps, action_dim) tensor;
  predict_motion_chunk slices out the motion portion and reshapes to
  (B, motion_dim).

Batch keys consumed:
  "observation.motion_past"        — (B, 256) float, the past-window diff
  "observation.motion_future"      — (B, 256) float, the future-window diff (target)
  "observation.motion_future_is_pad" — (B,) bool, optional
"""

from __future__ import annotations

import builtins
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lerobot.utils.constants import ACTION, OBS_STATE

from vlash.policies.pi05.modeling_pi05 import (
    PI05Model,
    PI05Policy,
    build_attention_mask_and_position_ids,
    pad_vector,
)
from vlash.policies.pi05.utils import build_shared_obs_attention_mask_and_position_ids
from vlash.policies.pi05_motion.configuration_pi05_motion import PI05MotionConfig


OBS_MOTION_PAST = "observation.motion_past"
OBS_MOTION_FUTURE = "observation.motion_future"
OBS_MOTION_FUTURE_IS_PAD = "observation.motion_future_is_pad"


class _MotionCondPath(nn.Module):
    """motion_past (B, 256) -> additive cond vector in action-expert hidden dim."""
    def __init__(self, motion_dim: int, hidden_size: int):
        super().__init__()
        self.in_proj = nn.Linear(motion_dim, hidden_size)
        self.mlp_in = nn.Linear(hidden_size, hidden_size)
        self.mlp_out = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.mlp_out.weight)
        nn.init.zeros_(self.mlp_out.bias)

    def forward(self, motion_past: Tensor) -> Tensor:
        h = self.in_proj(motion_past)
        h = F.silu(self.mlp_in(h))
        h = F.silu(self.mlp_out(h))
        return h


class PI05MotionModel(PI05Model):
    """PI05Model with the chunk extended by `motion_steps` and a motion-past
    conditioning path."""

    def __init__(self, config: PI05MotionConfig):
        # Hack: pi05's __init__ sizes action_expert position embeddings / etc.
        # based on config.chunk_size. We want the *extended* chunk length to be
        # what pi05 sees, so swap chunk_size before delegating then restore the
        # 'action-only' chunk size afterwards.
        self._n_action_steps_orig = config.chunk_size
        self._motion_steps = config.motion_steps
        ext = config.chunk_size + config.motion_steps
        # Mutate temporarily for parent init.
        config.chunk_size = ext
        super().__init__(config)
        # Note: we deliberately leave config.chunk_size = extended length so
        # the rest of pi05 (e.g. sample_actions noise shape) uses it.

        h = config.action_expert_config.hidden_size
        self.motion_cond_path = _MotionCondPath(config.motion_dim, h)

    def _motion_cond(self, motion_past: Tensor | None, base_cond: Tensor) -> Tensor:
        if motion_past is None:
            return base_cond
        u = motion_past.to(dtype=self.motion_cond_path.in_proj.weight.dtype)
        res = self.motion_cond_path(u).to(dtype=base_cond.dtype)
        return base_cond + res

    def forward(self, images, img_masks, tokens, masks, state, actions,
                noise=None, time=None, motion_past: Tensor | None = None):
        """Replicates PI05Model.forward but adds motion_past into adarms_cond.
        `actions` is already the EXTENDED chunk (chunk_size = n_action_steps
        + motion_steps), built by the policy."""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.prefix_embedder(
            images, img_masks, tokens, masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, suffix_adarms_cond = self.suffix_embedder(
            state, x_t, time
        )
        suffix_adarms_cond = self._motion_cond(motion_past, suffix_adarms_cond)

        backbone_dtype = self.vlm.model.language_model.layers[0].input_layernorm.weight.dtype
        prefix_embs = prefix_embs.to(dtype=backbone_dtype)
        suffix_embs = suffix_embs.to(dtype=backbone_dtype)

        attention_mask, position_ids = build_attention_mask_and_position_ids(
            torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1),
            torch.cat([prefix_att_masks, suffix_att_masks], dim=1),
            prefix_embs.dtype,
        )

        hidden_states = [prefix_embs, suffix_embs]
        conds = [None, suffix_adarms_cond]
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids,
                                   conds, use_cache=False)

        norms = [self.vlm.language_model.norm, self.action_expert.model.norm]
        final_hidden_states: list[Tensor | None] = []
        for i, hs in enumerate(hidden_states):
            if hs is None:
                final_hidden_states.append(None)
                continue
            hs, _ = norms[i](hs, cond=conds[i])
            final_hidden_states.append(hs)
        hidden_states = final_hidden_states

        suffix_out = hidden_states[1][:, -self.config.chunk_size:]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        v_t = self.action_out_proj(suffix_out)        # (B, chunk_size, max_action_dim)

        # Per-element MSE losses (caller will mask).
        losses = F.mse_loss(v_t, u_t, reduction="none")
        return losses

    def forward_shared_observation(
        self,
        images,
        img_masks,
        tokens,
        masks,
        states,
        actions,
        offset_mask,
        noise=None,
        time=None,
        motion_past: Tensor | None = None,
    ):
        """Shared-observation forward with motion conditioning.

        Mirrors `PI05Model.forward_shared_observation` but routes
        `motion_past` (shape [B, motion_dim]) into the suffix_adarms_cond as
        a zero-init residual, broadcast across all offsets per sample.

        `actions` is the already-extended chunk (chunk_size = n_action_steps
        + motion_steps); shape [B, num_offsets, chunk_size_ext, max_action_dim].
        """
        batch_size = states.shape[0]
        num_offsets = states.shape[1]

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(batch_size * num_offsets, actions.device)
            time = time.view(batch_size, num_offsets)

        time_expanded = time[:, :, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.prefix_embedder(
            images, img_masks, tokens, masks
        )

        states_flat = states.view(batch_size * num_offsets, -1)
        x_t_flat = x_t.view(batch_size * num_offsets, x_t.shape[2], -1)
        time_flat = time.view(batch_size * num_offsets)

        suffix_embs_flat, suffix_pad_masks_flat, suffix_att_masks_flat, suffix_adarms_cond_flat = self.suffix_embedder(
            states_flat, x_t_flat, time_flat
        )
        suffix_length = suffix_embs_flat.shape[1]

        # Motion-past residual on adarms_cond, broadcast across offsets.
        if motion_past is not None and suffix_adarms_cond_flat is not None:
            mp = motion_past.to(dtype=self.motion_cond_path.in_proj.weight.dtype)
            mp_res = self.motion_cond_path(mp).to(dtype=suffix_adarms_cond_flat.dtype)
            # mp_res: [B, hidden] -> [B, num_offsets, hidden] -> [B*num_offsets, hidden]
            mp_res = mp_res.unsqueeze(1).expand(-1, num_offsets, -1).reshape(
                batch_size * num_offsets, -1
            )
            suffix_adarms_cond_flat = suffix_adarms_cond_flat + mp_res

        suffix_pad_masks = suffix_pad_masks_flat[:batch_size]
        suffix_att_masks = suffix_att_masks_flat[:batch_size]

        suffix_embs = suffix_embs_flat.view(batch_size, num_offsets, suffix_length, -1)
        suffix_embs_concat = suffix_embs.view(batch_size, num_offsets * suffix_length, -1)

        suffix_adarms_conds = (
            suffix_adarms_cond_flat.view(batch_size, num_offsets, -1)
            if suffix_adarms_cond_flat is not None else None
        )

        backbone_dtype = self.vlm.model.language_model.layers[0].input_layernorm.weight.dtype
        prefix_embs = prefix_embs.to(dtype=backbone_dtype)
        suffix_embs_concat = suffix_embs_concat.to(dtype=backbone_dtype)

        attention_mask, position_ids = build_shared_obs_attention_mask_and_position_ids(
            prefix_pad_masks=prefix_pad_masks,
            prefix_att_masks=prefix_att_masks,
            suffix_pad_masks=suffix_pad_masks,
            suffix_att_masks=suffix_att_masks,
            num_offsets=num_offsets,
            offset_mask=offset_mask,
            dtype=prefix_embs.dtype,
        )

        hidden_states = [prefix_embs, suffix_embs_concat]
        for layer in self.layers:
            hidden_states = layer.forward_shared_observation(
                hidden_states, attention_mask, position_ids,
                suffix_adarms_conds, num_offsets, suffix_length
            )

        norms = [self.vlm.language_model.norm, self.action_expert.model.norm]
        prefix_out = hidden_states[0]
        prefix_out, _ = norms[0](prefix_out, cond=None)

        suffix_out = hidden_states[1]
        hidden_dim = suffix_out.shape[-1]
        suffix_flat = suffix_out.view(batch_size * num_offsets, suffix_length, hidden_dim)
        cond_flat = (
            suffix_adarms_conds.view(batch_size * num_offsets, -1)
            if suffix_adarms_conds is not None else None
        )
        suffix_normed_flat, _ = norms[1](suffix_flat, cond=cond_flat)

        action_out = suffix_normed_flat.view(batch_size, num_offsets, suffix_length, hidden_dim)
        action_out = action_out.to(dtype=self.action_out_proj.weight.dtype)
        v_t = self.action_out_proj(action_out)  # [B, num_offsets, chunk_size_ext, max_action_dim]

        # Element-wise MSE (caller will mask & average).
        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    @torch.no_grad()
    def denoise_step(self, prefix_pad_masks, prefix_att_masks, state, x_t, timestep,
                     motion_past: Tensor | None = None):
        suffix_embs, suffix_pad_masks, suffix_att_masks, suffix_adarms_cond = self.suffix_embedder(
            state, x_t, timestep
        )
        suffix_adarms_cond = self._motion_cond(motion_past, suffix_adarms_cond)

        backbone_dtype = self.vlm.model.language_model.layers[0].input_layernorm.weight.dtype
        suffix_embs = suffix_embs.to(dtype=backbone_dtype)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        full_attention_mask, full_position_ids = build_attention_mask_and_position_ids(
            pad_masks, att_masks, suffix_embs.dtype
        )
        bsz, L_suf = suffix_embs.shape[:2]
        attention_mask = full_attention_mask[:, :, -L_suf:, :]
        position_ids = full_position_ids[:, -L_suf:]
        hidden_states = [None, suffix_embs]
        conds = [None, suffix_adarms_cond]
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids,
                                   conds, use_cache=True)
        suffix_hidden = hidden_states[1]
        suffix_hidden, _ = self.action_expert.model.norm(suffix_hidden, cond=suffix_adarms_cond)
        suffix_out = suffix_hidden[:, -self.config.chunk_size:]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        return self.action_out_proj(suffix_out)

    @torch.no_grad()
    def sample_actions(self, images, img_masks, tokens, masks, state, noise=None,
                       num_steps=None, motion_past: Tensor | None = None) -> Tensor:
        if num_steps is None:
            num_steps = self.config.num_inference_steps
        bsz = tokens.shape[0]
        device = tokens.device
        if noise is None:
            actions_shape = (bsz, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.prefix_embedder(
            images, img_masks, tokens, masks
        )
        for layer in self.layers:
            layer.self_attn.attn.reset_cache()
        prefix_attention_mask, prefix_position_ids = build_attention_mask_and_position_ids(
            prefix_pad_masks, prefix_att_masks, prefix_embs.dtype
        )
        hidden_states_prefill = [prefix_embs, None]
        conds_prefill = [None, None]
        for layer in self.layers:
            hidden_states_prefill = layer(
                hidden_states_prefill, prefix_attention_mask, prefix_position_ids,
                conds_prefill, use_cache=True
            )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        for _ in range(num_steps):
            expanded_time = time.expand(bsz)
            v_t = self.denoise_step(prefix_pad_masks, prefix_att_masks, state, x_t,
                                     expanded_time, motion_past=motion_past)
            x_t = x_t + dt * v_t
            time = time + dt
        return x_t


class PI05MotionPolicy(PI05Policy):
    """pi05 + motion conditioning + joint action/motion FM target."""

    config_class = PI05MotionConfig
    name = "pi05_motion"

    def __init__(self, config: PI05MotionConfig, dataset_stats=None):
        super().__init__(config, dataset_stats=dataset_stats)
        # Replace the model with the motion-aware variant; pretrained weights
        # for action_in/out_proj are reused (chunk_size only changes the position
        # embedding inputs, which live in the suffix_embedder — these are
        # reseeded by the new __init__).
        self.model = PI05MotionModel(config)
        self._n_action_steps = config.n_action_steps
        self._motion_steps = config.motion_steps
        self._motion_loss_weight = config.motion_loss_weight

    @staticmethod
    def _extract_motion(batch: dict):
        return (batch.get(OBS_MOTION_PAST, None),
                batch.get(OBS_MOTION_FUTURE, None),
                batch.get(OBS_MOTION_FUTURE_IS_PAD, None))

    def forward(self, batch: dict, noise=None, time=None) -> tuple[Tensor, dict[str, Tensor]]:
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        actions = self.prepare_action(batch)                # (B, n_action_steps, max_action_dim)
        actions_is_pad = batch.get("action_is_pad")

        motion_past, motion_future, motion_is_pad = self._extract_motion(batch)

        # Build extended chunk (B, n_action_steps + motion_steps, max_action_dim).
        cfg = self.config
        bsz = actions.shape[0]
        if motion_future is None:
            raise RuntimeError("pi05_motion requires observation.motion_future in batch")
        motion_tokens = motion_future.float().reshape(
            bsz, cfg.motion_steps, cfg.max_action_dim
        ).to(actions.dtype).to(actions.device)
        ext = torch.cat([actions, motion_tokens], dim=1)

        losses = self.model(images, img_masks, lang_tokens, lang_masks,
                             state, ext, noise=noise, time=time,
                             motion_past=motion_past)

        # Position mask + action_is_pad mask.
        # losses: (B, chunk_size_ext, max_action_dim)
        per_pos_mask = torch.ones(ext.shape[1], dtype=losses.dtype, device=losses.device)
        action_dim = cfg.output_features[ACTION].shape[0]
        # Action positions: only count first `action_dim` channels.
        action_losses = losses[:, : cfg.n_action_steps, : action_dim]
        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            action_losses = action_losses * in_episode_bound.unsqueeze(-1)
        action_loss = action_losses.mean()

        # Motion positions: count all `max_action_dim` channels.
        motion_losses = losses[:, cfg.n_action_steps:, :]  # (B, motion_steps, max_action_dim)
        if motion_is_pad is not None:
            valid = (~motion_is_pad.to(motion_losses.device).bool()).float().view(bsz, 1, 1)
            denom = valid.sum().clamp_min(1.0) * motion_losses.shape[1] * motion_losses.shape[2]
            motion_loss = (motion_losses * valid).sum() / denom
        else:
            motion_loss = motion_losses.mean()

        total_loss = action_loss + self._motion_loss_weight * motion_loss
        loss_dict = {
            "loss": total_loss.item(),
            "action_loss": action_loss.item(),
            "motion_loss": motion_loss.item(),
        }
        return total_loss, loss_dict

    def forward_shared_observation(
        self, batch: dict, noise=None, time=None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Motion-aware shared-observation forward.

        Mirrors PI05Policy.forward_shared_observation but:
          - Builds an extended chunk per offset by concatenating the action
            target with motion_future tokens reshaped to (motion_steps, max_action_dim).
          - Applies per-position loss masks: action positions count only the
            real action_dim channels; motion positions count all max_action_dim
            channels (masked by motion_future_is_pad and offset_mask).
          - Threads motion_past into the model as a per-sample adarms residual.
        """
        cfg = self.config
        offset_mask = batch["offset_mask"]  # [B, num_offsets]
        batch_size, num_offsets = offset_mask.shape

        # ---- normalize states (per offset) ----
        states = batch[OBS_STATE]  # [B, num_offsets, state_dim]
        states_flat = states.view(batch_size * num_offsets, -1)
        state_batch = {OBS_STATE: states_flat}
        state_batch = self.normalize_inputs(state_batch)
        states_normalized = state_batch[OBS_STATE].view(batch_size, num_offsets, -1)
        states_normalized = pad_vector(states_normalized, cfg.max_state_dim)

        # ---- normalize actions (per offset) ----
        actions = batch[ACTION]  # [B, num_offsets, n_action_steps, action_dim]
        orig_shape = actions.shape
        actions_flat = actions.view(batch_size * num_offsets * orig_shape[2], -1)
        action_batch = {ACTION: actions_flat}
        action_batch = self.normalize_targets(action_batch)
        actions_normalized = action_batch[ACTION].view(orig_shape)
        actions_normalized = pad_vector(actions_normalized, cfg.max_action_dim)
        # actions_normalized: [B, num_offsets, n_action_steps, max_action_dim]

        # ---- motion: future (per-offset target), past (shared) ----
        motion_future = batch.get(OBS_MOTION_FUTURE, None)
        motion_future_is_pad = batch.get(OBS_MOTION_FUTURE_IS_PAD, None)
        motion_past = batch.get(OBS_MOTION_PAST, None)
        if motion_future is None:
            raise RuntimeError(
                "pi05_motion shared-observation requires "
                "observation.motion_future in batch"
            )
        # Reshape motion target -> [B, num_offsets, motion_steps, max_action_dim]
        motion_tokens = motion_future.float().reshape(
            batch_size, num_offsets, cfg.motion_steps, cfg.max_action_dim
        ).to(actions_normalized.dtype).to(actions_normalized.device)

        # Build extended target chunk: [B, num_offsets, n_action_steps+motion_steps, max_action_dim]
        ext = torch.cat([actions_normalized, motion_tokens], dim=2)

        # ---- prepare shared prefix inputs ----
        images, img_masks = self.prepare_images(batch)
        if not cfg.state_cond:
            raise ValueError(
                "state_cond must be True for shared observation training"
            )
        lang_tokens, lang_masks = self.prepare_language(batch)

        # ---- model forward ----
        losses = self.model.forward_shared_observation(
            images, img_masks, lang_tokens, lang_masks,
            states_normalized, ext, offset_mask,
            noise=noise, time=time, motion_past=motion_past,
        )  # [B, num_offsets, chunk_size_ext, max_action_dim]

        # ---- per-position masking ----
        n_action = cfg.n_action_steps
        action_dim = cfg.output_features[ACTION].shape[0]

        # Action part: [B, num_offsets, n_action_steps, action_dim]
        action_losses = losses[:, :, :n_action, :action_dim]
        actions_is_pad = batch.get("action_is_pad")  # [B, num_offsets, n_action_steps]
        if actions_is_pad is not None:
            action_losses = action_losses * (~actions_is_pad).unsqueeze(-1)
        # Offset-mask + average
        action_losses = action_losses * offset_mask[:, :, None, None]
        num_valid = offset_mask.sum().clamp(min=1)
        action_denom = (num_valid * n_action * action_dim).to(action_losses.dtype)
        action_loss = action_losses.sum() / action_denom

        # Motion part: [B, num_offsets, motion_steps, max_action_dim]
        motion_losses = losses[:, :, n_action:, :]
        if motion_future_is_pad is not None:
            motion_valid = (~motion_future_is_pad.bool()).to(motion_losses.dtype)
            motion_losses = motion_losses * motion_valid[:, :, None, None]
            motion_offset_mask = offset_mask.to(motion_losses.dtype) * motion_valid
        else:
            motion_offset_mask = offset_mask.to(motion_losses.dtype)
        motion_losses = motion_losses * offset_mask[:, :, None, None]
        motion_num_valid = motion_offset_mask.sum().clamp(min=1)
        motion_denom = (
            motion_num_valid * cfg.motion_steps * cfg.max_action_dim
        ).to(motion_losses.dtype)
        motion_loss = motion_losses.sum() / motion_denom

        total_loss = action_loss + self._motion_loss_weight * motion_loss
        loss_dict = {
            "loss": total_loss.item(),
            "action_loss": action_loss.item(),
            "motion_loss": motion_loss.item(),
            "num_offsets": num_offsets,
            "avg_valid_offsets": offset_mask.float().sum(dim=1).mean().item(),
        }
        return total_loss, loss_dict

    @torch.no_grad()
    def predict_chunk(self, batch: dict, noise: Tensor | None = None):
        """Run the full ODE and return (action_chunk, motion_pred).
        action_chunk: (B, n_action_steps, action_dim) in raw units.
        motion_pred:  (B, motion_dim) in normalized space (no unnorm here).
        """
        self.eval()
        batch = self.normalize_inputs(batch)
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch, pad_to_max_length=False)
        motion_past = batch.get(OBS_MOTION_PAST, None)
        ext = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise,
            motion_past=motion_past,
        )  # (B, chunk_size_ext, max_action_dim)
        cfg = self.config
        actions = ext[:, : cfg.n_action_steps]
        original_action_dim = cfg.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]
        actions = self.unnormalize_outputs({ACTION: actions})[ACTION]
        motion_pred = ext[:, cfg.n_action_steps:].reshape(ext.shape[0], cfg.motion_dim)
        return actions, motion_pred

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict, noise: Tensor | None = None) -> Tensor:
        actions, _ = self.predict_chunk(batch, noise=noise)
        return actions
