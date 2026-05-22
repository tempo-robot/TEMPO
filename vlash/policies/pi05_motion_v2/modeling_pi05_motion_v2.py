"""PI05-Motion-V2: pi05_motion + SAM2 spatial tokens + past-action history.

New conditioning routes (relative to v1):
  - SAM2 mem-attn latent (2x-pooled to 16 spatial tokens of 256-d) is projected
    to VLM hidden_size=2048 and appended to the prefix as 16 extra tokens.
    It does NOT feed the action expert directly — the expert sees it via
    cross-attention to the prefix.
  - Past-action history (`action_history_steps` aggregated vectors) is fed BOTH
    to the prefix (projected to vlm_hidden) AND as an additive residual on the
    action-expert adaRMS cond (a zero-init SiLU MLP, so it starts as identity).

Output target unchanged from v1: 256-d motion_future as 8 suffix positions x
max_action_dim, joint with the action positions in a single flow-matching ODE.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lerobot.utils.constants import ACTION, OBS_STATE

from vlash.policies.pi05.modeling_pi05 import (
    build_attention_mask_and_position_ids,
    pad_vector,
)
from vlash.policies.pi05.utils import build_shared_obs_attention_mask_and_position_ids
from vlash.policies.pi05_motion.modeling_pi05_motion import (
    PI05MotionModel,
    PI05MotionPolicy,
    OBS_MOTION_FUTURE,
    OBS_MOTION_FUTURE_IS_PAD,
)
from vlash.policies.pi05_motion_v2.configuration_pi05_motion_v2 import PI05MotionV2Config


OBS_SAM2_TOKENS = "observation.sam2_tokens"
OBS_ACTION_HISTORY = "observation.action_history"
OBS_ACTION_HISTORY_IS_PAD = "observation.action_history_is_pad"


class _CondMLP(nn.Module):
    """in_dim -> hidden -> silu -> hidden -> silu, with zero-init output so
    the module starts as an identity residual contribution."""

    def __init__(self, in_dim: int, hidden_size: int):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_size)
        self.mlp_in = nn.Linear(hidden_size, hidden_size)
        self.mlp_out = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.mlp_out.weight)
        nn.init.zeros_(self.mlp_out.bias)

    def forward(self, x: Tensor) -> Tensor:
        h = self.in_proj(x)
        h = F.silu(self.mlp_in(h))
        h = F.silu(self.mlp_out(h))
        return h


class _PerTokenVLMProjection(nn.Module):
    """Project a sequence of N feature vectors into vlm_hidden_size.

    Shared per-token linear projection (`in_dim -> vlm_hidden`) plus a
    learnable per-position embedding (init zero). Generic enough to embed
    either past-action vectors (action_dim -> 2048) or SAM2 spatial tokens
    (256 -> 2048).
    """

    def __init__(self, in_dim: int, num_tokens: int, vlm_hidden: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, vlm_hidden)
        self.pos_emb = nn.Parameter(torch.zeros(num_tokens, vlm_hidden))

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, N, in_dim) -> (B, N, vlm_hidden)
        return self.proj(x) + self.pos_emb.unsqueeze(0)


class PI05MotionV2Model(PI05MotionModel):
    """pi05_motion model with the three extra conditioning paths.

    Inherits the extended chunk_size handling and motion-target loss layout
    from v1; replaces `motion_cond_path` with three new modules.
    """

    def __init__(self, config: PI05MotionV2Config):
        super().__init__(config)
        # The v1 parent attached `motion_cond_path`; drop it (v2 has no SAM2
        # adaRMS path).
        if hasattr(self, "motion_cond_path"):
            del self.motion_cond_path

        h_ae = config.action_expert_config.hidden_size
        action_dim_raw = int(config.output_features[ACTION].shape[0]) if (
            getattr(config, "output_features", None)
            and ACTION in config.output_features
        ) else config.max_action_dim
        self._action_dim_raw = action_dim_raw
        self._action_history_steps = int(config.action_history_steps)
        self._sam2_num_tokens = int(config.sam2_num_tokens)
        self._sam2_token_dim = int(config.sam2_token_dim)

        # Past-action adaRMS residual (zero-init).
        self.action_history_cond_path = _CondMLP(
            self._action_history_steps * action_dim_raw, h_ae
        )

        # VLM prefix token projections.
        vlm_hidden = int(config.vlm_config.text_config.hidden_size)
        self.sam2_vlm_tokens = _PerTokenVLMProjection(
            self._sam2_token_dim, self._sam2_num_tokens, vlm_hidden
        )
        self.action_history_vlm_tokens = _PerTokenVLMProjection(
            action_dim_raw, self._action_history_steps, vlm_hidden
        )

    # ---------- helpers ----------

    def _v2_cond_residual(
        self,
        action_history: Optional[Tensor],
        target_dtype: torch.dtype,
    ) -> Optional[Tensor]:
        """Build a single (B, h) adaRMS residual from action_history."""
        if action_history is None:
            return None
        bsz = action_history.shape[0]
        flat = action_history.reshape(bsz, -1).to(
            dtype=self.action_history_cond_path.in_proj.weight.dtype
        )
        return self.action_history_cond_path(flat).to(dtype=target_dtype)

    def _append_v2_prefix(
        self,
        prefix_embs: Tensor,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
        sam2_tokens: Optional[Tensor],
        action_history: Optional[Tensor],
        action_history_is_pad: Optional[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Append SAM2 spatial tokens and past-action tokens to the prefix.

        Order at the end of the prefix: [..., sam2_tokens, action_history_tokens].
        att_masks=0 (continue existing block, no new attention boundary).
        pad_masks: SAM2 tokens are always valid; action-history tokens use
        action_history_is_pad to mask off pre-episode buckets.
        """
        device = prefix_embs.device
        out_dtype = prefix_embs.dtype

        if sam2_tokens is not None:
            bsz, n_sam = sam2_tokens.shape[0], sam2_tokens.shape[1]
            embs = self.sam2_vlm_tokens(
                sam2_tokens.to(dtype=self.sam2_vlm_tokens.proj.weight.dtype)
            ).to(dtype=out_dtype)
            pad = torch.ones(bsz, n_sam, dtype=torch.bool, device=device)
            att = torch.zeros(bsz, n_sam, dtype=prefix_att_masks.dtype, device=device)
            prefix_embs = torch.cat([prefix_embs, embs], dim=1)
            prefix_pad_masks = torch.cat([prefix_pad_masks, pad], dim=1)
            prefix_att_masks = torch.cat([prefix_att_masks, att], dim=1)

        if action_history is not None:
            bsz, n_hist = action_history.shape[0], action_history.shape[1]
            embs = self.action_history_vlm_tokens(
                action_history.to(dtype=self.action_history_vlm_tokens.proj.weight.dtype)
            ).to(dtype=out_dtype)
            if action_history_is_pad is None:
                pad = torch.ones(bsz, n_hist, dtype=torch.bool, device=device)
            else:
                pad = (~action_history_is_pad.bool()).to(device=device)
            att = torch.zeros(bsz, n_hist, dtype=prefix_att_masks.dtype, device=device)
            prefix_embs = torch.cat([prefix_embs, embs], dim=1)
            prefix_pad_masks = torch.cat([prefix_pad_masks, pad], dim=1)
            prefix_att_masks = torch.cat([prefix_att_masks, att], dim=1)

        return prefix_embs, prefix_pad_masks, prefix_att_masks

    # ---------- forwards ----------

    def forward(
        self,
        images,
        img_masks,
        tokens,
        masks,
        state,
        actions,
        noise=None,
        time=None,
        motion_past: Optional[Tensor] = None,  # v1 compat, ignored in v2
        sam2_tokens: Optional[Tensor] = None,
        action_history: Optional[Tensor] = None,
        action_history_is_pad: Optional[Tensor] = None,
    ):
        """Single-offset forward (v1-style, but with v2 conditioning).

        `actions` is the EXTENDED chunk (n_action_steps + motion_steps).
        """
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
        prefix_embs, prefix_pad_masks, prefix_att_masks = self._append_v2_prefix(
            prefix_embs, prefix_pad_masks, prefix_att_masks,
            sam2_tokens, action_history, action_history_is_pad,
        )

        suffix_embs, suffix_pad_masks, suffix_att_masks, suffix_adarms_cond = self.suffix_embedder(
            state, x_t, time
        )
        res = self._v2_cond_residual(action_history, suffix_adarms_cond.dtype)
        if res is not None:
            suffix_adarms_cond = suffix_adarms_cond + res

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
        final_hidden_states = []
        for i, hs in enumerate(hidden_states):
            if hs is None:
                final_hidden_states.append(None); continue
            hs, _ = norms[i](hs, cond=conds[i])
            final_hidden_states.append(hs)
        hidden_states = final_hidden_states

        suffix_out = hidden_states[1][:, -self.config.chunk_size:]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        v_t = self.action_out_proj(suffix_out)

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
        motion_past: Optional[Tensor] = None,  # v1 compat, ignored
        sam2_tokens: Optional[Tensor] = None,
        action_history: Optional[Tensor] = None,
        action_history_is_pad: Optional[Tensor] = None,
    ):
        """Shared-observation forward. action_history + sam2 are observation-
        time inputs (shared across offsets), broadcast across the num_offsets
        suffix branches for the adaRMS residual.
        """
        batch_size = states.shape[0]
        num_offsets = states.shape[1]

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(batch_size * num_offsets, actions.device).view(
                batch_size, num_offsets
            )

        time_expanded = time[:, :, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.prefix_embedder(
            images, img_masks, tokens, masks
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks = self._append_v2_prefix(
            prefix_embs, prefix_pad_masks, prefix_att_masks,
            sam2_tokens, action_history, action_history_is_pad,
        )

        states_flat = states.view(batch_size * num_offsets, -1)
        x_t_flat = x_t.view(batch_size * num_offsets, x_t.shape[2], -1)
        time_flat = time.view(batch_size * num_offsets)

        suffix_embs_flat, suffix_pad_masks_flat, suffix_att_masks_flat, suffix_adarms_cond_flat = self.suffix_embedder(
            states_flat, x_t_flat, time_flat
        )
        suffix_length = suffix_embs_flat.shape[1]

        # Broadcast v2 residual across offsets.
        res = self._v2_cond_residual(action_history, suffix_adarms_cond_flat.dtype)
        if res is not None:
            res = res.unsqueeze(1).expand(-1, num_offsets, -1).reshape(
                batch_size * num_offsets, -1
            )
            suffix_adarms_cond_flat = suffix_adarms_cond_flat + res

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
                suffix_adarms_conds, num_offsets, suffix_length,
            )

        norms = [self.vlm.language_model.norm, self.action_expert.model.norm]
        prefix_out, _ = norms[0](hidden_states[0], cond=None)

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
        v_t = self.action_out_proj(action_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    # Inference ODE paths: thread v2 conds through denoise_step / sample_actions.

    @torch.no_grad()
    def denoise_step(self, prefix_pad_masks, prefix_att_masks, state, x_t, timestep,
                     motion_past: Optional[Tensor] = None,
                     sam2_tokens: Optional[Tensor] = None,  # unused (prefix-only)
                     action_history: Optional[Tensor] = None,
                     action_history_is_pad: Optional[Tensor] = None):
        suffix_embs, suffix_pad_masks, suffix_att_masks, suffix_adarms_cond = self.suffix_embedder(
            state, x_t, timestep
        )
        res = self._v2_cond_residual(action_history, suffix_adarms_cond.dtype)
        if res is not None:
            suffix_adarms_cond = suffix_adarms_cond + res

        backbone_dtype = self.vlm.model.language_model.layers[0].input_layernorm.weight.dtype
        suffix_embs = suffix_embs.to(dtype=backbone_dtype)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        full_attention_mask, full_position_ids = build_attention_mask_and_position_ids(
            pad_masks, att_masks, suffix_embs.dtype
        )
        L_suf = suffix_embs.shape[1]
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
                       num_steps: Optional[int] = None,
                       motion_past: Optional[Tensor] = None,
                       sam2_tokens: Optional[Tensor] = None,
                       action_history: Optional[Tensor] = None,
                       action_history_is_pad: Optional[Tensor] = None) -> Tensor:
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
        prefix_embs, prefix_pad_masks, prefix_att_masks = self._append_v2_prefix(
            prefix_embs, prefix_pad_masks, prefix_att_masks,
            sam2_tokens, action_history, action_history_is_pad,
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
                conds_prefill, use_cache=True,
            )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        for _ in range(num_steps):
            v_t = self.denoise_step(
                prefix_pad_masks, prefix_att_masks, state, x_t,
                time.expand(bsz),
                action_history=action_history,
                action_history_is_pad=action_history_is_pad,
            )
            x_t = x_t + dt * v_t
            time = time + dt
        return x_t


class PI05MotionV2Policy(PI05MotionPolicy):
    """pi05_motion_v2 policy."""

    config_class = PI05MotionV2Config
    name = "pi05_motion_v2"

    def __init__(self, config: PI05MotionV2Config, dataset_stats=None):
        # Skip v1's PI05MotionPolicy.__init__ swap-to-MotionModel: we want the
        # MotionV2Model instead. Call grandparent PI05Policy.__init__ explicitly
        # via super-of-super, then attach our model.
        from vlash.policies.pi05.modeling_pi05 import PI05Policy
        PI05Policy.__init__(self, config, dataset_stats=dataset_stats)
        self.model = PI05MotionV2Model(config)
        self._n_action_steps = config.n_action_steps
        self._motion_steps = config.motion_steps
        self._motion_loss_weight = config.motion_loss_weight

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

        motion_future = batch.get(OBS_MOTION_FUTURE, None)
        motion_is_pad = batch.get(OBS_MOTION_FUTURE_IS_PAD, None)
        sam2_tokens, action_history, action_history_is_pad = self._extract_v2(batch)

        cfg = self.config
        bsz = actions.shape[0]
        if motion_future is None:
            raise RuntimeError("pi05_motion_v2 requires observation.motion_future")
        motion_tokens = motion_future.float().reshape(
            bsz, cfg.motion_steps, cfg.max_action_dim
        ).to(actions.dtype).to(actions.device)
        ext = torch.cat([actions, motion_tokens], dim=1)

        losses = self.model(
            images, img_masks, lang_tokens, lang_masks,
            state, ext, noise=noise, time=time,
            sam2_tokens=sam2_tokens,
            action_history=action_history,
            action_history_is_pad=action_history_is_pad,
        )

        action_dim = cfg.output_features[ACTION].shape[0]
        action_losses = losses[:, : cfg.n_action_steps, : action_dim]
        if actions_is_pad is not None:
            action_losses = action_losses * (~actions_is_pad).unsqueeze(-1)
        action_loss = action_losses.mean()

        motion_losses = losses[:, cfg.n_action_steps:, :]
        if motion_is_pad is not None:
            valid = (~motion_is_pad.bool()).to(motion_losses.dtype).view(bsz, 1, 1)
            denom = valid.sum().clamp_min(1.0) * motion_losses.shape[1] * motion_losses.shape[2]
            motion_loss = (motion_losses * valid).sum() / denom
        else:
            motion_loss = motion_losses.mean()

        total_loss = action_loss + self._motion_loss_weight * motion_loss
        return total_loss, {
            "loss": total_loss.item(),
            "action_loss": action_loss.item(),
            "motion_loss": motion_loss.item(),
        }

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

        motion_future = batch.get(OBS_MOTION_FUTURE, None)
        motion_future_is_pad = batch.get(OBS_MOTION_FUTURE_IS_PAD, None)
        sam2_tokens, action_history, action_history_is_pad = self._extract_v2(batch)
        if motion_future is None:
            raise RuntimeError(
                "pi05_motion_v2 shared-observation requires observation.motion_future"
            )
        motion_tokens = motion_future.float().reshape(
            batch_size, num_offsets, cfg.motion_steps, cfg.max_action_dim
        ).to(actions_normalized.dtype).to(actions_normalized.device)
        ext = torch.cat([actions_normalized, motion_tokens], dim=2)

        images, img_masks = self.prepare_images(batch)
        if not cfg.state_cond:
            raise ValueError("state_cond must be True for shared observation training")
        lang_tokens, lang_masks = self.prepare_language(batch)

        losses = self.model.forward_shared_observation(
            images, img_masks, lang_tokens, lang_masks,
            states_normalized, ext, offset_mask,
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
        action_denom = (num_valid * n_action * action_dim).to(action_losses.dtype)
        action_loss = action_losses.sum() / action_denom

        motion_losses = losses[:, :, n_action:, :]
        if motion_future_is_pad is not None:
            motion_valid = (~motion_future_is_pad.bool()).to(motion_losses.dtype)
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
        return total_loss, {
            "loss": total_loss.item(),
            "action_loss": action_loss.item(),
            "motion_loss": motion_loss.item(),
            "num_offsets": num_offsets,
            "avg_valid_offsets": offset_mask.float().sum(dim=1).mean().item(),
        }

    @torch.no_grad()
    def predict_chunk(self, batch: dict, noise: Optional[Tensor] = None):
        self.eval()
        batch = self.normalize_inputs(batch)
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch, pad_to_max_length=False)
        sam2_tokens, action_history, action_history_is_pad = self._extract_v2(batch)
        ext = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise,
            sam2_tokens=sam2_tokens,
            action_history=action_history,
            action_history_is_pad=action_history_is_pad,
        )
        cfg = self.config
        actions = ext[:, : cfg.n_action_steps]
        original_action_dim = cfg.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]
        actions = self.unnormalize_outputs({ACTION: actions})[ACTION]
        motion_pred = ext[:, cfg.n_action_steps:].reshape(ext.shape[0], cfg.motion_dim)
        return actions, motion_pred
