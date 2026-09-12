import logging
import math
import pathlib

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
from openpi.models_pytorch.ut_decoder import UtActionDecoder
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class SAM2CrossAttnFusion(nn.Module):
    """Zero-gated cross-attention fusing frozen SAM2 memory-attention tokens into
    the current head-frame SigLIP visual tokens.

    The SAM2 motion cue is fused *into* the current visual tokens (query
    = the head cam's visual tokens, key/value = the 64 SAM2 spatial tokens), so the
    backbone token count / latency is unchanged. The block is a no-op at init
    (zero-init gate AND zero-init output projection) so a warm start from a trained
    video-path-only checkpoint is bit-identical to it, then it learns to pull in SAM2. Compute
    runs in float32 for dtype safety; the residual is cast back to the query dtype.
    """

    def __init__(self, vis_dim: int, sam2_dim: int, num_tokens: int, num_heads: int = 8):
        super().__init__()
        if vis_dim % num_heads != 0:
            raise ValueError(f"vis_dim {vis_dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = vis_dim // num_heads
        self.norm_q = nn.LayerNorm(vis_dim)
        self.kv_in = nn.Linear(sam2_dim, vis_dim)
        self.sam2_pos = nn.Parameter(torch.zeros(num_tokens, vis_dim))
        self.norm_kv = nn.LayerNorm(vis_dim)
        self.q_proj = nn.Linear(vis_dim, vis_dim)
        self.k_proj = nn.Linear(vis_dim, vis_dim)
        self.v_proj = nn.Linear(vis_dim, vis_dim)
        self.out_proj = nn.Linear(vis_dim, vis_dim)
        self.gate = nn.Parameter(torch.zeros(1))
        # Zero-init output => block contributes nothing at init.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, vis_tokens: torch.Tensor, sam2_tokens: torch.Tensor) -> torch.Tensor:
        # vis_tokens: (B, Nv, vis_dim); sam2_tokens: (B, Ns, sam2_dim)
        dtype_in = vis_tokens.dtype
        vis = vis_tokens.float()
        sam2 = sam2_tokens.float()
        b, nv, d = vis.shape
        ns = sam2.shape[1]
        q = self.q_proj(self.norm_q(vis))
        kv = self.norm_kv(self.kv_in(sam2) + self.sam2_pos.unsqueeze(0))
        k = self.k_proj(kv)
        v = self.v_proj(kv)
        q = q.view(b, nv, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, ns, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, ns, self.num_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(b, nv, d)
        out = self.gate.tanh() * self.out_proj(attn)
        return vis_tokens + out.to(dtype_in)


class ActionHistoryCondMLP(nn.Module):
    """in_dim -> hidden -> silu -> hidden -> silu with a zero-init output layer, so the
    module starts as a zero contribution to the adaRMS cond (identity residual).

    """

    def __init__(self, in_dim: int, hidden_size: int):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_size)
        self.mlp_in = nn.Linear(hidden_size, hidden_size)
        self.mlp_out = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.mlp_out.weight)
        nn.init.zeros_(self.mlp_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = F.silu(self.mlp_in(h))
        return F.silu(self.mlp_out(h))


class ActionHistoryTokens(nn.Module):
    """Project `num_tokens` past-action vectors into the VLM hidden size: one shared
    per-token linear plus a learnable per-position embedding (zero-init).

    """

    def __init__(self, in_dim: int, num_tokens: int, vlm_hidden: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, vlm_hidden)
        self.pos_emb = nn.Parameter(torch.zeros(num_tokens, vlm_hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, N, in_dim) -> (B, N, vlm_hidden)
        return self.proj(x) + self.pos_emb.unsqueeze(0)


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
            tempo_mot_temporal_period=getattr(config, "temporal_attn_period", 4),
        )
        # TEMPO-MOT: number of observation frames (incl. current) fed to the video encoder. 1 = no memory.
        self.obs_history = getattr(config, "obs_history", 1)

        # Prediction head. predict_ut: the flow target is the `ut_dim` u_t latent carried by a
        # single suffix token. Otherwise it is the padded action chunk, one token per step.
        # `flow_dim` / `flow_positions` are the only things that differ between the two.
        self.predict_ut = bool(getattr(config, "predict_ut", False))
        self.flow_dim = int(getattr(config, "flow_dim", 32))
        self.flow_positions = int(getattr(config, "flow_positions", config.action_horizon))
        self.action_in_proj = nn.Linear(self.flow_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, self.flow_dim)

        # Frozen u_t -> action-chunk decoder, needed only to SAMPLE actions -- training reads its
        # targets from ut_cache_dir. Kept out of the module tree (a plain attribute) so it never
        # lands in the policy's state dict. A configured-but-missing path is not fatal here: the
        # decoder is built after the policy in the normal workflow, and _decode_ut raises a clear
        # error if sampling is attempted without one.
        self._ut_decoder = None
        ut_decoder_path = getattr(config, "ut_decoder_path", None)
        if self.predict_ut and ut_decoder_path is not None:
            if pathlib.Path(ut_decoder_path).exists():
                object.__setattr__(self, "_ut_decoder", UtActionDecoder.from_pretrained(ut_decoder_path))
                logging.info(f"Loaded u_t action decoder from {ut_decoder_path}")
            else:
                logging.warning(
                    f"ut_decoder_path {ut_decoder_path} does not exist; training will work but "
                    "sampling actions will fail until it is built (tools/ut/train_ut_decoder.py)."
                )

        # TEMPO-MOT SAM2 motion cue: fuse frozen SAM2 memory-attention tokens into the head-cam
        # visual tokens. Zero-gated => no-op at init (warm start is bit-identical).
        self.use_sam2_fusion = getattr(config, "use_sam2_fusion", False)
        if self.use_sam2_fusion:
            self.sam2_fusion = SAM2CrossAttnFusion(
                vis_dim=paligemma_config.width,
                sam2_dim=getattr(config, "sam2_token_dim", 256),
                num_tokens=getattr(config, "sam2_num_tokens", 64),
                num_heads=getattr(config, "sam2_fusion_heads", 8),
            )
        else:
            self.sam2_fusion = None

        # TEMPO-ACT: compact past-action history -- `action_history_steps` extra prefix tokens
        # AND a zero-init additive residual on the action expert's adaRMS cond.
        self.use_action_history = getattr(config, "use_action_history", False)
        if self.use_action_history:
            self.action_history_steps = int(getattr(config, "action_history_steps", 10))
            hist_dim = int(getattr(config, "action_history_dim", 14))
            self.action_history_cond_path = ActionHistoryCondMLP(
                self.action_history_steps * hist_dim, action_expert_config.width
            )
            self.action_history_vlm_tokens = ActionHistoryTokens(
                hist_dim, self.action_history_steps, paligemma_config.width
            )
        else:
            self.action_history_steps = 0
            self.action_history_cond_path = None
            self.action_history_vlm_tokens = None

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
            getattr(observation, "sam2_tokens", None),
            getattr(observation, "action_history", None),
            getattr(observation, "action_history_is_pad", None),
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, sam2_tokens=None, action_history=None,
        action_history_is_pad=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for cam_idx, (img, img_mask) in enumerate(zip(images, img_masks, strict=True)):
            # TEMPO-MOT video encoder: images may carry a temporal frame axis (B, K, C, H, W). Fold the
            # K frames into the batch dim; embed_image returns only the current frame's tokens.
            if img.ndim == 5:
                b, k = img.shape[:2]
                img_in = img.reshape(b * k, *img.shape[2:])
            else:
                k = 1
                img_in = img

            def image_embed_func(x, num_frames=k):
                return self.paligemma_with_expert.embed_image(x, num_frames=num_frames)

            img_emb = self._apply_checkpoint(image_embed_func, img_in)

            # SAM2 motion fusion into the head cam's current tokens (cam index 0 = base_0_rgb).
            if self.use_sam2_fusion and sam2_tokens is not None and cam_idx == 0:
                img_emb = self.sam2_fusion(img_emb, sam2_tokens)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        # TEMPO-ACT prefix tokens: the past-action buckets go last in the prefix.
        # att_masks=0 -> no new attention boundary; buckets that are entirely pre-episode are
        # masked out via pad_masks.
        if self.use_action_history and action_history is not None:
            hist_emb = self.action_history_vlm_tokens(
                action_history.to(dtype=self.action_history_vlm_tokens.proj.weight.dtype)
            )
            n_hist = hist_emb.shape[1]
            if action_history_is_pad is None:
                hist_mask = torch.ones(
                    hist_emb.shape[0], n_hist, dtype=torch.bool, device=hist_emb.device
                )
            else:
                hist_mask = ~action_history_is_pad.bool().to(device=hist_emb.device)
            embs.append(hist_emb.to(dtype=embs[0].dtype))
            pad_masks.append(hist_mask)
            att_masks += [0] * n_hist

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep, action_history=None):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

            # TEMPO-ACT adaRMS residual: zero-init MLP over the flattened history, so this is
            # exactly 0 at init and the cond is unchanged (TEMPO-ACT's second injection route).
            if self.use_action_history and action_history is not None:
                flat = action_history.reshape(action_history.shape[0], -1).to(
                    dtype=self.action_history_cond_path.in_proj.weight.dtype
                )
                adarms_cond = adarms_cond + self.action_history_cond_path(flat).to(
                    dtype=adarms_cond.dtype
                )

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.flow_positions - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def _predict_velocity(
        self, prefix_embs, prefix_pad_masks, prefix_att_masks, state, x_t, time, action_history=None
    ):
        """Joint prefix+suffix transformer pass -> predicted flow velocity v_t (B, H, A)."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state, x_t, time, action_history=action_history
        )
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.flow_positions :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
        return v_t

    def _flow_target(self, observation, actions):
        """(target, target_is_pad) for the flow-matching loss, per the active head."""
        if not self.predict_ut:
            return actions, None
        ut = getattr(observation, "ut_target", None)
        if ut is None:
            raise ValueError(
                "predict_ut is True but the batch carries no `ut_target`. Set `ut_cache_dir` on the "
                "data config (see tools/ut/) or set predict_ut=False to regress action chunks."
            )
        ut = ut.to(dtype=torch.float32, device=actions.device)
        if ut.ndim == 2:  # (B, ut_dim) -> one suffix token
            ut = ut[:, None, :]
        is_pad = getattr(observation, "ut_target_is_pad", None)
        return ut, is_pad

    def _decode_ut(self, u_t, device):
        """Denoised u_t latent -> action chunk, padded to the model's action dim."""
        if self._ut_decoder is None:
            raise ValueError(
                "predict_ut is True but no u_t decoder is loaded. Set `ut_decoder_path` on the model "
                "config to a checkpoint from tools/ut/train_ut_decoder.py."
            )
        dec = self._ut_decoder.to(device=device, dtype=torch.float32)
        decoded = dec(u_t.squeeze(1).to(torch.float32))  # (B, window, raw action dim)
        bsize, horizon, raw_dim = decoded.shape
        padded = torch.zeros(bsize, horizon, self.config.action_dim, device=device, dtype=decoded.dtype)
        padded[:, :, :raw_dim] = decoded
        return padded

    def forward(
        self,
        observation,
        actions,
        noise=None,
        time=None,
    ) -> Tensor:
        """Do a full training forward pass and compute the flow-matching loss
        (batch_size x num_steps x num_motors)."""
        (
            images, img_masks, lang_tokens, lang_masks, state, sam2_tokens, action_history,
            action_history_is_pad,
        ) = self._preprocess_observation(observation, train=True)

        target, target_is_pad = self._flow_target(observation, actions)

        if noise is None:
            noise = self.sample_noise(target.shape, target.device)

        if time is None:
            time = self.sample_time(target.shape[0], target.device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, sam2_tokens=sam2_tokens,
            action_history=action_history, action_history_is_pad=action_history_is_pad,
        )

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * target
        u_t = noise - target

        v_t = self._predict_velocity(
            prefix_embs, prefix_pad_masks, prefix_att_masks, state, x_t, time,
            action_history=action_history,
        )

        flow_loss = F.mse_loss(u_t, v_t, reduction="none")
        if target_is_pad is not None:
            # Frames too close to the end of the episode have no full observation window and
            # therefore no u_t target; zero their contribution instead of regressing to zeros.
            flow_loss = flow_loss * (~target_is_pad.bool()).to(flow_loss.dtype)[:, None, None]

        return flow_loss

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            noise = self.sample_noise((bsize, self.flow_positions, self.flow_dim), device)

        (
            images, img_masks, lang_tokens, lang_masks, state, sam2_tokens, action_history,
            action_history_is_pad,
        ) = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, sam2_tokens=sam2_tokens,
            action_history=action_history, action_history_is_pad=action_history_is_pad,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
                action_history=action_history,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt

        if self.predict_ut:
            return self._decode_ut(x_t, device)
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        action_history=None,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state, x_t, timestep, action_history=action_history
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.flow_positions :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
