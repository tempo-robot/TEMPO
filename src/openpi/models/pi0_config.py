import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # --- TEMPO-MOT: visual motion memory (short-term video memory) ---
    # Number of observation frames K fed to the video encoder, including the current frame.
    # 1 == no memory == standard single-frame pi0.5 (exactly reproduces the base model).
    obs_history: int = 1
    # Temporal spacing between consecutive memory frames, in seconds.
    obs_history_stride_s: float = 1.0
    # Apply space-time separable (causal temporal) attention every N-th ViT layer.
    temporal_attn_period: int = 4

    # --- TEMPO-MOT: SAM2 motion cue ---
    # Fuse a frozen SAM2 memory-attention feature (head cam) into the current head-frame
    # visual tokens via a zero-gated cross-attention block. Fuses into the current tokens, so
    # the backbone token count is unchanged. False == off (TEMPO-MOT's video path only).
    use_sam2_fusion: bool = False
    # SAM2 memory-attention channel dim (feature "straight out of memory attention", no pool).
    sam2_token_dim: int = 256
    # Number of SAM2 spatial tokens per frame (8x8 = 64).
    sam2_num_tokens: int = 64
    # Number of heads in the fusion cross-attention.
    sam2_fusion_heads: int = 8

    # --- TEMPO-ACT: compact past-action history ---
    # Injected two ways: (a) `action_history_steps` extra prefix
    # tokens after the language tokens, (b) an additive zero-init residual on the action
    # expert's adaRMS conditioning vector. False == off (TEMPO-MOT only).
    use_action_history: bool = False
    # Number of aggregated past-action vectors (bucket means over the history window).
    action_history_steps: int = 10
    # Raw dataset action dim of one history vector (14 = YAM dual-arm, 7 = single arm).
    action_history_dim: int = 14

    # --- Prediction head: u_t latent (default) vs. direct action-chunk regression ---
    # True  == the flow-matching target is the `ut_dim` MVAE latent mu_v of the observation
    #          window starting at t; a frozen decoder (`ut_decoder_path`) turns the denoised
    #          latent into the action chunk at inference.
    # False == the flow-matching target is the padded action chunk itself (standard pi0.5).
    # This is the only switch: everything else about the model is identical, and the data
    # pipeline loads both targets, so flipping it needs no other change.
    predict_ut: bool = True
    # Dimensionality of the u_t latent (= the MVAE's latent_dim).
    ut_dim: int = 64
    # Number of action steps the u_t decoder emits (= the MVAE window W). Must equal
    # action_horizon so the decoded chunk lines up with the chunk the dataset supplies.
    ut_window: int = 16
    # Checkpoint of the trained u_t -> action-chunk decoder (tools/ut/train_ut_decoder.py).
    # Required to SAMPLE actions when predict_ut is True; not needed for training.
    ut_decoder_path: str | None = None

    @property
    def flow_dim(self) -> int:
        """Width of the flow-matching target: the u_t latent, or the padded action dim."""
        return self.ut_dim if self.predict_ut else self.action_dim

    @property
    def flow_positions(self) -> int:
        """Number of suffix tokens carrying the flow target: one latent, or one per action step."""
        return 1 if self.predict_ut else self.action_horizon

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.predict_ut and self.ut_window != self.action_horizon:
            raise ValueError(
                f"ut_window ({self.ut_window}) must equal action_horizon ({self.action_horizon}): "
                "the decoder emits one chunk of ut_window steps and the dataset supplies "
                "action_horizon steps to compare it against."
            )

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
