#!/usr/bin/env python

# Copyright 2025 VLASH team. All rights reserved.
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""VLASH Training Configuration.

This module defines configuration classes for VLASH training:
- LoRAConfig: Low-Rank Adaptation settings for efficient fine-tuning
- VLASHTrainConfig: Main training config extending LeRobot's TrainPipelineConfig
"""

from dataclasses import dataclass, field
from typing import List

from lerobot.configs.train import TrainPipelineConfig


@dataclass
class LoRAConfig:
    """Configuration for Low-Rank Adaptation (LoRA).
        """

    # Enable/disable LoRA
    enable: bool = False

    # Backend implementation (currently only "peft" supported)
    backend: str = "peft"

    # LoRA hyperparameters
    r: int = 16  # Rank of low-rank matrices
    alpha: int = 16  # Scaling factor (effective lr = alpha/r * lr)
    dropout: float = 0.0  # Dropout on LoRA layers

    # Additional modules to keep trainable (pattern-based matching on param names)
    extra_trainable_modules: List[str] = field(default_factory=list)

    # Target modules for LoRA injection
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ]
    )

    # QLoRA: 4-bit quantization of frozen weights
    use_qlora: bool = False
    qlora_quant_type: str = "nf4"  # "nf4" (normalized float4) or "fp4"
    qlora_compute_dtype: str = "bfloat16"  # Dtype for forward pass

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.backend not in {"peft"}:
            raise ValueError(f"Invalid LoRA backend: {self.backend!r}.")
        
        if self.use_qlora:
            if self.qlora_quant_type not in {"nf4", "fp4"}:
                raise ValueError(
                    f"Invalid qlora_quant_type: {self.qlora_quant_type!r}. "
                    "Supported values are 'nf4' and 'fp4'."
                )
            if self.qlora_compute_dtype not in {"float16", "bfloat16", "float32"}:
                raise ValueError(
                    f"Invalid qlora_compute_dtype: {self.qlora_compute_dtype!r}. "
                    "Supported values are 'float16', 'bfloat16', 'float32'."
                )


@dataclass
class VLASHTrainConfig(TrainPipelineConfig):
    """VLASH training configuration.

    Extends TrainPipelineConfig with VLASH-specific parameters.
    """

    # Temporal delay augmentation: random offset in [0, max_delay_steps]
    # Set to 0 to disable (standard training without delay)
    max_delay_steps: int = 0

    # Gradient accumulation steps
    grad_accum_steps: int = 1

    # Shared observation optimization: train all offsets together with shared
    # observation (images + language). This provides ~(max_delay_steps+1)x speedup
    # by computing observation embeddings only once and using custom attention
    # masks to prevent cross-offset attention.
    shared_observation: bool = False

    # If set, restrict the dataset to only the listed camera keys (e.g.
    # ["observation.images.head"]). Cameras in the dataset but not listed here
    # are dropped before the policy sees the metadata, so they are not loaded
    # or fed to the model. None keeps all dataset cameras.
    selected_cameras: list[str] | None = None

    # Per-camera multiplicative brightness adjustment applied at dataset load
    # time, after video decoding (frames are float in [0, 1]). Useful when one
    # camera is much darker than others (e.g. wrist cams with poor exposure).
    # Output is clipped to [0, 1]. Cameras not listed are left unchanged.
    camera_brightness: dict = field(default_factory=dict)

    # LoRA configuration
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # UVT augmentation: per-episode precomputed u_window sidecar dir. If set,
    # the dataset is wrapped to inject observation.uvt_past and
    # observation.uvt_target keys per sample. Used by the pi05_uvt policy.
    uvt_dir: str | None = None
    uvt_window: int = 16
    uvt_dim: int = 64

    # Motion augmentation: per-episode precomputed motion_past/motion_future
    # sidecar dir (see /home/dfeng8/sam2_uvt/precompute_motion_diff.py). When
    # set, the dataset is wrapped to inject observation.motion_past +
    # observation.motion_future. Used by the pi05_motion policy.
    motion_dir: str | None = None
    motion_dim: int = 256

    # SAM2 raw mem-attn latent cache dir (per-episode .npz with `latent`
    # key shape (T, C, H, W)). When set together with motion_dir and a
    # pi05_motion_v2 policy, the dataset wraps with the V2 wrapper which
    # also emits observation.sam2_tokens + observation.action_history.
    sam2_dir: str | None = None
