"""SAM2.1-tiny post-memory-attention latent extractor for 16-frame head-cam
windows. Streams frames through SAM2's memory attention with a *full-frame*
synthetic mask at each step -- no per-frame object position is assumed -- so
memory_attention propagates whole-image features through time. This is
TEMPO-MOT's SAM2 input.

Input: float tensor [T, 3, 1024, 1024] in [0, 1], bfloat16, on CUDA.
Output: tensor [T, 256, 8, 8] float32 (per-frame post-memory-attn features,
adaptive-avg-pooled from the native 64x64 grid).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# SAM2 source tree and checkpoint. Set SAM2_REPO / SAM2_CKPT to point at your own clone of
# facebookresearch/sam2 and a sam2.1_hiera_tiny.pt.
_SAM2_REPO_CANDIDATES = [
    *([os.environ["SAM2_REPO"]] if os.environ.get("SAM2_REPO") else []),
    "./sam2",
]
SAM2_REPO = Path(next((p for p in _SAM2_REPO_CANDIDATES if Path(p).is_dir()),
                       _SAM2_REPO_CANDIDATES[0]))
# SAM2.1-tiny checkpoint.
_CKPT_CANDIDATES = [
    *([os.environ["SAM2_CKPT"]] if os.environ.get("SAM2_CKPT") else []),
    "./checkpoints/sam2.1_hiera_tiny.pt",
]
SAM2_CKPT = Path(next((p for p in _CKPT_CANDIDATES if Path(p).exists()),
                       _CKPT_CANDIDATES[0]))
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"

# Pooling grid for cached spatial features.
LATENT_GRID = 8
LATENT_DIM = 256  # SAM2 d_model
SAM2_INPUT_SIZE = 1024


def _ensure_sam2_on_path():
    # Insert at front so we shadow any pip-installed sam2.
    if str(SAM2_REPO) not in sys.path[:3]:
        sys.path.insert(0, str(SAM2_REPO))


def build_sam2_tiny(device: str = "cuda"):
    _ensure_sam2_on_path()
    from sam2.build_sam import build_sam2_video_predictor
    model = build_sam2_video_predictor(SAM2_CFG, str(SAM2_CKPT), device=device).eval()
    return model


def _full_frame_mask(size: int, device, dtype=torch.bfloat16):
    """Logits-like full-image mask: + everywhere. Shape (1, 1, size, size)."""
    m = torch.full((1, 1, size, size), 10.0, device=device, dtype=dtype)
    return m


@torch.no_grad()
def stream_extract(model, frames_bchw01: torch.Tensor, grid: int = LATENT_GRID,
                    use_full_frame_mask: bool = True) -> torch.Tensor:
    """frames_bchw01: [T, 3, H, W] bfloat16 [0,1] on CUDA, H=W=SAM2_INPUT_SIZE.
    Returns [T, 256, grid, grid] float32 (CPU).
    """
    assert frames_bchw01.dim() == 4 and frames_bchw01.shape[1] == 3
    T = frames_bchw01.shape[0]
    device = frames_bchw01.device
    output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    out = []
    size = model.image_size

    for t in range(T):
        x = frames_bchw01[t:t + 1]
        bo = model.forward_image(x)
        _, vf, vp, fs = model._prepare_backbone_features(bo)
        pix = model._prepare_memory_conditioned_features(
            frame_idx=t,
            is_init_cond_frame=(t == 0),
            current_vision_feats=vf[-1:],
            current_vision_pos_embeds=vp[-1:],
            feat_sizes=fs[-1:],
            output_dict=output_dict,
            num_frames=T,
            track_in_reverse=False,
        )  # (1, 256, 64, 64) at 1024 input
        pooled = F.adaptive_avg_pool2d(pix, (grid, grid))  # (1, 256, grid, grid)
        out.append(pooled.squeeze(0).float().cpu())

        if use_full_frame_mask:
            m = _full_frame_mask(size, device=device, dtype=x.dtype)
            mm_feats, mm_pos = model._encode_new_memory(
                current_vision_feats=vf,
                feat_sizes=fs,
                pred_masks_high_res=m,
                object_score_logits=torch.tensor([[1.0]], device=device, dtype=x.dtype),
                is_mask_from_pts=False,
            )
            slot = "cond_frame_outputs" if t == 0 else "non_cond_frame_outputs"
            output_dict[slot][t] = {
                "maskmem_features": mm_feats,
                "maskmem_pos_enc": mm_pos,
                "obj_ptr": torch.zeros(1, model.hidden_dim, device=device, dtype=x.dtype),
                "object_score_logits": torch.tensor([[1.0]], device=device, dtype=x.dtype),
            }
    return torch.stack(out, dim=0)  # (T, 256, grid, grid)


def preprocess_video_to_sam2_input(video_thwc_uint8: np.ndarray) -> torch.Tensor:
    """video_thwc_uint8: numpy [T, H, W, 3] uint8 (RGB). Returns CUDA bf16 tensor
    [T, 3, 1024, 1024] in [0, 1].
    """
    import cv2
    T = video_thwc_uint8.shape[0]
    resized = np.empty((T, SAM2_INPUT_SIZE, SAM2_INPUT_SIZE, 3), dtype=np.uint8)
    for i in range(T):
        resized[i] = cv2.resize(video_thwc_uint8[i], (SAM2_INPUT_SIZE, SAM2_INPUT_SIZE),
                                interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(resized).permute(0, 3, 1, 2).contiguous()
    return t.to(device="cuda", dtype=torch.bfloat16) / 255.0
