"""Frozen u_t -> action-chunk decoder used at inference when `predict_ut` is on.

The policy's flow-matching head denoises a `ut_dim` latent (the MVAE's mu_v for the
observation window starting at t). This module turns that latent into the action chunk.

Architecture mirrors the MVAE's ActionDecoder (tools/ut/mvae.py) so a checkpoint trained by
tools/ut/train_ut_decoder.py loads directly. The decoder is trained to emit actions in the
SAME normalized space the policy works in (openpi quantile / z-score norm stats), so its
output can be returned from `sample_actions` unchanged and the standard Unnormalize output
transform recovers raw actions. `train_ut_decoder.py` records which space it used and
`from_pretrained` refuses a checkpoint trained in any other one.
"""

import math
import pathlib

import torch
from torch import nn


def _sinusoidal_pe(n: int, d: int, device=None) -> torch.Tensor:
    pe = torch.zeros(n, d, device=device)
    pos = torch.arange(n, dtype=torch.float32, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class _TransformerEncoder(nn.Module):
    """Matches tools/ut/mvae.py's wrapper so decoder state-dict keys line up."""

    def __init__(self, enc_layer, n_layers):
        super().__init__()
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, x):
        return self.enc(x)


class UtActionDecoder(nn.Module):
    """u (B, latent_dim) -> action chunk (B, window, action_dim), in the policy's normalized space."""

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 512,
        window: int = 16,
        action_dim: int = 14,
        n_layers: int = 2,
        nhead: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.window = window
        self.action_dim = action_dim
        self.from_u = nn.Linear(latent_dim, hidden_dim)
        self.temporal_pe = nn.Parameter(torch.zeros(1, window, hidden_dim))
        nn.init.trunc_normal_(self.temporal_pe, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        # Wrapped one level deep (`.enc`) to mirror the MVAE's _TransformerEncoder, so a
        # checkpoint from tools/ut/train_ut_decoder.py loads with matching key names.
        self.transformer = _TransformerEncoder(enc_layer, n_layers)
        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        cond = self.from_u(u)
        q = self.temporal_pe.expand(u.shape[0], -1, -1) + cond[:, None, :]
        q = self.transformer(q)
        return self.head(q)

    @classmethod
    def from_pretrained(cls, path: str | pathlib.Path) -> "UtActionDecoder":
        """Load a checkpoint written by tools/ut/train_ut_decoder.py."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if "act_dec" not in ckpt:
            raise ValueError(f"{path} is not a u_t decoder checkpoint (no 'act_dec' state dict)")
        space = ckpt.get("action_space")
        if space != "policy_norm":
            raise ValueError(
                f"u_t decoder at {path} was trained in action space {space!r}, not 'policy_norm'. "
                "Retrain it with tools/ut/train_ut_decoder.py --norm-stats <assets>/norm_stats.json "
                "so its output matches the space the policy's Unnormalize transform expects."
            )
        cfg = ckpt["model_cfg"]
        dec = cls(
            latent_dim=cfg["latent_dim"],
            hidden_dim=cfg["hidden_dim"],
            window=cfg["window"],
            action_dim=cfg["action_dim"],
            n_layers=cfg.get("dec_layers", 2),
            nhead=cfg.get("nhead", 8),
            dropout=0.0,
        )
        dec.load_state_dict(ckpt["act_dec"])
        dec.eval()
        for p in dec.parameters():
            p.requires_grad_(False)
        return dec
