"""Product-of-Experts multimodal VAE fusing (SAM2 latent window, action chunk)
into a single Gaussian latent u.

Inputs per sample:
  latent: (W, 256, 8, 8) — post-memory-attn SAM2 feature, W=16
  action: (W, 14)        — yam_dual_arm action vector per frame

Encoders produce per-modality (mu, logvar) in R^d. Precision-weighted fusion:
  inv_var_joint = inv_var_v + inv_var_a + 1 (prior precision 1)
  mu_joint      = (inv_var_v * mu_v + inv_var_a * mu_a) / inv_var_joint
  var_joint     = 1 / inv_var_joint

Decoders map u back to each modality. Losses:
  L = mse(latent_recon, latent) + lambda_a * mse(action_recon, action)
      + beta * KL(q(u|both) || N(0, I))
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MVAEConfig:
    window: int = 16
    sam_dim: int = 256
    sam_grid: int = 8           # tokens per frame = grid*grid = 64
    action_dim: int = 14
    hidden_dim: int = 512
    latent_dim: int = 256
    enc_layers: int = 2
    dec_layers: int = 2
    nhead: int = 8
    dropout: float = 0.0
    beta: float = 1e-3
    action_loss_weight: float = 1.0


def _sinusoidal_pe(n: int, d: int, device=None) -> torch.Tensor:
    pe = torch.zeros(n, d, device=device)
    pos = torch.arange(n, dtype=torch.float32, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32, device=device)
                    * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class _TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, n_layers: int, dropout: float):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, x):  # x: (B, N, D)
        return self.enc(x)


class VideoLatentEncoder(nn.Module):
    """SAM2 latent window (W, 256, 8, 8) -> (mu, logvar) in R^latent_dim."""
    def __init__(self, cfg: MVAEConfig):
        super().__init__()
        self.cfg = cfg
        n_tokens_per_frame = cfg.sam_grid * cfg.sam_grid
        self.token_proj = nn.Linear(cfg.sam_dim, cfg.hidden_dim)
        # Learned spatial + temporal positional embeddings.
        self.spatial_pe = nn.Parameter(torch.zeros(1, 1, n_tokens_per_frame, cfg.hidden_dim))
        self.temporal_pe = nn.Parameter(torch.zeros(1, cfg.window, 1, cfg.hidden_dim))
        nn.init.trunc_normal_(self.spatial_pe, std=0.02)
        nn.init.trunc_normal_(self.temporal_pe, std=0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.hidden_dim))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.transformer = _TransformerEncoder(
            cfg.hidden_dim, cfg.nhead, cfg.enc_layers, cfg.dropout)
        self.to_mu_logvar = nn.Linear(cfg.hidden_dim, 2 * cfg.latent_dim)

    def forward(self, x):
        # x: (B, W, 256, 8, 8)
        B, W, C, H, Wd = x.shape
        x = x.permute(0, 1, 3, 4, 2).reshape(B, W, H * Wd, C)  # (B, W, N, C)
        x = self.token_proj(x)                                   # (B, W, N, H)
        x = x + self.spatial_pe + self.temporal_pe              # broadcast
        x = x.reshape(B, W * H * Wd, -1)                         # (B, W*N, H)
        cls = self.cls.expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.transformer(x)                                  # (B, 1+W*N, H)
        z = x[:, 0]                                              # (B, H)
        out = self.to_mu_logvar(z)                               # (B, 2L)
        mu, logvar = out.chunk(2, dim=-1)
        return mu, logvar


class ActionEncoder(nn.Module):
    """Action chunk (W, 14) -> (mu, logvar) in R^latent_dim."""
    def __init__(self, cfg: MVAEConfig):
        super().__init__()
        self.cfg = cfg
        self.token_proj = nn.Linear(cfg.action_dim, cfg.hidden_dim)
        self.temporal_pe = nn.Parameter(torch.zeros(1, cfg.window, cfg.hidden_dim))
        nn.init.trunc_normal_(self.temporal_pe, std=0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.hidden_dim))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.transformer = _TransformerEncoder(
            cfg.hidden_dim, cfg.nhead, cfg.enc_layers, cfg.dropout)
        self.to_mu_logvar = nn.Linear(cfg.hidden_dim, 2 * cfg.latent_dim)

    def forward(self, a):
        # a: (B, W, 14)
        B, W, _ = a.shape
        x = self.token_proj(a) + self.temporal_pe                # (B, W, H)
        cls = self.cls.expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.transformer(x)
        z = x[:, 0]
        out = self.to_mu_logvar(z)
        mu, logvar = out.chunk(2, dim=-1)
        return mu, logvar


class VideoLatentDecoder(nn.Module):
    """u in R^latent_dim -> SAM2 latent window (W, 256, 8, 8) reconstruction."""
    def __init__(self, cfg: MVAEConfig):
        super().__init__()
        self.cfg = cfg
        n_per_frame = cfg.sam_grid * cfg.sam_grid
        self.tokens_per_window = cfg.window * n_per_frame
        self.from_u = nn.Linear(cfg.latent_dim, cfg.hidden_dim)
        self.spatial_pe = nn.Parameter(torch.zeros(1, 1, n_per_frame, cfg.hidden_dim))
        self.temporal_pe = nn.Parameter(torch.zeros(1, cfg.window, 1, cfg.hidden_dim))
        nn.init.trunc_normal_(self.spatial_pe, std=0.02)
        nn.init.trunc_normal_(self.temporal_pe, std=0.02)
        self.transformer = _TransformerEncoder(
            cfg.hidden_dim, cfg.nhead, cfg.dec_layers, cfg.dropout)
        self.head = nn.Linear(cfg.hidden_dim, cfg.sam_dim)

    def forward(self, u):
        # u: (B, L)
        B = u.shape[0]
        cond = self.from_u(u)                                    # (B, H)
        # Build query tokens = positional embeddings + cond broadcast.
        q = (self.spatial_pe + self.temporal_pe).expand(B, -1, -1, -1)
        q = q + cond[:, None, None, :]
        q = q.reshape(B, self.tokens_per_window, -1)
        q = self.transformer(q)                                   # (B, W*N, H)
        q = self.head(q)                                          # (B, W*N, 256)
        cfg = self.cfg
        q = q.reshape(B, cfg.window, cfg.sam_grid, cfg.sam_grid, cfg.sam_dim)
        q = q.permute(0, 1, 4, 2, 3).contiguous()                # (B, W, 256, 8, 8)
        return q


class ActionDecoder(nn.Module):
    """u in R^latent_dim -> action chunk (W, 14) reconstruction."""
    def __init__(self, cfg: MVAEConfig):
        super().__init__()
        self.cfg = cfg
        self.from_u = nn.Linear(cfg.latent_dim, cfg.hidden_dim)
        self.temporal_pe = nn.Parameter(torch.zeros(1, cfg.window, cfg.hidden_dim))
        nn.init.trunc_normal_(self.temporal_pe, std=0.02)
        self.transformer = _TransformerEncoder(
            cfg.hidden_dim, cfg.nhead, cfg.dec_layers, cfg.dropout)
        self.head = nn.Linear(cfg.hidden_dim, cfg.action_dim)

    def forward(self, u):
        B = u.shape[0]
        cond = self.from_u(u)                                    # (B, H)
        q = self.temporal_pe.expand(B, -1, -1) + cond[:, None, :]
        q = self.transformer(q)
        return self.head(q)                                       # (B, W, 14)


class PoEMVAE(nn.Module):
    def __init__(self, cfg: MVAEConfig):
        super().__init__()
        self.cfg = cfg
        self.video_enc = VideoLatentEncoder(cfg)
        self.action_enc = ActionEncoder(cfg)
        self.video_dec = VideoLatentDecoder(cfg)
        self.action_dec = ActionDecoder(cfg)

    def fuse_poe(self, mu_v, lv_v, mu_a, lv_a):
        """Precision-weighted Gaussian fusion of two diagonal Gaussians plus the
        standard-normal prior. Returns (mu_joint, logvar_joint).
        """
        # Clamp logvars for numerical safety.
        lv_v = lv_v.clamp(-10.0, 10.0)
        lv_a = lv_a.clamp(-10.0, 10.0)
        inv_var_v = torch.exp(-lv_v)
        inv_var_a = torch.exp(-lv_a)
        inv_var_p = torch.ones_like(inv_var_v)  # standard-normal prior
        inv_var_joint = inv_var_v + inv_var_a + inv_var_p
        mu_joint = (mu_v * inv_var_v + mu_a * inv_var_a) / inv_var_joint
        logvar_joint = -torch.log(inv_var_joint)
        return mu_joint, logvar_joint

    def reparameterize(self, mu, logvar):
        if self.training:
            eps = torch.randn_like(mu)
            return mu + eps * torch.exp(0.5 * logvar)
        return mu

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> dict:
        mu_v, lv_v = self.video_enc(latent)
        mu_a, lv_a = self.action_enc(action)
        mu, lv = self.fuse_poe(mu_v, lv_v, mu_a, lv_a)
        u = self.reparameterize(mu, lv)
        latent_recon = self.video_dec(u)
        action_recon = self.action_dec(u)
        # KL(N(mu, var) || N(0,I)) summed over latent_dim, averaged over batch.
        kl_per_dim = 0.5 * (mu.pow(2) + lv.exp() - 1.0 - lv)
        kl = kl_per_dim.sum(dim=-1).mean()
        mse_v = F.mse_loss(latent_recon, latent)
        mse_a = F.mse_loss(action_recon, action)
        loss = mse_v + self.cfg.action_loss_weight * mse_a + self.cfg.beta * kl
        return {
            "loss": loss,
            "mse_latent": mse_v.detach(),
            "mse_action": mse_a.detach(),
            "kl": kl.detach(),
            "mu_v": mu_v.detach(), "logvar_v": lv_v.detach(),
            "mu_a": mu_a.detach(), "logvar_a": lv_a.detach(),
            "mu": mu.detach(), "logvar": lv.detach(),
            "latent_recon": latent_recon,
            "action_recon": action_recon,
        }
