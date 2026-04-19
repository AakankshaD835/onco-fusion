"""
Early Fusion — simple concatenation baseline.

Concatenates all modality embeddings and passes through an MLP.
This is the simplest possible fusion strategy and serves as the
lower baseline that cross-attention fusion must beat.
"""

import torch
import torch.nn as nn


class EarlyFusion(nn.Module):
    """
    Concatenation + MLP fusion baseline.

    Args:
        image_dim, clinical_dim, genomic_dim, text_dim: Input embedding dims.
        output_dim: Output representation dimension (default 256).
        dropout:    Dropout rate (default 0.3).
    """

    def __init__(
        self,
        image_dim:    int = 512,
        clinical_dim: int = 128,
        genomic_dim:  int = 256,
        text_dim:     int = 256,
        output_dim:   int = 256,
        dropout:      float = 0.3,
    ):
        super().__init__()
        total_dim = image_dim + clinical_dim + genomic_dim + text_dim

        self.mlp = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        image_embed:    torch.Tensor,
        clinical_embed: torch.Tensor,
        genomic_embed:  torch.Tensor,
        text_embed:     torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        fused = torch.cat([image_embed, clinical_embed, genomic_embed, text_embed], dim=-1)
        return {"fused": self.mlp(fused)}
