"""
Genomic Encoder — 1D-CNN over RNA-seq gene expression profiles.

Input:  (batch, n_genes)  — raw or log1p-normalised expression values
Output: (batch, embed_dim) — compact genomic embedding

Architecture:
    Linear(n_genes -> 4096) -> reshape -> Conv1D stack -> GlobalAvgPool -> Linear(embed_dim)

The initial linear projection reduces the ~20k gene dimension to a manageable
sequence length before the convolutional layers extract local co-expression patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GenomicEncoder(nn.Module):
    """
    1D-CNN encoder for bulk RNA-seq gene expression data.

    Args:
        n_genes:    Number of input genes (default 20_530 for TCGA-BRCA).
        embed_dim:  Output embedding dimension (default 256).
        dropout:    Dropout rate applied after each conv block (default 0.3).
    """

    def __init__(self, n_genes: int = 20_530, embed_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.embed_dim = embed_dim

        # Project to sequence: (batch, 1, 4096)
        self.input_proj = nn.Linear(n_genes, 4096)

        # Conv blocks: extract local co-expression patterns
        self.conv_blocks = nn.Sequential(
            self._conv_block(1,   64, kernel_size=7, dropout=dropout),   # (batch, 64,  4096)
            self._conv_block(64,  128, kernel_size=5, dropout=dropout),  # (batch, 128, 2048)
            self._conv_block(128, 256, kernel_size=3, dropout=dropout),  # (batch, 256, 1024)
            self._conv_block(256, 512, kernel_size=3, dropout=dropout),  # (batch, 512, 512)
        )

        # Global average pooling + final projection
        self.head = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    @staticmethod
    def _conv_block(in_ch: int, out_ch: int, kernel_size: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_genes) — gene expression values.
        Returns:
            (batch, embed_dim)
        """
        # Project and reshape to sequence format
        x = self.input_proj(x)          # (batch, 4096)
        x = x.unsqueeze(1)              # (batch, 1, 4096)

        x = self.conv_blocks(x)         # (batch, 512, ...)
        x = x.mean(dim=-1)              # global average pool -> (batch, 512)

        return self.head(x)             # (batch, embed_dim)
