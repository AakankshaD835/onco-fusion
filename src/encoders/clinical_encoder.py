"""
Clinical Encoder — TabNet over structured clinical tabular features.

Input:  (batch, n_features) — normalised clinical features
Output: (batch, embed_dim)  — clinical embedding

Features used from TCGA-BRCA clinical.csv:
    AGE, AJCC_PATHOLOGIC_TUMOR_STAGE, PATH_T_STAGE, PATH_N_STAGE, PATH_M_STAGE,
    ER status, PR status, HER2 status, RADIATION_THERAPY, tumour size, grade
"""

import torch
import torch.nn as nn
from pytorch_tabnet.tab_network import TabNet


class ClinicalEncoder(nn.Module):
    """
    TabNet-based encoder for structured clinical tabular data.

    TabNet uses sequential attention to select relevant features at each
    decision step, providing built-in feature importance (no SHAP needed
    for clinical features — attention masks are the explanation).

    Args:
        n_features:     Number of input clinical features.
        embed_dim:      Output embedding dimension (default 128).
        n_steps:        Number of TabNet decision steps (default 3).
        n_a:            Attention embedding dimension (default 64).
        n_shared:       Number of shared GLU layers (default 2).
    """

    def __init__(
        self,
        n_features: int = 20,
        embed_dim:  int = 128,
        n_steps:    int = 3,
        n_a:        int = 64,
        n_shared:   int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.tabnet = TabNet(
            input_dim=n_features,
            output_dim=embed_dim,
            n_steps=n_steps,
            attn_dim=n_a,
            n_shared=n_shared,
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, n_features) — normalised clinical features.
        Returns:
            embedding:       (batch, embed_dim)
            attention_masks: (batch, n_steps, n_features) — feature importances per step
        """
        embedding, attention_masks = self.tabnet(x)
        return self.norm(embedding), attention_masks
