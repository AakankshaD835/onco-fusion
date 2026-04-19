"""
Subtype Classification Head — 3-class breast cancer subtype prediction.

Classes (derived from ER/PR/HER2 receptor status in TCGA-BRCA):
    0: HR+   — ER+ or PR+ (HER2-); LumA + LumB merged due to small per-class n
    1: HER2+ — HER2 amplified, targeted therapy (Herceptin)
    2: TNBC  — triple-negative, worst prognosis, no targeted therapy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

SUBTYPE_LABELS = {0: "HR+", 1: "HER2+", 2: "TNBC"}
SUBTYPE_CODES  = {"HR+": 0, "HER2+": 1, "TNBC": 2}


class SubtypeHead(nn.Module):
    """
    3-class classifier head for breast cancer subtype prediction (HR+/HER2+/TNBC).

    Args:
        d_model:    Input embedding dimension.
        n_classes:  Number of subtype classes (default 3).
        dropout:    Dropout rate before final linear layer.
    """

    def __init__(self, d_model: int = 256, n_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, d_model) — fused patient embedding.
        Returns:
            dict with:
                "logits":  (batch, n_classes)
                "probs":   (batch, n_classes) — softmax probabilities
                "pred":    (batch,) — predicted class index
        """
        logits = self.classifier(x)
        probs  = F.softmax(logits, dim=-1)
        pred   = probs.argmax(dim=-1)
        return {"logits": logits, "probs": probs, "pred": pred}
