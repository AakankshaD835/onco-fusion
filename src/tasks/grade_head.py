"""
Grade Classification Head — 3-class tumour grade prediction.

Classes:
    0: Grade 1 — well-differentiated, slowest growing
    1: Grade 2 — moderately differentiated
    2: Grade 3 — poorly differentiated, fastest growing, worst prognosis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

GRADE_LABELS = {0: "Grade 1", 1: "Grade 2", 2: "Grade 3"}


class GradeHead(nn.Module):
    """
    3-class classifier head for histological tumour grade.

    Args:
        d_model:    Input embedding dimension.
        n_classes:  Number of grade classes (default 3).
        dropout:    Dropout rate.
    """

    def __init__(self, d_model: int = 256, n_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, d_model)
        Returns:
            dict with logits, probs, pred
        """
        logits = self.classifier(x)
        probs  = F.softmax(logits, dim=-1)
        return {"logits": logits, "probs": probs, "pred": probs.argmax(dim=-1)}
