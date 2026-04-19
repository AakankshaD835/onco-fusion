"""
Late Fusion — independent classifiers + probability averaging baseline.

Each modality trains its own classifier independently.
Final prediction = learnable weighted average of per-modality softmax outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LateFusion(nn.Module):
    """
    Late fusion via learned weighted combination of per-modality predictions.

    Args:
        embed_dims: Dict of modality name -> embedding dimension.
        n_classes:  Number of output classes.
    """

    def __init__(self, embed_dims: dict[str, int], n_classes: int = 4):
        super().__init__()
        self.classifiers = nn.ModuleDict({
            name: nn.Linear(dim, n_classes)
            for name, dim in embed_dims.items()
        })
        # Learnable fusion weights (one per modality)
        self.fusion_weights = nn.Parameter(torch.ones(len(embed_dims)))

    def forward(self, embeddings: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        weights  = F.softmax(self.fusion_weights, dim=0)
        probs_all = []
        for i, (name, embed) in enumerate(embeddings.items()):
            logits = self.classifiers[name](embed)
            probs_all.append(F.softmax(logits, dim=-1) * weights[i])

        fused_probs = torch.stack(probs_all, dim=0).sum(dim=0)  # (batch, n_classes)
        return {
            "probs": fused_probs,
            "pred":  fused_probs.argmax(dim=-1),
            "weights": weights.detach(),
        }
