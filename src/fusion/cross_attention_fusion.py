"""
Cross-Attention Fusion Module — the core model.

Architecture:
    Each modality embedding is projected to a shared d_model dimension,
    forming a sequence of 4 modality tokens. These are passed through a
    standard Transformer encoder with multi-head self-attention, allowing
    every modality to attend to every other.

    The output CLS token (prepended learnable token) is the fused
    patient representation used by all task heads.

    Modality Dropout:
        During training, modalities can be randomly zeroed out to simulate
        missing data. This makes the model robust to incomplete inputs at
        test time — a critical property for real clinical deployment.
"""

import torch
import torch.nn as nn
import random
from typing import Optional


MODALITIES = ["image", "clinical", "genomic", "text"]


class CrossAttentionFusion(nn.Module):
    """
    Multi-modal fusion via Transformer cross-attention over modality tokens.

    Args:
        image_dim:      Dimension of image embeddings (default 512).
        clinical_dim:   Dimension of clinical embeddings (default 128).
        genomic_dim:    Dimension of genomic embeddings (default 256).
        text_dim:       Dimension of text embeddings (default 256).
        d_model:        Internal transformer dimension (default 256).
        n_heads:        Number of attention heads (default 8).
        n_layers:       Number of transformer encoder layers (default 4).
        dropout:        Dropout rate (default 0.1).
        modality_dropout_p: Probability of dropping each modality during
                            training (default 0.3). Set to 0.0 to disable.
    """

    def __init__(
        self,
        image_dim:           int   = 512,
        clinical_dim:        int   = 128,
        genomic_dim:         int   = 256,
        text_dim:            int   = 256,
        d_model:             int   = 256,
        n_heads:             int   = 8,
        n_layers:            int   = 4,
        dropout:             float = 0.1,
        modality_dropout_p:  float = 0.3,
    ):
        super().__init__()
        self.d_model            = d_model
        self.modality_dropout_p = modality_dropout_p

        # Per-modality linear projections to shared d_model
        self.proj = nn.ModuleDict({
            "image":    nn.Linear(image_dim,    d_model),
            "clinical": nn.Linear(clinical_dim, d_model),
            "genomic":  nn.Linear(genomic_dim,  d_model),
            "text":     nn.Linear(text_dim,     d_model),
        })

        # Learnable modality-type embeddings (like token type IDs in BERT)
        self.modality_embeddings = nn.Embedding(len(MODALITIES), d_model)

        # Learnable CLS token — its output is the fused representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        image_embed:    torch.Tensor,
        clinical_embed: torch.Tensor,
        genomic_embed:  torch.Tensor,
        text_embed:     torch.Tensor,
        return_attention_weights: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            image_embed:    (batch, image_dim)
            clinical_embed: (batch, clinical_dim)
            genomic_embed:  (batch, genomic_dim)
            text_embed:     (batch, text_dim)
            return_attention_weights: If True, return per-modality attention.

        Returns:
            dict with:
                "fused":   (batch, d_model)  — patient representation
                "tokens":  (batch, 5, d_model) — all output tokens (CLS + 4 modality)
                "weights": (batch, 4) — mean attention weights per modality (if requested)
        """
        batch = image_embed.size(0)
        embeds = {
            "image":    image_embed,
            "clinical": clinical_embed,
            "genomic":  genomic_embed,
            "text":     text_embed,
        }

        # Modality dropout during training
        if self.training and self.modality_dropout_p > 0:
            embeds = self._apply_modality_dropout(embeds)

        # Project each modality to d_model and add modality-type embedding
        tokens = []
        for i, name in enumerate(MODALITIES):
            tok = self.proj[name](embeds[name])                          # (batch, d_model)
            tok = tok + self.modality_embeddings(
                torch.tensor(i, device=tok.device)
            )
            tokens.append(tok.unsqueeze(1))                              # (batch, 1, d_model)

        # Prepend CLS token: (batch, 5, d_model)
        cls = self.cls_token.expand(batch, -1, -1)
        sequence = torch.cat([cls] + tokens, dim=1)

        # Transformer
        output = self.transformer(sequence)                              # (batch, 5, d_model)
        output = self.norm(output)

        fused = output[:, 0, :]                                          # CLS output

        result = {"fused": fused, "tokens": output}

        if return_attention_weights:
            # Mean attention weight each modality token received (from CLS position)
            modality_tokens = output[:, 1:, :]                          # (batch, 4, d_model)
            weights = modality_tokens.norm(dim=-1)                      # (batch, 4)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
            result["weights"] = weights

        return result

    def _apply_modality_dropout(self, embeds: dict) -> dict:
        """Randomly zero-mask modality embeddings during training."""
        n_active = random.randint(2, len(MODALITIES))
        active   = set(random.sample(MODALITIES, n_active))
        return {
            name: (emb if name in active else torch.zeros_like(emb))
            for name, emb in embeds.items()
        }
