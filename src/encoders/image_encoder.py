"""
Image Encoder — wraps BiomedCLIP, Phikon, and PLIP for patch-level embeddings.

Supports three pathology-domain pretrained vision encoders:
  - BiomedCLIP (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
  - Phikon     (owkin/phikon)
  - PLIP       (vinid/plip)

All return a (batch, embed_dim) tensor suitable for the fusion module.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor
from PIL import Image
from typing import Literal


ENCODER_IDS = {
    "biomedclip": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    "phikon":     "owkin/phikon",
    "plip":       "vinid/plip",
    "vit":        "google/vit-base-patch16-224",
}


class ImageEncoder(nn.Module):
    """
    Wraps a pretrained pathology vision encoder and projects its output
    to a fixed embedding dimension.

    Args:
        encoder_name: One of 'biomedclip', 'phikon', 'plip', 'vit'.
        embed_dim:    Output embedding dimension (default 512).
        freeze:       If True, freeze pretrained weights and only train
                      the projection head.
    """

    def __init__(
        self,
        encoder_name: Literal["biomedclip", "phikon", "plip", "vit"] = "biomedclip",
        embed_dim: int = 512,
        freeze: bool = True,
    ):
        super().__init__()
        self.encoder_name = encoder_name
        model_id = ENCODER_IDS[encoder_name]

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.backbone  = AutoModel.from_pretrained(model_id)

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Determine backbone output dim by a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out   = self.backbone.get_image_features(dummy) if hasattr(
                self.backbone, "get_image_features"
            ) else self.backbone(pixel_values=dummy).last_hidden_state[:, 0]
            backbone_dim = out.shape[-1]

        self.proj = nn.Sequential(
            nn.Linear(backbone_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.embed_dim = embed_dim

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (batch, 3, 224, 224) normalised image tensor.
        Returns:
            embeddings:   (batch, embed_dim)
        """
        if hasattr(self.backbone, "get_image_features"):
            features = self.backbone.get_image_features(pixel_values)
        else:
            features = self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0]

        return self.proj(features)

    def preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        """Convenience: convert PIL images to model-ready tensors."""
        inputs = self.processor(images=images, return_tensors="pt")
        return inputs["pixel_values"]
