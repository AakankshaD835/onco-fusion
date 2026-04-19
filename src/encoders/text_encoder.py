"""
Text Encoder — BioClinicalBERT over synthetic patient text reports.

Synthetic reports are generated from TCGA clinical fields, e.g.:
    "Female patient, 52 years old. Stage IIB breast cancer.
     ER-positive, PR-negative, HER2-negative. Grade 2 tumour,
     size 2.3 cm. Received radiation therapy."

This gives the model a natural language view of the clinical record,
which BioClinicalBERT encodes using domain-specific pretraining on
MIMIC-III clinical notes.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


MODEL_ID = "emilyalsentzer/Bio_ClinicalBERT"


class TextEncoder(nn.Module):
    """
    Encodes synthetic clinical text reports using BioClinicalBERT.

    Uses the [CLS] token embedding as the sentence representation,
    projected to a fixed embedding dimension.

    Args:
        embed_dim: Output embedding dimension (default 256).
        freeze:    If True, freeze BERT weights (default True).
        max_len:   Maximum token length (default 128).
    """

    def __init__(self, embed_dim: int = 256, freeze: bool = True, max_len: int = 128):
        super().__init__()
        self.max_len   = max_len
        self.embed_dim = embed_dim

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.bert      = AutoModel.from_pretrained(MODEL_ID)

        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

        bert_hidden = self.bert.config.hidden_size  # 768 for BERT-base

        self.proj = nn.Sequential(
            nn.Linear(bert_hidden, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids:      (batch, seq_len)
            attention_mask: (batch, seq_len)
        Returns:
            embeddings: (batch, embed_dim)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]  # [CLS] representation
        return self.proj(cls_token)

    def tokenize(self, reports: list[str]) -> dict[str, torch.Tensor]:
        """Tokenize a batch of text reports."""
        return self.tokenizer(
            reports,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

    @staticmethod
    def build_report(row: dict) -> str:
        """
        Construct a natural language clinical report from a TCGA patient record dict.

        Args:
            row: Dict with keys: age, stage, er_status, pr_status, her2_status,
                 grade, tumour_size, radiation_therapy.
        Returns:
            A short free-text clinical summary string.
        """
        return (
            f"Female patient, {row.get('age', 'unknown age')} years old. "
            f"Stage {row.get('stage', 'unknown')} breast cancer. "
            f"ER-{row.get('er_status', 'unknown')}, "
            f"PR-{row.get('pr_status', 'unknown')}, "
            f"HER2-{row.get('her2_status', 'unknown')}. "
            f"Grade {row.get('grade', 'unknown')} tumour. "
            f"Tumour size {row.get('tumour_size', 'unknown')} cm. "
            f"Radiation therapy: {row.get('radiation_therapy', 'unknown')}."
        )
