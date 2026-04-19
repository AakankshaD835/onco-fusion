"""
Survival Prediction Head — Cox proportional hazards regression.

Input:  (batch, d_model) fused embedding
Output: (batch, 1) continuous risk score

Loss: Partial Cox likelihood (Breslow approximation).
Metric: C-index (concordance index) — fraction of correctly ordered patient pairs.

A higher risk score means higher predicted mortality risk.
"""

import torch
import torch.nn as nn


class SurvivalHead(nn.Module):
    """
    Outputs a scalar risk score from the fused patient embedding.

    The Cox model assumes: h(t|x) = h0(t) * exp(risk_score(x))
    We learn risk_score(x) — the log hazard ratio relative to baseline.

    Args:
        d_model:  Input embedding dimension.
        dropout:  Dropout before the final linear layer.
    """

    def __init__(self, d_model: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, d_model)
        Returns:
            risk_scores: (batch, 1)
        """
        return self.net(x)


def cox_loss(risk_scores: torch.Tensor, durations: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    """
    Negative partial log-likelihood for Cox proportional hazards model.
    Breslow approximation for tied event times.

    Args:
        risk_scores: (batch,) — predicted log hazard ratios.
        durations:   (batch,) — observed survival times.
        events:      (batch,) — event indicator (1=death, 0=censored).

    Returns:
        Scalar loss value.
    """
    # Sort by descending duration
    order       = torch.argsort(durations, descending=True)
    risk_scores = risk_scores[order]
    events      = events[order]

    # Log-sum-exp of risk scores for each risk set
    log_risk    = torch.logcumsumexp(risk_scores, dim=0)

    # Partial likelihood: only sum over uncensored patients
    loss = -torch.mean((risk_scores - log_risk) * events)
    return loss


def concordance_index(risk_scores: torch.Tensor, durations: torch.Tensor, events: torch.Tensor) -> float:
    """
    Compute the C-index (concordance index) for survival predictions.

    C-index = P(risk_i > risk_j | duration_i < duration_j, event_i = 1)

    Returns a float in [0, 1]; 0.5 = random, 1.0 = perfect ordering.
    """
    risk = risk_scores.detach().cpu().numpy()
    dur  = durations.detach().cpu().numpy()
    evt  = events.detach().cpu().numpy()

    concordant = 0
    permissible = 0
    for i in range(len(dur)):
        if evt[i] == 0:
            continue
        for j in range(len(dur)):
            if dur[i] >= dur[j]:
                continue
            permissible += 1
            if risk[i] > risk[j]:
                concordant += 1
            elif risk[i] == risk[j]:
                concordant += 0.5

    return concordant / permissible if permissible > 0 else 0.5
