import torch
import torch.nn as nn
import torch.nn.functional as F


def info_nce(
    query: torch.Tensor,
    positive_key: torch.Tensor,
    temperature: float = 0.1,
    reduction: str = "mean",
) -> torch.Tensor:
    """A minimal InfoNCE implementation for pairwise contrastive learning."""
    if query.dim() != 2 or positive_key.dim() != 2:
        raise ValueError("query and positive_key must be 2D tensors: [B, D]")
    if query.shape[0] != positive_key.shape[0]:
        raise ValueError("query and positive_key must have the same batch size")

    query = F.normalize(query, dim=-1)
    positive_key = F.normalize(positive_key, dim=-1)

    logits = torch.matmul(query, positive_key.t()) / max(float(temperature), 1e-8)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels, reduction=reduction)


class InfoNCE(nn.Module):
    def __init__(self, temperature: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.temperature = float(temperature)
        self.reduction = reduction

    def forward(self, query: torch.Tensor, positive_key: torch.Tensor) -> torch.Tensor:
        return info_nce(
            query=query,
            positive_key=positive_key,
            temperature=self.temperature,
            reduction=self.reduction,
        )
