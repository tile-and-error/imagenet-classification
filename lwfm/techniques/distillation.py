"""Knowledge-distillation losses: plain `nn.Module`s on logits [B, C] or embeddings
[B, D], with labels [B] where needed. Bind them to names with a LossBinding, e.g.

    LossBinding("kd", KD(T=4), ["student.logits", "teacher.logits"])
    LossBinding("dkd", DKD(), ["student.logits", "teacher.logits", "targets.class"])
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class KD(nn.Module):
    """Hinton KD: T^2 * KL(softmax(z_t / T) || softmax(z_s / T)), averaged over the
    batch. With per-class `weight`, each sample's KL is weighted by its true class's
    weight instead, so it also needs the labels."""

    weight: Tensor | None

    def __init__(self, T: float = 4.0, weight: Tensor | None = None):
        super().__init__()
        self.T = T
        self.weighted = weight is not None
        self.register_buffer("weight", weight)  # moves with .to(device)

    def forward(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        labels: Tensor | None = None,
    ) -> Tensor:
        T = self.T
        log_p_s = F.log_softmax(student_logits / T, 1)
        p_t = F.softmax(teacher_logits / T, 1)
        if self.weight is None:
            return T**2 * F.kl_div(log_p_s, p_t, reduction="batchmean")
        if labels is None:
            raise ValueError(
                "weighted KD needs the labels: bind it to targets.class too"
            )
        # Unweighted KD is dominated by a majority class (e.g. a large "other" class),
        # which students then learn to always predict. Class weights, as in weighted
        # CE, give each class an equal share: T^2 * sum_i w_yi KL_i / sum_i w_yi
        kl = F.kl_div(log_p_s, p_t, reduction="none").sum(1)  # one KL per sample
        w = self.weight[labels]
        return T**2 * (w * kl).sum() / w.sum()


class DKD(nn.Module):
    """Decoupled KD (Zhao et al., 2022): T^2 * (alpha * TCKD + beta * NCKD), where
    TCKD matches the target class against all others, and NCKD matches the
    distribution over the non-target classes."""

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, T: float = 1.0):
        super().__init__()
        self.alpha, self.beta, self.T = alpha, beta, T

    def forward(
        self, student_logits: Tensor, teacher_logits: Tensor, labels: Tensor
    ) -> Tensor:
        T = self.T
        gt = F.one_hot(labels, student_logits.shape[1]).bool()
        p_s = F.softmax(student_logits / T, 1)
        p_t = F.softmax(teacher_logits / T, 1)
        # Target vs everything else, as two-way distributions
        b_s = torch.stack([p_s[gt], 1 - p_s[gt]], 1).clamp_min(1e-12)
        b_t = torch.stack([p_t[gt], 1 - p_t[gt]], 1)
        tckd = F.kl_div(b_s.log(), b_t, reduction="batchmean")
        # Non-target classes only: -1e4 removes the target from both softmaxes
        nckd = F.kl_div(
            F.log_softmax(student_logits / T - 1e4 * gt, 1),
            F.softmax(teacher_logits / T - 1e4 * gt, 1),
            reduction="batchmean",
        )
        return T**2 * (self.alpha * tckd + self.beta * nckd)


def pearson(a: Tensor, b: Tensor, dim: int, eps: float = 1e-8) -> Tensor:
    """Pearson correlation of a and b along `dim`."""
    a, b = a - a.mean(dim, keepdim=True), b - b.mean(dim, keepdim=True)
    return (a * b).sum(dim) / (a.norm(dim=dim) * b.norm(dim=dim) + eps)


class DIST(nn.Module):
    """DIST (Huang et al., 2022): matches correlations rather than exact
    probabilities, across classes per sample (inter, weight beta) and across the
    batch per class (intra, weight gamma)."""

    def __init__(self, beta: float = 1.0, gamma: float = 1.0, T: float = 1.0):
        super().__init__()
        self.beta, self.gamma, self.T = beta, gamma, T

    def forward(self, student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
        p_s = F.softmax(student_logits / self.T, 1)
        p_t = F.softmax(teacher_logits / self.T, 1)
        inter = 1 - pearson(p_s, p_t, dim=1).mean()
        intra = 1 - pearson(p_s, p_t, dim=0).mean()
        return self.T**2 * (self.beta * inter + self.gamma * intra)


class EmbedCos(nn.Module):
    """1 - cos(h_s, h_t), averaged over the batch: matches the direction of the
    (projected) student embedding to the teacher's, whatever their norms."""

    def forward(self, student_embed: Tensor, teacher_embed: Tensor) -> Tensor:
        return 1 - F.cosine_similarity(student_embed, teacher_embed, dim=1).mean()


class EmbedMSE(nn.Module):
    """MSE between the (projected) student embedding and the teacher embedding
    standardised per dimension: (h_t - mean) / std, with mean and std usually taken
    over the training set."""

    mean: Tensor
    std: Tensor

    def __init__(self, mean: Tensor, std: Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, student_embed: Tensor, teacher_embed: Tensor) -> Tensor:
        return F.mse_loss(student_embed, (teacher_embed - self.mean) / self.std)
