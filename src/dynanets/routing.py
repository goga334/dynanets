from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class GateConfig:
    mode: str = "metric"
    metric: str = "confidence"
    threshold: float = 0.85
    min_threshold: float | None = None
    temperature: float = 0.25
    budget_weight: float = 0.0
    target_cost_ratio: float = 1.0
    distillation_weight: float = 0.0
    supervision_weight: float = 0.0
    teacher_confidence_floor: float = 0.55
    entropy_weight: float = 0.0
    target_accept_rate: float | None = None
    trace_limit: int = 8

    def threshold_for_progress(self, progress: float) -> float:
        if self.min_threshold is None:
            return float(self.threshold)
        clamped = max(0.0, min(1.0, float(progress)))
        return float(self.threshold + ((self.min_threshold - self.threshold) * clamped))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def gate_scores(logits: torch.Tensor, *, metric: str, temperature: float = 1.0) -> torch.Tensor:
    scaled_logits = logits / max(1e-6, float(temperature))
    probabilities = torch.softmax(scaled_logits, dim=1)
    if metric == "confidence":
        return probabilities.max(dim=1).values
    if metric == "margin":
        top2 = torch.topk(probabilities, k=min(2, probabilities.shape[1]), dim=1).values
        if top2.shape[1] == 1:
            return top2[:, 0]
        return top2[:, 0] - top2[:, 1]
    if metric == "entropy":
        entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-8))).sum(dim=1)
        max_entropy = math.log(max(2, probabilities.shape[1]))
        return 1.0 - (entropy / max_entropy)
    raise ValueError(f"Unsupported gate metric '{metric}'")


def relaxed_accept_probability(scores: torch.Tensor, *, threshold: float, temperature: float) -> torch.Tensor:
    return torch.sigmoid((scores - float(threshold)) / max(1e-6, float(temperature)))


def budget_penalty(*, expected_cost_ratio: torch.Tensor, target_cost_ratio: float, weight: float) -> torch.Tensor:
    if weight <= 0.0:
        return torch.zeros((), device=expected_cost_ratio.device)
    overflow = torch.relu(expected_cost_ratio - float(target_cost_ratio))
    return float(weight) * overflow.pow(2)


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, *, temperature: float = 2.0) -> torch.Tensor:
    softened_student = F.log_softmax(student_logits / temperature, dim=1)
    softened_teacher = F.softmax(teacher_logits.detach() / temperature, dim=1)
    return F.kl_div(softened_student, softened_teacher, reduction="batchmean") * (temperature ** 2)


def gate_accept_targets(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    teacher_confidence_floor: float,
) -> torch.Tensor:
    teacher_probs = torch.softmax(teacher_logits.detach(), dim=1)
    teacher_confidence, teacher_pred = teacher_probs.max(dim=1)
    student_pred = student_logits.detach().argmax(dim=1)
    accepts = (student_pred == teacher_pred) & (teacher_confidence >= float(teacher_confidence_floor))
    return accepts.float()


def gate_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    clipped = probabilities.clamp_min(1e-8)
    return -(clipped * torch.log(clipped) + (1.0 - clipped) * torch.log((1.0 - clipped).clamp_min(1e-8))).mean()


def accept_rate_penalty(probabilities: torch.Tensor, *, target_accept_rate: float | None, weight: float) -> torch.Tensor:
    if target_accept_rate is None or weight <= 0.0:
        return torch.zeros((), device=probabilities.device)
    deviation = probabilities.mean() - float(target_accept_rate)
    return float(weight) * deviation.pow(2)


__all__ = [
    "accept_rate_penalty",
    "GateConfig",
    "budget_penalty",
    "distillation_loss",
    "gate_accept_targets",
    "gate_entropy",
    "gate_scores",
    "relaxed_accept_probability",
]
