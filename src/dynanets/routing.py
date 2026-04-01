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
    eval_threshold_offset: float = 0.0
    eval_min_threshold: float | None = None
    temperature: float = 0.25
    metric_blend: float = 0.0
    budget_weight: float = 0.0
    target_cost_ratio: float = 1.0
    target_cost_ratio_min: float | None = None
    distillation_weight: float = 0.0
    supervision_weight: float = 0.0
    teacher_confidence_floor: float = 0.55
    student_confidence_floor: float = 0.35
    target_strategy: str = "hybrid"
    entropy_weight: float = 0.0
    target_accept_rate: float | None = None
    target_accept_rate_min: float | None = None
    accept_rate_weight: float = 0.0
    trace_limit: int = 8

    def threshold_for_progress(self, progress: float) -> float:
        return _interpolate(self.threshold, self.min_threshold, progress)

    def eval_threshold_for_progress(self, progress: float) -> float:
        base = self.threshold_for_progress(progress) + float(self.eval_threshold_offset)
        minimum = self.eval_min_threshold
        if minimum is not None:
            base = max(float(minimum), base)
        return float(max(0.0, min(1.0, base)))

    def target_cost_ratio_for_progress(self, progress: float) -> float:
        return _interpolate(self.target_cost_ratio, self.target_cost_ratio_min, progress)

    def target_accept_rate_for_progress(self, progress: float) -> float | None:
        if self.target_accept_rate is None:
            return None
        return _interpolate(self.target_accept_rate, self.target_accept_rate_min, progress)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _interpolate(base: float, minimum: float | None, progress: float) -> float:
    if minimum is None:
        return float(base)
    clamped = max(0.0, min(1.0, float(progress)))
    return float(base + ((minimum - base) * clamped))


def staged_accept_rate(target_accept_rate: float | None, *, remaining_gated_stages: int) -> float | None:
    if target_accept_rate is None:
        return None
    if remaining_gated_stages <= 0:
        return None
    clamped = min(0.999, max(0.0, float(target_accept_rate)))
    if clamped <= 0.0:
        return 0.0
    return float(1.0 - ((1.0 - clamped) ** (1.0 / remaining_gated_stages)))


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


def blend_gate_scores(learned_scores: torch.Tensor, metric_scores: torch.Tensor, *, metric_blend: float) -> torch.Tensor:
    blend = min(1.0, max(0.0, float(metric_blend)))
    if blend <= 0.0:
        return learned_scores
    if blend >= 1.0:
        return metric_scores
    return ((1.0 - blend) * learned_scores) + (blend * metric_scores)


def relaxed_accept_probability(scores: torch.Tensor, *, threshold: float, temperature: float) -> torch.Tensor:
    return torch.sigmoid((scores - float(threshold)) / max(1e-6, float(temperature)))

def select_accept_mask(
    scores: torch.Tensor,
    *,
    threshold: float,
    target_accept_rate: float | None = None,
    eligible_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    accepted = scores >= float(threshold)
    if target_accept_rate is None:
        return accepted
    desired = int(round(float(target_accept_rate) * scores.shape[0]))
    desired = max(0, min(scores.shape[0], desired))
    if desired <= 0 or int(accepted.sum().item()) >= desired:
        return accepted
    available_mask = torch.ones_like(accepted, dtype=torch.bool) if eligible_mask is None else eligible_mask.bool()
    candidate_mask = available_mask & (~accepted)
    candidate_indices = candidate_mask.nonzero(as_tuple=False).flatten()
    remaining = desired - int(accepted.sum().item())
    if remaining <= 0 or candidate_indices.numel() == 0:
        return accepted
    remaining = min(remaining, int(candidate_indices.numel()))
    candidate_scores = scores[candidate_indices]
    topk_local = torch.topk(candidate_scores, k=remaining, dim=0).indices
    forced = torch.zeros_like(accepted, dtype=torch.bool)
    forced[candidate_indices[topk_local]] = True
    return accepted | forced


def budget_penalty(*, expected_cost_ratio: torch.Tensor, target_cost_ratio: float, weight: float) -> torch.Tensor:
    if weight <= 0.0:
        return torch.zeros((), device=expected_cost_ratio.device)
    overflow = torch.relu(expected_cost_ratio - float(target_cost_ratio))
    return float(weight) * overflow.pow(2)


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, *, temperature: float = 2.0) -> torch.Tensor:
    softened_student = F.log_softmax(student_logits / temperature, dim=1)
    softened_teacher = F.softmax(teacher_logits.detach() / temperature, dim=1)
    return F.kl_div(softened_student, softened_teacher, reduction="batchmean") * (temperature ** 2)

def weighted_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    weights: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    softened_student = F.log_softmax(student_logits / temperature, dim=1)
    softened_teacher = F.softmax(teacher_logits.detach() / temperature, dim=1)
    per_sample = F.kl_div(softened_student, softened_teacher, reduction="none").sum(dim=1) * (temperature ** 2)
    normalized = weights / weights.sum().clamp_min(1e-6)
    return (per_sample * normalized).sum()


def gate_accept_targets(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    targets: torch.Tensor | None = None,
    teacher_confidence_floor: float,
    student_confidence_floor: float,
    strategy: str = "hybrid",
) -> torch.Tensor:
    teacher_probs = torch.softmax(teacher_logits.detach(), dim=1)
    student_probs = torch.softmax(student_logits.detach(), dim=1)
    teacher_confidence, teacher_pred = teacher_probs.max(dim=1)
    student_confidence, student_pred = student_probs.max(dim=1)

    teacher_ready = ((teacher_confidence - float(teacher_confidence_floor)) / max(1e-6, 1.0 - float(teacher_confidence_floor))).clamp(0.0, 1.0)
    student_ready = ((student_confidence - float(student_confidence_floor)) / max(1e-6, 1.0 - float(student_confidence_floor))).clamp(0.0, 1.0)
    agreement_score = (student_pred == teacher_pred).float() * torch.minimum(student_ready, teacher_ready)

    if targets is None:
        return agreement_score

    targets = targets.detach()
    student_correct = (student_pred == targets).float() * student_ready
    teacher_correct = (teacher_pred == targets).float() * teacher_ready

    if strategy == "agreement":
        return agreement_score
    if strategy == "correctness":
        return student_correct
    if strategy == "teacher_correct":
        return teacher_correct
    if strategy == "hybrid":
        hybrid = torch.maximum(agreement_score, 0.65 * student_correct)
        hybrid = torch.where(teacher_correct > 0.0, hybrid, 0.5 * student_correct)
        return hybrid.clamp(0.0, 1.0)

    raise ValueError(f"Unsupported gate target strategy '{strategy}'")


def gate_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    clipped = probabilities.clamp_min(1e-8)
    return -(clipped * torch.log(clipped) + (1.0 - clipped) * torch.log((1.0 - clipped).clamp_min(1e-8))).mean()


def accept_rate_penalty(probabilities: torch.Tensor, *, target_accept_rate: float | None, weight: float) -> torch.Tensor:
    if target_accept_rate is None or weight <= 0.0:
        return torch.zeros((), device=probabilities.device)
    deviation = probabilities.mean() - float(target_accept_rate)
    return float(weight) * deviation.pow(2)


def initial_gate_bias(target_accept_rate: float | None, *, default: float = 0.5) -> float:
    probability = float(target_accept_rate if target_accept_rate is not None else default)
    clamped = min(0.95, max(0.05, probability))
    return math.log(clamped / (1.0 - clamped))


__all__ = [
    "accept_rate_penalty",
    "blend_gate_scores",
    "GateConfig",
    "budget_penalty",
    "distillation_loss",
    "weighted_distillation_loss",
    "gate_accept_targets",
    "gate_entropy",
    "gate_scores",
    "initial_gate_bias",
    "relaxed_accept_probability",
    "staged_accept_rate",
    "select_accept_mask",
]
