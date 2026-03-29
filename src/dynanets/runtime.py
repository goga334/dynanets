from __future__ import annotations

import inspect
import random
from typing import Any

import numpy as np
import torch


def set_global_seed(seed: int | None) -> None:
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str | None = None) -> torch.device:
    choice = (requested or "auto").strip().lower()
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def runtime_environment(
    requested: str | None = None,
    *,
    resolved: str | None = None,
) -> dict[str, Any]:
    requested_choice = (requested or "auto").strip().lower()
    resolved_choice = resolved or str(resolve_device(requested_choice))
    cuda_available = torch.cuda.is_available()
    return {
        "requested_device": requested_choice,
        "resolved_device": resolved_choice,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
    }


def format_runtime_environment(environment: dict[str, Any]) -> str:
    parts = [
        f"device={environment['resolved_device']}",
        f"requested_device={environment['requested_device']}",
        f"torch={environment['torch_version']}",
        f"cuda_available={environment['cuda_available']}",
    ]
    if environment.get("torch_cuda_version") is not None:
        parts.append(f"torch_cuda={environment['torch_cuda_version']}")
    if environment.get("cuda_device_name"):
        parts.append(f"cuda_device={environment['cuda_device_name']}")
    return "; ".join(parts)


def prepare_factory_kwargs(
    factory: Any,
    params: dict[str, Any],
    *,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = dict(params)
    requested_device = (runtime or {}).get("device")

    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return resolved

    if "device" in signature.parameters and "device" not in resolved:
        resolved["device"] = str(resolve_device(requested_device))
    return resolved
