from __future__ import annotations

from typing import Any


def extract_architecture_artifacts(model: Any, *, name: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    spec_factory = getattr(model, "architecture_spec", None)
    if not callable(spec_factory):
        return None, None

    spec = spec_factory()
    spec_dict = spec.to_dict() if hasattr(spec, "to_dict") else None

    graph_dict = None
    if hasattr(spec, "to_graph"):
        graph = spec.to_graph(name=name)
        if hasattr(graph, "to_dict"):
            graph_dict = graph.to_dict()

    return spec_dict, graph_dict
