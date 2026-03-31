from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
]


def summarize_run(
    name: str,
    summary: Any,
    final_hidden_dim: int | None = None,
    metadata: dict[str, Any] | None = None,
    architecture_spec: dict[str, Any] | None = None,
    architecture_graph: dict[str, Any] | None = None,
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    train_history = summary.train_history
    metric_history = summary.metric_history
    adaptation_history = summary.adaptation_history
    stage_history = getattr(summary, "stage_history", [])
    workflow_metadata = getattr(summary, "workflow_metadata", {})

    final_train = train_history[-1] if train_history else {}
    final_metrics = metric_history[-1] if metric_history else {}
    best_accuracy = max((item.get("accuracy", float("nan")) for item in metric_history), default=float("nan"))

    return {
        "name": name,
        "epochs": len(train_history),
        "final_train_loss": final_train.get("loss"),
        "final_train_accuracy": final_train.get("accuracy"),
        "final_val_accuracy": final_metrics.get("accuracy"),
        "best_val_accuracy": best_accuracy,
        "adaptations_applied": sum(1 for item in adaptation_history if item.get("applied")),
        "final_hidden_dim": final_hidden_dim,
        "train_history": train_history,
        "metric_history": metric_history,
        "adaptation_history": adaptation_history,
        "stage_history": stage_history,
        "workflow_metadata": workflow_metadata,
        "architecture_spec": architecture_spec,
        "architecture_graph": architecture_graph,
        "constraints": constraints or {},
        "metadata": metadata or {},
    }


def write_json(path: Path, data: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_csv(path: Path, data: list[dict[str, Any]]) -> None:
    rows = [
        {
            "name": item["name"],
            "epochs": item["epochs"],
            "final_train_loss": item["final_train_loss"],
            "final_train_accuracy": item["final_train_accuracy"],
            "final_val_accuracy": item["final_val_accuracy"],
            "best_val_accuracy": item["best_val_accuracy"],
            "adaptations_applied": item["adaptations_applied"],
            "final_hidden_dim": item["final_hidden_dim"],
            "parameter_count": item["constraints"].get("parameter_count"),
            "nonzero_parameter_count": item["constraints"].get("nonzero_parameter_count"),
            "weight_sparsity": item["constraints"].get("weight_sparsity"),
            "forward_flop_proxy": item["constraints"].get("forward_flop_proxy"),
            "activation_elements": item["constraints"].get("activation_elements"),
            "method_type": item["metadata"].get("method_type", "train"),
            "notes": item["metadata"].get("notes", ""),
        }
        for item in data
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, data: list[dict[str, Any]]) -> None:
    lines = [
        "# Baseline Comparison",
        "",
        "| Experiment | Type | Epochs | Final train acc | Final val acc | Best val acc | Adaptations | Final hidden dim |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in data:
        lines.append(
            "| {name} | {method_type} | {epochs} | {final_train_accuracy:.4f} | {final_val_accuracy:.4f} | {best_val_accuracy:.4f} | {adaptations_applied} | {final_hidden_dim} |".format(
                name=item["name"],
                method_type=item["metadata"].get("method_type", "train"),
                epochs=item["epochs"],
                final_train_accuracy=item["final_train_accuracy"],
                final_val_accuracy=item["final_val_accuracy"],
                best_val_accuracy=item["best_val_accuracy"],
                adaptations_applied=item["adaptations_applied"],
                final_hidden_dim=item["final_hidden_dim"] if item["final_hidden_dim"] is not None else "-",
            )
        )

    lines.extend(
        [
            "",
            "## Validation Accuracy",
            "",
            "![Validation accuracy](validation_accuracy.png)",
            "",
            "## Training Accuracy",
            "",
            "![Training accuracy](training_accuracy.png)",
            "",
            "## Training Loss",
            "",
            "![Training loss](training_loss.png)",
            "",
            "## Experiment Notes",
            "",
        ]
    )
    for item in data:
        notes = item["metadata"].get("notes")
        if notes:
            lines.append(f"- `{item['name']}`: {notes}")

    route_items = [item for item in data if item["metadata"].get("route_summary") or item["metadata"].get("route_trace")]
    if route_items:
        lines.extend(["", "## Routing Details", ""])
        for item in route_items:
            lines.append(f"### {item['name']}")
            route_summary = item["metadata"].get("route_summary")
            if route_summary:
                lines.append(f"- route_summary={route_summary}")
            route_trace = item["metadata"].get("route_trace")
            if route_trace:
                lines.append(f"- route_trace={route_trace}")
            lines.append("")

    constraint_items = [item for item in data if item.get("constraints")]
    if constraint_items:
        lines.extend(
            [
                "",
                "## Constraint Summary",
                "",
                "| Experiment | Params | Nonzero params | Weight sparsity | FLOP proxy | Activation elems |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in constraint_items:
            constraints = item["constraints"]
            lines.append(
                "| {name} | {parameter_count} | {nonzero_parameter_count} | {weight_sparsity:.4f} | {forward_flop_proxy} | {activation_elements} |".format(
                    name=item["name"],
                    parameter_count=_format_int(constraints.get("parameter_count")),
                    nonzero_parameter_count=_format_int(constraints.get("nonzero_parameter_count")),
                    weight_sparsity=float(constraints.get("weight_sparsity", 0.0)),
                    forward_flop_proxy=_format_int(constraints.get("forward_flop_proxy")),
                    activation_elements=_format_int(constraints.get("activation_elements")),
                )
            )

    stage_items = [item for item in data if item.get("stage_history")]
    if stage_items:
        lines.extend(["", "## Workflow Stages", ""])
        for item in stage_items:
            lines.append(f"### {item['name']}")
            for stage in item["stage_history"]:
                lines.append(
                    f"- {stage['name']}: epochs={stage['epochs']}, range={stage['epoch_start']}..{stage['epoch_end']}, adaptation_enabled={stage['adaptation_enabled']}, final_val={stage['final_val_accuracy']}"
                )
            if item.get("workflow_metadata"):
                lines.append(f"- workflow_metadata={item['workflow_metadata']}")
            lines.append("")

    lines.extend(["", "## Adaptation Timeline", ""])
    for item in data:
        events = [event for event in item["adaptation_history"] if event.get("applied")]
        if not events:
            continue
        lines.append(f"### {item['name']}")
        for event in events:
            before_metadata = event.get("before_state", {}).get("metadata", {})
            after_metadata = event.get("after_state", {}).get("metadata", {})
            capabilities = event.get("model_capabilities", {}).get("supported_event_types", [])
            effect = event.get("effect_summary", {})
            lines.append(
                f"- epoch {event['epoch'] + 1}: `{event['event_type']}` params={event.get('params', {})} effect={effect} before={before_metadata} after={after_metadata} capabilities={capabilities}"
            )
        lines.append("")

    graph_items = [item for item in data if item.get("architecture_graph") or item.get("architecture_spec")]
    if graph_items:
        lines.extend(["## Architecture Graphs", ""])
        for item in graph_items:
            lines.append(f"### {item['name']}")
            lines.append(mermaid_architecture(item.get("architecture_graph"), item.get("architecture_spec"), item["name"]))
            lines.append("")

    lines.extend(["## Validation Accuracy By Epoch", "", _mermaid_xychart(data), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_plots(output_dir: Path, data: list[dict[str, Any]]) -> None:
    _write_line_plot(
        output_dir / "validation_accuracy.png",
        title="Validation Accuracy",
        data=data,
        series_getter=lambda item: [point["accuracy"] for point in item["metric_history"]],
        y_label="Accuracy",
        adaptation_markers=True,
        y_lim=(0.0, 1.0),
    )
    _write_line_plot(
        output_dir / "training_accuracy.png",
        title="Training Accuracy",
        data=data,
        series_getter=lambda item: [point["accuracy"] for point in item["train_history"]],
        y_label="Accuracy",
        y_lim=(0.0, 1.0),
    )
    max_loss = max(max(point["loss"] for point in item["train_history"]) for item in data)
    _write_line_plot(
        output_dir / "training_loss.png",
        title="Training Loss",
        data=data,
        series_getter=lambda item: [point["loss"] for point in item["train_history"]],
        y_label="Loss",
        y_lim=(0.0, max_loss * 1.1),
    )


def _write_line_plot(
    path: Path,
    title: str,
    data: list[dict[str, Any]],
    series_getter: Callable[[dict[str, Any]], list[float]],
    y_label: str,
    y_lim: tuple[float, float] | None = None,
    adaptation_markers: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)

    for index, item in enumerate(data):
        color = COLORS[index % len(COLORS)]
        values = series_getter(item)
        epochs = list(range(1, len(values) + 1))
        ax.plot(epochs, values, marker="o", linewidth=2.2, color=color, label=item["name"])

        if adaptation_markers:
            event_epochs = [event["epoch"] + 1 for event in item["adaptation_history"] if event.get("applied")]
            event_values = [values[epoch - 1] for epoch in event_epochs if 0 < epoch <= len(values)]
            if event_epochs:
                ax.scatter(
                    event_epochs,
                    event_values,
                    s=110,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=1.8,
                )

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    max_epochs = max(len(series_getter(item)) for item in data)
    ax.set_xticks(list(range(1, max_epochs + 1)))
    ax.legend(loc="best", fontsize=9)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _mermaid_xychart(data: list[dict[str, Any]]) -> str:
    max_epochs = max(len(item["metric_history"]) for item in data)
    max_accuracy = max(item["best_val_accuracy"] for item in data)
    epoch_labels = ", ".join(str(index) for index in range(1, max_epochs + 1))
    chart = [
        "```mermaid",
        "xychart-beta",
        '    title "Validation Accuracy Comparison"',
        f'    x-axis "Epoch" [{epoch_labels}]',
        f'    y-axis "Accuracy" 0 --> {max(1.0, round(max_accuracy + 0.05, 2))}',
    ]
    for item in data:
        values = ", ".join(f"{point['accuracy']:.4f}" for point in item["metric_history"])
        chart.append(f'    line "{item["name"]}" [{values}]')
    chart.append("```")
    return "\n".join(chart)


def mermaid_architecture(
    architecture_graph: dict[str, Any] | None,
    architecture_spec: dict[str, Any] | None,
    name: str,
) -> str:
    if architecture_graph is not None:
        nodes = architecture_graph.get("nodes", [])
        edges = architecture_graph.get("edges", [])
        lines = ["```mermaid", "flowchart LR"]
        safe_name = name.replace('"', "'")
        lines.append(f'    title["{safe_name}"]')
        for node in nodes:
            node_id = str(node["id"]).replace("-", "_")
            label = str(node.get("label", node["id"])).replace('"', "'")
            lines.append(f'    {node_id}["{label}"]')
        for edge in edges:
            source = str(edge["source"]).replace("-", "_")
            target = str(edge["target"]).replace("-", "_")
            lines.append(f'    {source} --> {target}')
        lines.append("```")
        return "\n".join(lines)

    hidden_dims = list((architecture_spec or {}).get("hidden_dims", []))
    nodes = [("input", f"Input ({(architecture_spec or {}).get('input_dim')})")]
    for index, width in enumerate(hidden_dims, start=1):
        nodes.append((f"hidden{index}", f"Hidden {index} ({width})"))
    nodes.append(("output", f"Output ({(architecture_spec or {}).get('output_dim')})"))

    lines = ["```mermaid", "flowchart LR"]
    safe_name = name.replace('"', "'")
    lines.append(f'    title["{safe_name}"]')
    for node_id, label in nodes:
        lines.append(f'    {node_id}["{label}"]')
    for left, right in zip(nodes, nodes[1:]):
        lines.append(f'    {left[0]} --> {right[0]}')
    lines.append("```")
    return "\n".join(lines)


def _format_int(value: Any) -> str:
    if value is None:
        return "-"
    return str(int(value))

