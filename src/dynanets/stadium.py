from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dynanets.benchmark import _pareto_frontier
from dynanets.protocol import BenchmarkProtocol


BASELINE_METHOD_TYPES = {"baseline"}
REQUIRED_RUNTIME_KEYS = {"requested_device", "resolved_device", "torch_version", "cuda_available"}


def evaluate_protocol_acceptance(
    protocol: BenchmarkProtocol,
    runs: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
    *,
    manifest_dir: Path | None = None,
) -> dict[str, Any]:
    unique_seeds = sorted(
        {
            int(run.get("benchmark_seed", run.get("seed", -1)))
            for run in runs
            if run.get("benchmark_seed", run.get("seed")) is not None
        }
    )
    baseline_count = sum(1 for item in aggregate if item.get("method_type") in BASELINE_METHOD_TYPES)
    method_count = sum(1 for item in aggregate if item.get("method_type") not in BASELINE_METHOD_TYPES)
    role_counts = _role_counts(protocol)
    experiment_names = _resolved_experiment_names(protocol, manifest_dir)

    checks: list[dict[str, Any]] = []
    acceptance = protocol.acceptance
    if acceptance.min_seeds is not None:
        checks.append(
            _check(
                "min_seeds",
                len(unique_seeds) >= acceptance.min_seeds,
                f"required>={acceptance.min_seeds}, observed={len(unique_seeds)}",
            )
        )
    if acceptance.required_roles:
        missing_roles = [role for role in acceptance.required_roles if role_counts.get(role, 0) == 0]
        checks.append(
            _check(
                "required_roles",
                not missing_roles,
                f"required={acceptance.required_roles}, missing={missing_roles or 'none'}",
            )
        )
    if acceptance.required_experiments:
        required_names = {_normalize_experiment_name(item) for item in acceptance.required_experiments}
        missing = sorted(required_names - experiment_names)
        checks.append(
            _check(
                "required_experiments",
                not missing,
                f"required={sorted(required_names)}, missing={missing or 'none'}",
            )
        )
    if acceptance.minimum_methods is not None:
        checks.append(
            _check(
                "minimum_methods",
                method_count >= acceptance.minimum_methods,
                f"required>={acceptance.minimum_methods}, observed={method_count}",
            )
        )
    if acceptance.minimum_baselines is not None:
        checks.append(
            _check(
                "minimum_baselines",
                baseline_count >= acceptance.minimum_baselines,
                f"required>={acceptance.minimum_baselines}, observed={baseline_count}",
            )
        )
    if acceptance.require_constraints:
        missing_constraints = [
            item["name"]
            for item in aggregate
            if item.get("mean_parameter_count") is None or item.get("mean_forward_flop_proxy") is None
        ]
        checks.append(_check("require_constraints", not missing_constraints, f"missing={missing_constraints or 'none'}"))
    if acceptance.require_runtime_environment:
        missing_runtime = [
            item["name"]
            for item in aggregate
            if not REQUIRED_RUNTIME_KEYS.issubset(set((item.get("runtime_environment") or {}).keys()))
        ]
        checks.append(_check("require_runtime_environment", not missing_runtime, f"missing={missing_runtime or 'none'}"))
    if acceptance.require_stage_history:
        missing_stage_history = [
            item["name"]
            for item in aggregate
            if item.get("method_type") == "workflow" and not item.get("representative_stage_history")
        ]
        checks.append(_check("require_stage_history", not missing_stage_history, f"missing={missing_stage_history or 'none'}"))

    missing_metrics = _missing_required_metrics(protocol.required_metrics, runs)
    checks.append(_check("required_metrics", not missing_metrics, f"missing={missing_metrics or 'none'}"))

    passed = all(check["passed"] for check in checks)
    return {
        "protocol_name": protocol.name,
        "track": protocol.track,
        "tier": protocol.tier,
        "passed": passed,
        "summary": {
            "num_runs": len(runs),
            "num_experiments": len(aggregate),
            "num_seeds": len(unique_seeds),
            "baseline_count": baseline_count,
            "method_count": method_count,
            "role_counts": role_counts,
        },
        "checks": checks,
    }


def build_protocol_leaderboard(protocol: BenchmarkProtocol, aggregate: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        aggregate,
        key=lambda item: (
            item.get("mean_final_val_accuracy", float("-inf")),
            item.get("mean_best_val_accuracy", float("-inf")),
        ),
        reverse=True,
    )
    baselines = [item for item in ordered if item.get("method_type") in BASELINE_METHOD_TYPES]
    methods = [item for item in ordered if item.get("method_type") not in BASELINE_METHOD_TYPES]
    pareto = _pareto_frontier(aggregate, x_key="mean_forward_flop_proxy", y_key="mean_final_val_accuracy")
    return {
        "protocol_name": protocol.name,
        "track": protocol.track,
        "tier": protocol.tier,
        "top_overall": _leaderboard_row(ordered[0]) if ordered else None,
        "top_method": _leaderboard_row(methods[0]) if methods else None,
        "top_baseline": _leaderboard_row(baselines[0]) if baselines else None,
        "accuracy_ranking": [_leaderboard_row(item) for item in ordered],
        "accuracy_flop_pareto_frontier": [_leaderboard_row(item) for item in pareto],
    }


def write_protocol_acceptance_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def write_protocol_acceptance_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Protocol Acceptance",
        "",
        f"Protocol: `{report['protocol_name']}`",
        f"Track: `{report['track']}`",
        f"Tier: `{report['tier']}`",
        f"Status: {'PASSED' if report['passed'] else 'FAILED'}",
        "",
        "## Summary",
        "",
        f"- runs: {report['summary']['num_runs']}",
        f"- experiments: {report['summary']['num_experiments']}",
        f"- seeds: {report['summary']['num_seeds']}",
        f"- baselines: {report['summary']['baseline_count']}",
        f"- methods: {report['summary']['method_count']}",
        "",
        "## Checks",
        "",
    ]
    for check in report["checks"]:
        lines.append(f"- [{'PASS' if check['passed'] else 'FAIL'}] `{check['name']}`: {check['detail']}")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_protocol_leaderboard_json(path: Path, leaderboard: dict[str, Any]) -> None:
    path.write_text(json.dumps(leaderboard, indent=2), encoding="utf-8")


def write_protocol_leaderboard_markdown(path: Path, leaderboard: dict[str, Any]) -> None:
    lines = [
        "# Protocol Leaderboard",
        "",
        f"Protocol: `{leaderboard['protocol_name']}`",
        f"Track: `{leaderboard['track']}`",
        f"Tier: `{leaderboard['tier']}`",
        "",
    ]
    if leaderboard.get("top_overall"):
        item = leaderboard["top_overall"]
        lines.extend([
            "## Top Overall",
            "",
            f"- `{item['name']}`: acc={item['mean_final_val_accuracy']:.4f}, flop_proxy={_fmt(item.get('mean_forward_flop_proxy'))}, params={_fmt(item.get('mean_parameter_count'))}",
            "",
        ])
    if leaderboard.get("top_method"):
        item = leaderboard["top_method"]
        lines.extend([
            "## Top Method",
            "",
            f"- `{item['name']}`: acc={item['mean_final_val_accuracy']:.4f}, flop_proxy={_fmt(item.get('mean_forward_flop_proxy'))}, params={_fmt(item.get('mean_parameter_count'))}",
            "",
        ])
    if leaderboard.get("top_baseline"):
        item = leaderboard["top_baseline"]
        lines.extend([
            "## Top Baseline",
            "",
            f"- `{item['name']}`: acc={item['mean_final_val_accuracy']:.4f}, flop_proxy={_fmt(item.get('mean_forward_flop_proxy'))}, params={_fmt(item.get('mean_parameter_count'))}",
            "",
        ])
    lines.extend([
        "## Accuracy Ranking",
        "",
        "| Rank | Method | Type | Mean final val acc | FLOP proxy | Params |",
        "| ---: | --- | --- | ---: | ---: | ---: |",
    ])
    for rank, item in enumerate(leaderboard.get("accuracy_ranking", []), start=1):
        lines.append(
            f"| {rank} | {item['name']} | {item['method_type']} | {item['mean_final_val_accuracy']:.4f} | {_fmt(item.get('mean_forward_flop_proxy'))} | {_fmt(item.get('mean_parameter_count'))} |"
        )
    pareto = leaderboard.get("accuracy_flop_pareto_frontier", [])
    if pareto:
        lines.extend(["", "## Accuracy-FLOP Pareto Frontier", ""])
        for item in pareto:
            lines.append(
                f"- `{item['name']}`: acc={item['mean_final_val_accuracy']:.4f}, flop_proxy={_fmt(item.get('mean_forward_flop_proxy'))}, params={_fmt(item.get('mean_parameter_count'))}"
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_protocol_leaderboard_latex(path: Path, leaderboard: dict[str, Any]) -> None:
    lines = [
        "% Auto-generated by dynanets.stadium.write_protocol_leaderboard_latex",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{" + _latex_escape(f"{leaderboard['protocol_name']} leaderboard ({leaderboard['track']}, {leaderboard['tier']})") + "}",
        "\\label{" + _latex_label(f"tab:{leaderboard['protocol_name']}_leaderboard") + "}",
        "\\begin{tabular}{rllrrr}",
        "\\toprule",
        r"Rank & Method & Type & Acc. & FLOPs & Params \\",
        "\\midrule",
    ]
    for rank, item in enumerate(leaderboard.get("accuracy_ranking", []), start=1):
        row = " & ".join(
            [
                str(rank),
                _latex_escape(item["name"]),
                _latex_escape(item["method_type"]),
                f"{item['mean_final_val_accuracy']:.4f}",
                _fmt(item.get("mean_forward_flop_proxy")),
                _fmt(item.get("mean_parameter_count")),
            ]
        ) + r" \\\\"
        lines.append(row)
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")

def _role_counts(protocol: BenchmarkProtocol) -> dict[str, int]:
    counts: dict[str, int] = {}
    for experiment in protocol.experiments:
        counts[experiment.role] = counts.get(experiment.role, 0) + 1
    return counts


def _resolved_experiment_names(protocol: BenchmarkProtocol, manifest_dir: Path | None) -> set[str]:
    experiments = protocol.resolve_configs(manifest_dir or Path(".")) if manifest_dir is not None else list(protocol.experiments)
    names = set()
    for item in experiments:
        names.add(_normalize_experiment_name(item.config))
    return names


def _normalize_experiment_name(value: str) -> str:
    path = Path(value)
    if path.suffix:
        return path.stem
    return value.strip()


def _missing_required_metrics(required_metrics: list[str], runs: list[dict[str, Any]]) -> list[str]:
    missing = []
    for metric in required_metrics:
        metric_name = metric.strip()
        if metric_name == "accuracy":
            if any(run.get("final_val_accuracy") is None for run in runs):
                missing.append(metric_name)
            continue
        if any(not _run_has_metric(run, metric_name) for run in runs):
            missing.append(metric_name)
    return missing


def _run_has_metric(run: dict[str, Any], metric_name: str) -> bool:
    for point in run.get("metric_history") or []:
        if metric_name in point and point[metric_name] is not None:
            return True
    return False


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _leaderboard_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": item["name"],
        "method_type": item.get("method_type", "unknown"),
        "mean_final_val_accuracy": float(item.get("mean_final_val_accuracy", 0.0)),
        "mean_best_val_accuracy": float(item.get("mean_best_val_accuracy", 0.0)),
        "mean_forward_flop_proxy": item.get("mean_forward_flop_proxy"),
        "mean_parameter_count": item.get("mean_parameter_count"),
        "best_seed": item.get("best_seed"),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    return str(int(round(float(value))))


def _latex_escape(value: str) -> str:
    replacements = {
        '\\': r'\\textbackslash{}',
        '&': r'\\&',
        '%': r'\\%',
        '$': r'\\$',
        '#': r'\\#',
        '_': r'\\_',
        '{': r'\\{',
        '}': r'\\}',
        '~': r'\\textasciitilde{}',
        '^': r'\\textasciicircum{}',
    }
    return ''.join(replacements.get(ch, ch) for ch in value)


def _latex_label(value: str) -> str:
    clean = []
    for ch in value.lower():
        if ch.isalnum() or ch in {':', '_'}:
            clean.append(ch)
        else:
            clean.append('_')
    return ''.join(clean)


__all__ = [
    "build_protocol_leaderboard",
    "evaluate_protocol_acceptance",
    "write_protocol_acceptance_json",
    "write_protocol_acceptance_markdown",
    "write_protocol_leaderboard_json",
    "write_protocol_leaderboard_markdown",
    "write_protocol_leaderboard_latex",
]

