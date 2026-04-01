from pathlib import Path

path = Path(r'D:\uni\asp\sem4\dynanets\src\dynanets\benchmark.py')
text = path.read_text(encoding='utf-8')
text = text.replace('PLOT_HATCHES = ["/", "\\\\", "x", "-", "+", ".", "o", "*", "|", "O"]\n', '')
text = text.replace('''    return {
        "color": PLOT_COLORS[index % len(PLOT_COLORS)],
        "marker": PLOT_MARKERS[index % len(PLOT_MARKERS)],
        "hatch": PLOT_HATCHES[index % len(PLOT_HATCHES)],
    }
''', '''    return {
        "color": PLOT_COLORS[index % len(PLOT_COLORS)],
        "marker": PLOT_MARKERS[index % len(PLOT_MARKERS)],
    }
''')
text = text.replace('''def _write_epoch_accuracy_plot(path: Path, aggregate: list[dict[str, Any]], styles: dict[str, dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    plotted = False
    for index, item in enumerate(aggregate):
        values = list(item.get("mean_metric_history") or [])
        if not values:
            continue
        epochs = list(range(1, len(values) + 1))
        style = styles[item["name"]]
        ax.plot(epochs, values, marker=style["marker"], linewidth=2.0, color=style["color"], label=item["name"])
        plotted = True
    if plotted:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean validation accuracy")
        ax.set_title("Mean Validation Accuracy by Epoch")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        ax.legend(loc="best", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No epoch history available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    fig.savefig(path, dpi=160)
    plt.close(fig)
''', '''def _write_epoch_accuracy_plot(path: Path, aggregate: list[dict[str, Any]], styles: dict[str, dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 7), constrained_layout=True)
    plotted = False
    for index, item in enumerate(aggregate):
        values = list(item.get("mean_metric_history") or [])
        if not values:
            continue
        epochs = list(range(1, len(values) + 1))
        style = styles[item["name"]]
        markevery = max(1, len(epochs) // 8)
        ax.plot(
            epochs,
            values,
            marker=style["marker"],
            markevery=markevery,
            linewidth=2.0,
            color=style["color"],
            label=item["name"],
        )
        plotted = True
    if plotted:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean validation accuracy")
        ax.set_title("Mean Validation Accuracy by Epoch")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False, borderaxespad=0.0)
    else:
        ax.text(0.5, 0.5, "No epoch history available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
''')
text = text.replace('''def _write_mean_accuracy_plot(path: Path, aggregate: list[dict[str, Any]], styles: dict[str, dict[str, str]]) -> None:
    names = [item["name"] for item in aggregate]
    means = [item["mean_final_val_accuracy"] for item in aggregate]
    stds = [item["std_final_val_accuracy"] for item in aggregate]
    colors = [styles[name]["color"] for name in names]
    hatches = [styles[name]["hatch"] for name in names]

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    positions = list(range(len(names)))
    bars = ax.bar(positions, means, yerr=stds, color=colors, capsize=5, edgecolor="#222222", linewidth=0.8)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Mean final validation accuracy")
    ax.set_ylim(0.0, min(1.0, max(means) + max(stds, default=0.0) + 0.08))
    ax.set_title("Benchmark Summary: Mean Final Validation Accuracy")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    fig.savefig(path, dpi=160)
    plt.close(fig)
''', '''def _write_mean_accuracy_plot(path: Path, aggregate: list[dict[str, Any]], styles: dict[str, dict[str, str]]) -> None:
    names = [item["name"] for item in aggregate]
    means = [item["mean_final_val_accuracy"] for item in aggregate]
    stds = [item["std_final_val_accuracy"] for item in aggregate]
    colors = [styles[name]["color"] for name in names]

    fig, ax = plt.subplots(figsize=(12.5, max(5.5, 0.42 * len(names) + 2.0)), constrained_layout=True)
    positions = list(range(len(names)))
    ax.barh(positions, means, xerr=stds, color=colors, capsize=5, edgecolor="#222222", linewidth=0.8)
    ax.set_yticks(positions)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Mean final validation accuracy")
    ax.set_xlim(0.0, min(1.0, max(means) + max(stds, default=0.0) + 0.08))
    ax.set_title("Benchmark Summary: Mean Final Validation Accuracy")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.5)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
''')
text = text.replace('''def _write_per_seed_plot(path: Path, aggregate: list[dict[str, Any]], styles: dict[str, dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for index, item in enumerate(aggregate):
        style = styles[item["name"]]
        seed_runs = item["seed_runs"]
        seeds = [run["seed"] for run in seed_runs]
        values = [run["final_val_accuracy"] for run in seed_runs]
        ax.plot(seeds, values, marker=style["marker"], linewidth=2.0, color=style["color"], label=item["name"])
    ax.set_xlabel("Seed")
    ax.set_ylabel("Final validation accuracy")
    ax.set_title("Per-Seed Final Validation Accuracy")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)
''', '''def _write_per_seed_plot(path: Path, aggregate: list[dict[str, Any]], styles: dict[str, dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 7), constrained_layout=True)
    for index, item in enumerate(aggregate):
        style = styles[item["name"]]
        seed_runs = item["seed_runs"]
        seeds = [run["seed"] for run in seed_runs]
        values = [run["final_val_accuracy"] for run in seed_runs]
        ax.plot(seeds, values, marker=style["marker"], linewidth=2.0, color=style["color"], label=item["name"])
    ax.set_xlabel("Seed")
    ax.set_ylabel("Final validation accuracy")
    ax.set_title("Per-Seed Final Validation Accuracy")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False, borderaxespad=0.0)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
''')
text = text.replace('''        names = [item["name"] for item in filtered]
        values = [float(item[metric_key]) for item in filtered]
        colors = [styles[name]["color"] for name in names]
        hatches = [styles[name]["hatch"] for name in names]
        positions = list(range(len(names)))
        bars = ax.bar(positions, values, color=colors, alpha=0.9, edgecolor="#222222", linewidth=0.8)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        ax.set_xticks(positions)
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    fig.savefig(path, dpi=160)
''', '''        names = [item["name"] for item in filtered]
        values = [float(item[metric_key]) for item in filtered]
        colors = [styles[name]["color"] for name in names]
        positions = list(range(len(names)))
        ax.barh(positions, values, color=colors, alpha=0.9, edgecolor="#222222", linewidth=0.8)
        ax.set_yticks(positions)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.5)
    fig.savefig(path, dpi=160, bbox_inches="tight")
''')
text = text.replace('''        for index, item in enumerate(filtered):
            style = styles[item["name"]]
            x_value = float(item[metric_key])
            y_value = float(item["mean_final_val_accuracy"])
            ax.scatter(x_value, y_value, color=style["color"], marker=style["marker"], s=80, edgecolors="#222222", linewidths=0.6)
            ax.annotate(item["name"], (x_value, y_value), textcoords="offset points", xytext=(5, 4), fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean final validation accuracy")
        ax.set_title(title)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    fig.savefig(path, dpi=160)
''', '''        for index, item in enumerate(filtered):
            style = styles[item["name"]]
            x_value = float(item[metric_key])
            y_value = float(item["mean_final_val_accuracy"])
            ax.scatter(
                x_value,
                y_value,
                color=style["color"],
                marker=style["marker"],
                s=80,
                edgecolors="#222222",
                linewidths=0.6,
                label=item["name"],
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean final validation accuracy")
        ax.set_title(title)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False, borderaxespad=0.0)
    fig.savefig(path, dpi=160, bbox_inches="tight")
''')
path.write_text(text, encoding='utf-8')
print('patched benchmark.py')
