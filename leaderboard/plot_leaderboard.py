"""Function to plot the leaderboard."""
import re
import pandas as pd
import matplotlib.pyplot as plt

LEADERBOARD_PATH = "leaderboard/leaderboard.csv"
TIMESTAMP_RE = re.compile(r"_\d{8}_\d{6}.*$")


def extract_model_name(run_name: str) -> str:
    """Extract model name from num name.

    Returns:
      model name stripped out of the timestamp regex, or just model name if timestamp
        is not present
    """
    return TIMESTAMP_RE.sub("", run_name) or run_name


def plot_leaderboard(csv_path: str = LEADERBOARD_PATH) -> None:
    """Plot the leaderboard results.

    Args:
      csv_path: String for the leaderboard path, defaults to "leaderboard/leaderboard.csv"
    """
    df = pd.read_csv(csv_path)
    df["model_name"] = df["run_name"].apply(extract_model_name)
    df["energy_model"] = df["energy_model"].astype(str).str.strip() == "True"

    groups = {True: "Energy Model", False: "Potential Damage Model"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (energy_flag, label) in zip(axes, groups.items()):
        subset = df[df["energy_model"] == energy_flag]
        if subset.empty:
            ax.set_title(f"{label} (no data)")
            ax.axis("off")
            continue

        best = (
            subset.groupby("model_name")["best_metric_value"]
            .max()
            .sort_values(ascending=False)
        )

        bars = ax.bar(best.index, best.values, color="steelblue", edgecolor="white")
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_xlabel("Model")
        ax.set_ylabel("Best F1 Macro")
        ax.set_ylim(0, min(best.values.max() * 1.2, 1.0))
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Leaderboard — Best F1 Macro by Model Type", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("leaderboard/leaderboard_plot.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_leaderboard()
