"""Render the recorded validation trajectory; never reads image datasets."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/2026-11-04-mnist-direct-weights"


def main():
    history = json.loads((EXPERIMENT / "evidence/development-history.json").read_text())
    history = history["rows"]
    x = [int(row["candidate"][1:]) for row in history]
    y = [100 * row["accuracy"] for row in history]
    fig, ax = plt.subplots(figsize=(11, 5.6), layout="constrained")
    fig.patch.set_facecolor("#faf9f5")
    ax.set_facecolor("#faf9f5")
    ax.plot(x, y, color="#176b5b", linewidth=2.6, marker="o", markersize=4.2)
    ax.axhline(95, color="#777777", linewidth=1.0, linestyle="--")
    ax.text(2.0, 95.7, "95% target", color="#666666", fontsize=10)
    ax.scatter([26], [95.72], color="#176b5b", s=60, zorder=4)
    ax.annotate("95.72% validation", xy=(26, 95.72), xytext=(18, 91.0),
                fontsize=11, color="#176b5b", arrowprops={"arrowstyle": "-", "color": "#176b5b"})
    ax.set_title("Codex authored and revised a digit recognizer's parameters", loc="left", fontsize=16, pad=35)
    ax.text(0, 1.035, "26 candidates · no numerical fitting routine · 96.03% on the frozen 10,000-image holdout",
            transform=ax.transAxes, fontsize=11, color="#555555")
    ax.set(xlabel="Candidate", ylabel="Validation accuracy (%)", xlim=(1, 27), ylim=(72, 100))
    ax.set_xticks([2, 6, 10, 14, 18, 22, 26])
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.015, -0.015,
             "Recorded development results. Holdout: fresh split of MNIST's training partition; not the canonical benchmark test set.",
             fontsize=9, color="#666666")
    assets = EXPERIMENT / "assets"
    assets.mkdir(exist_ok=True)
    fig.savefig(assets / "progress.png", dpi=180, bbox_inches="tight")
    fig.savefig(assets / "progress.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
