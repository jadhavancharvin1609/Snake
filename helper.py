"""Training progress display helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_scores(scores: list[int], mean_scores: list[float], output_path: str | Path) -> None:
    """Save a chart of per-game and average scores."""
    if not scores:
        return
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.title("Neural Snake training")
    plt.xlabel("Game")
    plt.ylabel("Score")
    plt.plot(scores, label="Game score", alpha=0.45)
    plt.plot(mean_scores, label="Average score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination)
    plt.close()
