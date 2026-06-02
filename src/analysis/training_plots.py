import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_training_curves(unconditioned_metrics, conditioned_metrics, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    histories = {
        "unconditioned": load_json(unconditioned_metrics),
        "conditioned": load_json(conditioned_metrics),
    }

    plt.figure(figsize=(8, 4.8))
    for label, history in histories.items():
        epochs = [row["epoch"] for row in history]
        train_loss = [row["train_loss"] for row in history]
        val_loss = [row["val_loss"] for row in history]
        plt.plot(epochs, train_loss, marker="o", linewidth=2, label=f"{label} train")
        plt.plot(epochs, val_loss, marker="s", linewidth=2, linestyle="--", label=f"{label} val")

    plt.title("GRU Training And Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return str(output_path)


def save_perplexity_comparison(training_summary, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = load_json(training_summary)

    labels = ["Task 1\nunconditioned", "Task 2\nconditioned"]
    ngram = [
        summary["baselines"]["ngram_unconditioned"]["validation_perplexity"],
        summary["baselines"]["ngram_conditioned"]["validation_perplexity"],
    ]
    gru = [
        summary["gru_models"]["unconditioned"]["best_val_perplexity"],
        summary["gru_models"]["conditioned"]["best_val_perplexity"],
    ]
    x_positions = range(len(labels))
    width = 0.34

    plt.figure(figsize=(7, 4.8))
    plt.bar([x - width / 2 for x in x_positions], ngram, width, label="trigram baseline", color="#8d99ae")
    plt.bar([x + width / 2 for x in x_positions], gru, width, label="GRU", color="#2a9d8f")
    plt.xticks(list(x_positions), labels)
    plt.ylabel("Validation perplexity")
    plt.title("GRU Versus N-Gram Baseline")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Create training and baseline comparison plots.")
    parser.add_argument(
        "--unconditioned-metrics",
        default="outputs/checkpoints/unconditioned/metrics.json",
    )
    parser.add_argument(
        "--conditioned-metrics",
        default="outputs/checkpoints/conditioned/metrics.json",
    )
    parser.add_argument("--training-summary", default="outputs/metrics/training_summary.json")
    parser.add_argument("--figures-dir", default="figures")
    args = parser.parse_args()

    figures_dir = Path(args.figures_dir)
    outputs = {
        "training_curves": save_training_curves(
            args.unconditioned_metrics,
            args.conditioned_metrics,
            figures_dir / "training_curves.png",
        ),
        "perplexity_comparison": save_perplexity_comparison(
            args.training_summary,
            figures_dir / "perplexity_comparison.png",
        ),
    }
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
