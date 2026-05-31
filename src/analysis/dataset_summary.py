import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from src.evaluation.metrics import (
    chord_tokens,
    duration_values,
    note_tokens,
    pitch_values,
    summarize_sequences,
)


DEFAULT_SPLITS = {
    "train_unconditioned": "train_unconditioned.jsonl",
    "val_unconditioned": "val_unconditioned.jsonl",
    "train_conditioned": "train_conditioned.jsonl",
    "val_conditioned": "val_conditioned.jsonl",
}


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)


def split_rows(processed_dir):
    processed_dir = Path(processed_dir)
    rows_by_split = {}
    for split_name, filename in DEFAULT_SPLITS.items():
        path = processed_dir / filename
        if path.exists():
            rows_by_split[split_name] = read_jsonl(path)
    if not rows_by_split:
        expected = ", ".join(DEFAULT_SPLITS.values())
        raise FileNotFoundError(f"No processed JSONL splits found in {processed_dir}; expected {expected}")
    return rows_by_split


def all_tokens(rows):
    return [token for row in rows for token in row["tokens"]]


def sequence_lengths(rows):
    return [len(row["tokens"]) for row in rows]


def save_bar_chart(counter, path, title, xlabel, ylabel, limit=20):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    items = counter.most_common(limit)
    if not items:
        return None

    labels = [item[0] for item in items]
    values = [item[1] for item in items]
    fig_width = max(7, min(14, len(labels) * 0.55))
    plt.figure(figsize=(fig_width, 4.5))
    plt.bar(range(len(labels)), values, color="#386641")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return str(path)


def save_histogram(values, path, title, xlabel, ylabel, bins=30):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        return None

    plt.figure(figsize=(7, 4.5))
    plt.hist(values, bins=bins, color="#3a86ff", edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return str(path)


def build_summary(processed_dir, output_json, figures_dir):
    rows_by_split = split_rows(processed_dir)
    figures_dir = Path(figures_dir)

    summary = {"splits": {}, "figures": {}}
    for split_name, rows in rows_by_split.items():
        sequences = [row["tokens"] for row in rows]
        summary["splits"][split_name] = summarize_sequences(sequences)

    unconditioned_rows = rows_by_split.get("train_unconditioned", []) + rows_by_split.get(
        "val_unconditioned", []
    )
    conditioned_rows = rows_by_split.get("train_conditioned", []) + rows_by_split.get(
        "val_conditioned", []
    )

    if unconditioned_rows:
        tokens = all_tokens(unconditioned_rows)
        summary["figures"]["sequence_lengths"] = save_histogram(
            sequence_lengths(unconditioned_rows),
            figures_dir / "sequence_lengths.png",
            "Tokenized Tune Lengths",
            "Tokens per tune",
            "Tunes",
            bins=25,
        )
        summary["figures"]["pitch_distribution"] = save_histogram(
            pitch_values(tokens),
            figures_dir / "pitch_distribution.png",
            "Melody Pitch Distribution",
            "MIDI pitch",
            "Notes",
            bins=range(35, 91, 2),
        )
        summary["figures"]["duration_distribution"] = save_bar_chart(
            Counter(duration_values(tokens)),
            figures_dir / "duration_distribution.png",
            "Duration Token Distribution",
            "Sixteenth-note units",
            "Events",
            limit=16,
        )
        summary["figures"]["top_note_tokens"] = save_bar_chart(
            Counter(note_tokens(tokens)),
            figures_dir / "top_note_tokens.png",
            "Most Common Note Tokens",
            "Token",
            "Count",
            limit=20,
        )

    if conditioned_rows:
        tokens = all_tokens(conditioned_rows)
        summary["figures"]["top_chords"] = save_bar_chart(
            Counter(chord_tokens(tokens)),
            figures_dir / "top_chords.png",
            "Most Common Conditioning Chords",
            "Chord token",
            "Count",
            limit=20,
        )

    write_json(output_json, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Create dataset statistics and plots.")
    parser.add_argument("--processed-dir", default="data/processed/nottingham")
    parser.add_argument("--output-json", default="outputs/metrics/dataset_summary.json")
    parser.add_argument("--figures-dir", default="figures")
    args = parser.parse_args()

    summary = build_summary(args.processed_dir, args.output_json, args.figures_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
