import argparse
import json
from pathlib import Path


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_training_history(path):
    history = load_json(path)
    if not history:
        return {"epochs": 0}

    best = min(history, key=lambda row: row["val_loss"])
    final = history[-1]
    return {
        "epochs": len(history),
        "best_epoch": best["epoch"],
        "best_val_loss": best["val_loss"],
        "best_val_perplexity": best["val_perplexity"],
        "final_train_loss": final["train_loss"],
        "final_val_loss": final["val_loss"],
        "final_val_perplexity": final["val_perplexity"],
        "final_generalization_gap": final["val_loss"] - final["train_loss"],
    }


def build_summary(unconditioned_metrics, conditioned_metrics, baseline_jsons, output_json):
    summary = {"gru_models": {}, "baselines": {}}

    model_inputs = {
        "unconditioned": unconditioned_metrics,
        "conditioned": conditioned_metrics,
    }
    for label, path in model_inputs.items():
        path = Path(path)
        if path.exists():
            summary["gru_models"][label] = summarize_training_history(path)
        else:
            summary["gru_models"][label] = {"missing": str(path)}

    for path in baseline_jsons:
        path = Path(path)
        if not path.exists():
            summary["baselines"][path.stem] = {"missing": str(path)}
            continue
        row = load_json(path)
        summary["baselines"][path.stem] = {
            "n": row.get("n"),
            "train_perplexity": row.get("train_perplexity"),
            "validation_perplexity": row.get("validation_perplexity"),
            "sample_metrics": row.get("sample_metrics", {}),
        }

    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Summarize model training and baseline metrics.")
    parser.add_argument(
        "--unconditioned-metrics",
        default="outputs/checkpoints/unconditioned/metrics.json",
    )
    parser.add_argument(
        "--conditioned-metrics",
        default="outputs/checkpoints/conditioned/metrics.json",
    )
    parser.add_argument(
        "--baseline-json",
        action="append",
        default=[],
        help="Optional baseline metric JSON file. Can be repeated.",
    )
    parser.add_argument("--output-json", default="outputs/metrics/training_summary.json")
    args = parser.parse_args()

    summary = build_summary(
        args.unconditioned_metrics,
        args.conditioned_metrics,
        args.baseline_json,
        args.output_json,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
