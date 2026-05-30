import argparse
import json
from collections import Counter
from pathlib import Path


def load_tokens(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def note_tokens(tokens):
    return [token for token in tokens if token.startswith("NOTE_")]


def duration_values(tokens):
    values = []
    for token in tokens:
        parts = token.split("_")
        if len(parts) == 3 and parts[0] == "NOTE":
            try:
                values.append(int(parts[2]))
            except ValueError:
                pass
        elif len(parts) == 2 and parts[0] == "REST":
            try:
                values.append(int(parts[1]))
            except ValueError:
                pass
    return values


def pitch_values(tokens):
    values = []
    for token in note_tokens(tokens):
        parts = token.split("_")
        try:
            values.append(int(parts[1]))
        except (IndexError, ValueError):
            pass
    return values


def repetition_rate(tokens):
    if len(tokens) < 2:
        return 0.0
    repeats = sum(1 for left, right in zip(tokens, tokens[1:]) if left == right)
    return repeats / (len(tokens) - 1)


def summarize(tokens):
    pitches = pitch_values(tokens)
    durations = duration_values(tokens)
    return {
        "num_tokens": len(tokens),
        "num_note_tokens": len(pitches),
        "unique_note_tokens": len(set(note_tokens(tokens))),
        "unique_pitches": len(set(pitches)),
        "pitch_range": [min(pitches), max(pitches)] if pitches else None,
        "unique_durations": len(set(durations)),
        "repetition_rate": repetition_rate(tokens),
        "top_notes": Counter(note_tokens(tokens)).most_common(10),
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize generated token metrics.")
    parser.add_argument("--tokens-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    metrics = summarize(load_tokens(args.tokens_json))
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
