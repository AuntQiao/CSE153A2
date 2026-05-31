import argparse
import json
from collections import Counter
from math import log2
from pathlib import Path


def load_tokens(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl_tokens(path):
    sequences = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sequences.append(json.loads(line)["tokens"])
    return sequences


def note_tokens(tokens):
    return [token for token in tokens if token.startswith("NOTE_")]


def rest_tokens(tokens):
    return [token for token in tokens if token.startswith("REST_")]


def chord_tokens(tokens):
    return [token for token in tokens if token.startswith("CHORD_")]


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


def pitch_classes(tokens):
    return [pitch % 12 for pitch in pitch_values(tokens)]


def entropy(values):
    if not values:
        return 0.0
    counts = Counter(values)
    total = sum(counts.values())
    value = -sum((count / total) * log2(count / total) for count in counts.values())
    return 0.0 if abs(value) < 1e-12 else value


def repetition_rate(tokens):
    if len(tokens) < 2:
        return 0.0
    repeats = sum(1 for left, right in zip(tokens, tokens[1:]) if left == right)
    return repeats / (len(tokens) - 1)


def interval_values(tokens):
    pitches = pitch_values(tokens)
    return [right - left for left, right in zip(pitches, pitches[1:])]


ROOT_TO_PC = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}


def parse_chord_root_and_quality(token):
    if not token.startswith("CHORD_") or token == "CHORD_NC":
        return None

    figure = token.removeprefix("CHORD_")
    if not figure or figure[0] not in ROOT_TO_PC:
        return None

    root = ROOT_TO_PC[figure[0]]
    idx = 1
    if idx < len(figure) and figure[idx] in {"#", "b", "-"}:
        root += 1 if figure[idx] == "#" else -1
        idx += 1
    root %= 12

    suffix = figure[idx:].lower()
    if suffix.startswith("maj"):
        quality = "major"
    elif suffix.startswith("m") or "min" in suffix:
        quality = "minor"
    elif "dim" in suffix or suffix.startswith("o"):
        quality = "diminished"
    elif "aug" in suffix or suffix.startswith("+"):
        quality = "augmented"
    elif "sus" in suffix:
        quality = "suspended"
    else:
        quality = "major"
    return root, quality, suffix


def chord_pitch_classes(token):
    parsed = parse_chord_root_and_quality(token)
    if parsed is None:
        return None
    root, quality, suffix = parsed
    if quality == "minor":
        pcs = {root, (root + 3) % 12, (root + 7) % 12}
    elif quality == "diminished":
        pcs = {root, (root + 3) % 12, (root + 6) % 12}
    elif quality == "augmented":
        pcs = {root, (root + 4) % 12, (root + 8) % 12}
    elif quality == "suspended":
        pcs = {root, (root + 5) % 12, (root + 7) % 12}
    else:
        pcs = {root, (root + 4) % 12, (root + 7) % 12}

    if "maj7" in suffix:
        pcs.add((root + 11) % 12)
    elif "7" in suffix:
        pcs.add((root + 10) % 12)
    return pcs


def chord_tone_rate(tokens):
    active_chord = None
    matching = 0
    checked = 0
    per_chord = Counter()
    per_chord_hits = Counter()

    for token in tokens:
        if token.startswith("CHORD_"):
            active_chord = token
            continue
        if not token.startswith("NOTE_") or active_chord is None:
            continue
        allowed = chord_pitch_classes(active_chord)
        if allowed is None:
            continue
        pitches = pitch_values([token])
        if not pitches:
            continue
        checked += 1
        per_chord[active_chord] += 1
        if pitches[0] % 12 in allowed:
            matching += 1
            per_chord_hits[active_chord] += 1

    return {
        "checked_notes": checked,
        "chord_tone_rate": matching / checked if checked else None,
        "per_chord": {
            chord: {
                "notes": per_chord[chord],
                "chord_tone_rate": per_chord_hits[chord] / per_chord[chord],
            }
            for chord in sorted(per_chord)
        },
    }


def summarize_sequences(sequences):
    lengths = [len(tokens) for tokens in sequences]
    notes_per_sequence = [len(note_tokens(tokens)) for tokens in sequences]
    all_tokens = [token for sequence in sequences for token in sequence]
    pitches = pitch_values(all_tokens)
    durations = duration_values(all_tokens)
    chords = chord_tokens(all_tokens)

    return {
        "num_sequences": len(sequences),
        "num_tokens": len(all_tokens),
        "vocab_size": len(set(all_tokens)),
        "sequence_length": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": sum(lengths) / len(lengths) if lengths else 0.0,
        },
        "notes_per_sequence": {
            "min": min(notes_per_sequence) if notes_per_sequence else 0,
            "max": max(notes_per_sequence) if notes_per_sequence else 0,
            "mean": sum(notes_per_sequence) / len(notes_per_sequence)
            if notes_per_sequence
            else 0.0,
        },
        "num_note_tokens": len(note_tokens(all_tokens)),
        "num_rest_tokens": len(rest_tokens(all_tokens)),
        "num_bar_tokens": all_tokens.count("<BAR>"),
        "num_chord_tokens": len(chords),
        "unique_chords": len(set(chords)),
        "unique_pitches": len(set(pitches)),
        "pitch_range": [min(pitches), max(pitches)] if pitches else None,
        "unique_durations": len(set(durations)),
        "pitch_class_entropy": entropy(pitch_classes(all_tokens)),
        "duration_entropy": entropy(durations),
        "top_tokens": Counter(all_tokens).most_common(15),
        "top_chords": Counter(chords).most_common(15),
    }


def summarize(tokens):
    pitches = pitch_values(tokens)
    durations = duration_values(tokens)
    intervals = interval_values(tokens)
    absolute_intervals = [abs(value) for value in intervals]
    chord_summary = chord_tone_rate(tokens)
    return {
        "num_tokens": len(tokens),
        "num_note_tokens": len(pitches),
        "num_rest_tokens": len(rest_tokens(tokens)),
        "num_bar_tokens": tokens.count("<BAR>"),
        "num_chord_tokens": len(chord_tokens(tokens)),
        "unique_note_tokens": len(set(note_tokens(tokens))),
        "unique_pitches": len(set(pitches)),
        "pitch_range": [min(pitches), max(pitches)] if pitches else None,
        "unique_durations": len(set(durations)),
        "pitch_class_entropy": entropy(pitch_classes(tokens)),
        "duration_entropy": entropy(durations),
        "repetition_rate": repetition_rate(tokens),
        "mean_abs_interval": sum(absolute_intervals) / len(absolute_intervals)
        if absolute_intervals
        else 0.0,
        "large_leap_rate": sum(1 for value in absolute_intervals if value >= 7)
        / len(absolute_intervals)
        if absolute_intervals
        else 0.0,
        "chord_tone_rate": chord_summary["chord_tone_rate"],
        "chord_tone_checked_notes": chord_summary["checked_notes"],
        "chord_tone_by_chord": chord_summary["per_chord"],
        "top_notes": Counter(note_tokens(tokens)).most_common(10),
        "top_chords": Counter(chord_tokens(tokens)).most_common(10),
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize generated token metrics.")
    parser.add_argument("--tokens-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--reference-jsonl",
        default=None,
        help="Optional tokenized dataset split to summarize beside the generated sample.",
    )
    args = parser.parse_args()

    metrics = {"generated": summarize(load_tokens(args.tokens_json))}
    if args.reference_jsonl:
        metrics["reference"] = summarize_sequences(load_jsonl_tokens(args.reference_jsonl))
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
