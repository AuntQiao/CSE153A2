import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from music21 import converter, harmony, note, stream
from tqdm import tqdm

from src.preprocessing.vocab import BAR_TOKEN, END_TOKEN, START_TOKEN, Vocabulary


@dataclass
class TuneTokens:
    source: str
    melody: list
    conditioned: list
    chords: list


def duration_units(duration, grid=4, max_units=32):
    units = int(round(float(duration.quarterLength) * grid))
    return max(1, min(max_units, units))


def pitch_token(element, grid=4):
    units = duration_units(element.duration, grid=grid)
    if isinstance(element, note.Rest):
        return f"REST_{units}"
    if isinstance(element, note.Note):
        return f"NOTE_{int(element.pitch.midi)}_{units}"
    return None


def chord_token(chord_symbol):
    if chord_symbol is None:
        return "CHORD_NC"
    root = chord_symbol.root()
    if root is None:
        return "CHORD_NC"
    figure = chord_symbol.figure.replace(" ", "_").replace("/", "_")
    if not figure:
        figure = root.name
    return f"CHORD_{figure}"


def first_melody_part(score):
    parts = score.parts
    if parts:
        return parts[0]
    return score


def measure_chord(measure, previous):
    symbols = list(measure.recurse().getElementsByClass(harmony.ChordSymbol))
    if symbols:
        return symbols[0]
    return previous


def tokenize_score(score, source, grid=4):
    part = first_melody_part(score)
    measures = list(part.getElementsByClass(stream.Measure))
    if not measures:
        measures = [part]

    melody = [START_TOKEN]
    conditioned = [START_TOKEN]
    chords = []
    previous_chord = None

    for measure in measures:
        current_chord = measure_chord(measure, previous_chord)
        previous_chord = current_chord
        chord = chord_token(current_chord)
        chords.append(chord)
        conditioned.append(chord)

        measure_tokens = []
        for element in measure.notesAndRests:
            token = pitch_token(element, grid=grid)
            if token:
                measure_tokens.append(token)

        if not measure_tokens:
            continue

        melody.extend(measure_tokens)
        conditioned.extend(measure_tokens)
        melody.append(BAR_TOKEN)
        conditioned.append(BAR_TOKEN)

    melody.append(END_TOKEN)
    conditioned.append(END_TOKEN)
    return TuneTokens(source=source, melody=melody, conditioned=conditioned, chords=chords)


def load_abc_files(raw_dir):
    raw_dir = Path(raw_dir)
    return sorted(raw_dir.rglob("*.abc")) + sorted(raw_dir.rglob("*.ABC"))


def split_abc_tunes(path):
    text = Path(path).read_text(encoding="latin-1")
    tunes = []
    current = []
    for line in text.splitlines():
        if line.startswith("X:") and current:
            tunes.append("\n".join(current).strip() + "\n")
            current = []
        if line.strip() or current:
            current.append(line)
    if current:
        tunes.append("\n".join(current).strip() + "\n")
    return [tune for tune in tunes if tune.lstrip().startswith("X:")]


def parse_tunes(raw_dir, grid=4, max_tunes=None):
    tunes = []
    failed = []
    records = []
    for path in load_abc_files(raw_dir):
        for tune_idx, abc_text in enumerate(split_abc_tunes(path), start=1):
            records.append((path, tune_idx, abc_text))
    if max_tunes:
        records = records[:max_tunes]

    for path, tune_idx, abc_text in tqdm(records, desc="Parsing ABC tunes"):
        source = f"{path}::X{tune_idx}"
        try:
            score = converter.parse(abc_text, format="abc")
            tune = tokenize_score(score, source=source, grid=grid)
            if len(tune.melody) >= 16 and len(tune.conditioned) >= 16:
                tunes.append(tune)
        except Exception as exc:
            failed.append({"file": str(path), "tune_index": tune_idx, "error": str(exc)})
    return tunes, failed


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def split_tunes(tunes, val_fraction=0.15, seed=153):
    rng = random.Random(seed)
    shuffled = list(tunes)
    rng.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_fraction))
    return shuffled[val_count:], shuffled[:val_count]


def save_split(train, val, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_sequences = []
    for tune in train + val:
        all_sequences.append(tune.melody)
        all_sequences.append(tune.conditioned)

    vocab = Vocabulary.build(all_sequences)
    vocab.save(output_dir / "vocab.json")

    for split_name, split in [("train", train), ("val", val)]:
        write_jsonl(
            output_dir / f"{split_name}_unconditioned.jsonl",
            [{"source": tune.source, "tokens": tune.melody} for tune in split],
        )
        write_jsonl(
            output_dir / f"{split_name}_conditioned.jsonl",
            [
                {"source": tune.source, "tokens": tune.conditioned, "chords": tune.chords}
                for tune in split
            ],
        )

    summary = {
        "train_tunes": len(train),
        "val_tunes": len(val),
        "vocab_size": len(vocab),
        "avg_train_melody_tokens": sum(len(t.melody) for t in train) / max(1, len(train)),
        "avg_train_conditioned_tokens": sum(len(t.conditioned) for t in train) / max(1, len(train)),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Tokenize Nottingham ABC files.")
    parser.add_argument("--raw-dir", default="data/raw/nottingham/abc")
    parser.add_argument("--output-dir", default="data/processed/nottingham")
    parser.add_argument("--grid", type=int, default=4, help="Duration units per quarter note.")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=153)
    parser.add_argument("--max-tunes", type=int, default=None)
    args = parser.parse_args()

    tunes, failed = parse_tunes(args.raw_dir, grid=args.grid, max_tunes=args.max_tunes)
    if not tunes:
        raise RuntimeError(f"No usable tunes found in {args.raw_dir}")

    train, val = split_tunes(tunes, val_fraction=args.val_fraction, seed=args.seed)
    summary = save_split(train, val, args.output_dir)

    if failed:
        write_jsonl(Path(args.output_dir) / "parse_failures.jsonl", failed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
