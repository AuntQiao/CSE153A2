# CSE 153/253 Assignment 2

## Project Overview

This project completes two symbolic music generation tasks using models trained from
scratch on the Nottingham folk tune dataset.

### Task 1: Symbolic Unconditioned Melody Generation

The model is trained on melody tokens only. At generation time it receives only
`<START>` and autoregressively samples a new melody.

Final output:

- `symbolic_unconditioned.mid`

### Task 2: Symbolic Chord-Conditioned Melody Generation

The model is trained on paired chord tokens and melody tokens. At generation time
it receives a chord progression and samples melody events aligned with those
chords.

Final output:

- `symbolic_conditioned.mid`

## Repository Structure

- `data/raw/` - downloaded source data, including Nottingham ABC files.
- `data/processed/` - tokenized train/validation splits and vocabulary.
- `notebooks/` - workbook notebook for EDA, modeling discussion, and evaluation.
- `src/preprocessing/` - dataset download, parsing, tokenization, and splits.
- `src/models/` - n-gram baseline and GRU model definitions.
- `src/training/` - PyTorch dataset and training loop.
- `src/generation/` - GRU sampling and MIDI writing.
- `src/evaluation/` - generated-token metrics.
- `outputs/midi/` - generated MIDI files and token traces.
- `outputs/checkpoints/` - trained model checkpoints.
- `outputs/metrics/` - training and generation metrics.
- `figures/` - plots used in the workbook and presentation.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproducible Pipeline

Download Nottingham:

```bash
python -m src.preprocessing.download_nottingham
```

Tokenize into melody-only and chord-conditioned training files:

```bash
python -m src.preprocessing.tokenize_nottingham \
  --raw-dir data/raw/nottingham/abc \
  --output-dir data/processed/nottingham
```

Create exploratory dataset statistics and plots for the workbook:

```bash
python -m src.analysis.dataset_summary \
  --processed-dir data/processed/nottingham \
  --output-json outputs/metrics/dataset_summary.json \
  --figures-dir figures
```

Create training and baseline comparison plots:

```bash
python -m src.analysis.training_plots \
  --figures-dir figures
```

Train the unconditioned GRU:

```bash
python -m src.training.train_gru \
  --train-jsonl data/processed/nottingham/train_unconditioned.jsonl \
  --val-jsonl data/processed/nottingham/val_unconditioned.jsonl \
  --output-dir outputs/checkpoints/unconditioned \
  --epochs 20
```

Train the chord-conditioned GRU:

```bash
python -m src.training.train_gru \
  --train-jsonl data/processed/nottingham/train_conditioned.jsonl \
  --val-jsonl data/processed/nottingham/val_conditioned.jsonl \
  --output-dir outputs/checkpoints/conditioned \
  --epochs 20
```

Run n-gram baselines for comparison:

```bash
python -m src.models.baselines \
  --train-jsonl data/processed/nottingham/train_unconditioned.jsonl \
  --val-jsonl data/processed/nottingham/val_unconditioned.jsonl \
  --n 3 \
  --output-json outputs/metrics/ngram_unconditioned.json \
  --sample-json outputs/metrics/ngram_unconditioned_sample.tokens.json \
  --sample-midi outputs/midi/ngram_unconditioned.mid

python -m src.models.baselines \
  --train-jsonl data/processed/nottingham/train_conditioned.jsonl \
  --val-jsonl data/processed/nottingham/val_conditioned.jsonl \
  --n 3 \
  --output-json outputs/metrics/ngram_conditioned.json \
  --sample-json outputs/metrics/ngram_conditioned_sample.tokens.json \
  --sample-midi outputs/midi/ngram_conditioned.mid
```

Generate the required submission MIDI files:

```bash
python -m src.generation.sample_gru \
  --checkpoint outputs/checkpoints/unconditioned/best.pt \
  --mode unconditioned \
  --output-midi symbolic_unconditioned.mid

python -m src.generation.sample_gru \
  --checkpoint outputs/checkpoints/conditioned/best.pt \
  --mode conditioned \
  --chords CHORD_C,CHORD_G,CHORD_Am,CHORD_F,CHORD_C,CHORD_G,CHORD_C \
  --output-midi symbolic_conditioned.mid
```

Evaluate generated token traces:

```bash
python -m src.evaluation.metrics \
  --tokens-json symbolic_unconditioned.tokens.json \
  --reference-jsonl data/processed/nottingham/val_unconditioned.jsonl \
  --output-json outputs/metrics/unconditioned_generation.json

python -m src.evaluation.metrics \
  --tokens-json symbolic_conditioned.tokens.json \
  --reference-jsonl data/processed/nottingham/val_conditioned.jsonl \
  --output-json outputs/metrics/conditioned_generation.json

python -m src.evaluation.training_summary \
  --baseline-json outputs/metrics/ngram_unconditioned.json \
  --baseline-json outputs/metrics/ngram_conditioned.json \
  --output-json outputs/metrics/training_summary.json
```

Export the workbook:

```bash
jupyter nbconvert --to html notebooks/workbook.ipynb --output ../workbook.html
```

## References

- Nottingham Music Database: https://abc.sourceforge.net/NMD/
- Music21: https://web.mit.edu/music21/
- PyTorch: https://pytorch.org/
