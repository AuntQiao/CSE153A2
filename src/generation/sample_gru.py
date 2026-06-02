import argparse
import json
from pathlib import Path

import torch

from src.generation.midi_writer import tokens_to_midi
from src.models.gru_model import GRUMusicModel
from src.preprocessing.vocab import BAR_TOKEN, END_TOKEN, PAD_TOKEN, START_TOKEN, Vocabulary


BLOCKED_SAMPLE_PREFIXES = ("CHORD_",)


def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint_path, vocab):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    model = GRUMusicModel(
        vocab_size=len(vocab),
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        pad_idx=vocab.token_to_id[PAD_TOKEN],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def sample_next(logits, vocab, temperature=0.9, top_k=12, allow_chords=False, blocked_tokens=None):
    logits = logits.clone()
    blocked_tokens = set(blocked_tokens or [])
    if temperature <= 0:
        temperature = 1.0
    logits = logits / temperature

    for token, idx in vocab.token_to_id.items():
        if token == PAD_TOKEN or token in blocked_tokens:
            logits[idx] = -float("inf")
        if not allow_chords and token.startswith(BLOCKED_SAMPLE_PREFIXES):
            logits[idx] = -float("inf")

    if top_k and top_k > 0:
        values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
        filtered = torch.full_like(logits, -float("inf"))
        filtered[indices] = values
        logits = filtered

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def generate_unconditioned(
    model,
    vocab,
    max_tokens=256,
    min_note_tokens=32,
    temperature=0.9,
    top_k=12,
    device=None,
):
    device = device or choose_device()
    model = model.to(device)
    generated = [START_TOKEN]
    input_id = torch.tensor([[vocab.token_to_id[START_TOKEN]]], dtype=torch.long, device=device)
    hidden = None

    for _ in range(max_tokens):
        logits, hidden = model(input_id, hidden)
        note_count = sum(1 for token in generated if token.startswith("NOTE_"))
        blocked = [END_TOKEN] if note_count < min_note_tokens else []
        next_id = sample_next(
            logits[0, -1],
            vocab,
            temperature=temperature,
            top_k=top_k,
            blocked_tokens=blocked,
        )
        token = vocab.id_to_token[next_id]
        generated.append(token)
        if token == END_TOKEN:
            break
        input_id = torch.tensor([[next_id]], dtype=torch.long, device=device)
    return generated


@torch.no_grad()
def generate_conditioned(
    model,
    vocab,
    chords,
    max_tokens_per_chord=24,
    temperature=0.9,
    top_k=12,
    device=None,
):
    device = device or choose_device()
    model = model.to(device)
    generated = [START_TOKEN]
    hidden = None

    def feed(token):
        nonlocal hidden
        token_id = vocab.token_to_id.get(token, vocab.token_to_id[START_TOKEN])
        input_id = torch.tensor([[token_id]], dtype=torch.long, device=device)
        logits, hidden = model(input_id, hidden)
        return logits[0, -1]

    feed(START_TOKEN)
    for chord in chords:
        chord = chord if chord.startswith("CHORD_") else f"CHORD_{chord}"
        if chord not in vocab.token_to_id:
            chord = "CHORD_NC"
        generated.append(chord)
        logits = feed(chord)

        for _ in range(max_tokens_per_chord):
            next_id = sample_next(logits, vocab, temperature=temperature, top_k=top_k)
            token = vocab.id_to_token[next_id]
            if token == END_TOKEN:
                token = BAR_TOKEN
            generated.append(token)
            logits = feed(token)
            if token == BAR_TOKEN:
                break
        if generated[-1] != BAR_TOKEN:
            generated.append(BAR_TOKEN)
            feed(BAR_TOKEN)

    generated.append(END_TOKEN)
    return generated


def read_chords(chords_arg, chords_json):
    if chords_json:
        with Path(chords_json).open("r", encoding="utf-8") as f:
            row = json.load(f)
        return row["chords"]
    return [item.strip() for item in chords_arg.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Generate MIDI from a trained GRU model.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", default="data/processed/nottingham/vocab.json")
    parser.add_argument("--mode", choices=["unconditioned", "conditioned"], required=True)
    parser.add_argument("--output-midi", required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-tokens-per-chord", type=int, default=24)
    parser.add_argument("--min-note-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--chords", default="CHORD_C,CHORD_G,CHORD_Am,CHORD_F,CHORD_C,CHORD_G,CHORD_C")
    parser.add_argument("--chords-json", default=None)
    parser.add_argument("--seed", type=int, default=153)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    vocab = Vocabulary.load(args.vocab)
    model = load_model(args.checkpoint, vocab)

    if args.mode == "unconditioned":
        tokens = generate_unconditioned(
            model,
            vocab,
            max_tokens=args.max_tokens,
            min_note_tokens=args.min_note_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    else:
        tokens = generate_conditioned(
            model,
            vocab,
            chords=read_chords(args.chords, args.chords_json),
            max_tokens_per_chord=args.max_tokens_per_chord,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    output_path = tokens_to_midi(tokens, args.output_midi)
    token_path = Path(args.output_midi).with_suffix(".tokens.json")
    with token_path.open("w", encoding="utf-8") as f:
        json.dump(tokens, f, indent=2)
    print(f"Wrote {output_path}")
    print(f"Wrote {token_path}")


if __name__ == "__main__":
    main()
