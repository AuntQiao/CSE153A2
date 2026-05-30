import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.gru_model import GRUMusicModel
from src.preprocessing.vocab import PAD_TOKEN, Vocabulary
from src.training.dataset import MusicSequenceDataset, collate_batch


def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for inputs, targets in tqdm(loader, desc="train" if train else "val"):
        inputs = inputs.to(device)
        targets = targets.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits, _ = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        non_pad = targets.ne(criterion.ignore_index).sum().item()
        total_loss += loss.item() * max(1, non_pad)
        total_tokens += max(1, non_pad)

    return total_loss / max(1, total_tokens)


def train(args):
    vocab = Vocabulary.load(args.vocab)
    pad_idx = vocab.token_to_id[PAD_TOKEN]

    train_dataset = MusicSequenceDataset(args.train_jsonl, args.vocab, args.sequence_length)
    val_dataset = MusicSequenceDataset(args.val_jsonl, args.vocab, args.sequence_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, pad_idx),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, pad_idx),
    )

    device = choose_device()
    model = GRUMusicModel(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pad_idx=pad_idx,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    best_val = float("inf")

    config = vars(args).copy()
    config["vocab_size"] = len(vocab)
    config["pad_idx"] = pad_idx
    config["device"] = str(device)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_perplexity": float(torch.exp(torch.tensor(val_loss)).item()),
        }
        metrics.append(row)
        print(json.dumps(row, indent=2))

        checkpoint = {
            "model_state": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "val_loss": val_loss,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, output_dir / "best.pt")

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train a GRU symbolic music model from scratch.")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--vocab", default="data/processed/nottingham/vocab.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
