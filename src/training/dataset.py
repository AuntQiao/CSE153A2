import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.preprocessing.vocab import Vocabulary


class MusicSequenceDataset(Dataset):
    def __init__(self, jsonl_path, vocab_path, sequence_length=128):
        self.vocab = Vocabulary.load(vocab_path)
        self.sequence_length = sequence_length
        self.examples = []
        self.sources = []

        with Path(jsonl_path).open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                ids = self.vocab.encode(row["tokens"])
                self.sources.append(row.get("source", ""))
                self.examples.extend(self._windows(ids))

    def _windows(self, ids):
        if len(ids) <= 1:
            return []
        stride = max(1, self.sequence_length // 2)
        windows = []
        for start in range(0, max(1, len(ids) - 1), stride):
            chunk = ids[start : start + self.sequence_length + 1]
            if len(chunk) >= 2:
                windows.append(chunk)
        return windows

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids = self.examples[idx]
        return torch.tensor(ids[:-1], dtype=torch.long), torch.tensor(ids[1:], dtype=torch.long)


def collate_batch(batch, pad_idx):
    inputs, targets = zip(*batch)
    max_len = max(item.size(0) for item in inputs)
    batch_inputs = torch.full((len(batch), max_len), pad_idx, dtype=torch.long)
    batch_targets = torch.full((len(batch), max_len), pad_idx, dtype=torch.long)

    for idx, (input_ids, target_ids) in enumerate(zip(inputs, targets)):
        batch_inputs[idx, : input_ids.size(0)] = input_ids
        batch_targets[idx, : target_ids.size(0)] = target_ids

    return batch_inputs, batch_targets
