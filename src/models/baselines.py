import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.preprocessing.vocab import END_TOKEN, START_TOKEN


class NGramModel:
    def __init__(self, n=3):
        self.n = n
        self.counts = defaultdict(Counter)
        self.unigram = Counter()

    def fit(self, sequences):
        prefix_len = self.n - 1
        for sequence in sequences:
            padded = [START_TOKEN] * prefix_len + sequence
            for idx in range(prefix_len, len(padded)):
                prefix = tuple(padded[idx - prefix_len : idx])
                token = padded[idx]
                self.counts[prefix][token] += 1
                self.unigram[token] += 1

    def next_distribution(self, prefix):
        prefix = tuple(prefix[-(self.n - 1) :])
        if prefix in self.counts:
            return self.counts[prefix]
        return self.unigram

    def sample(self, prefix, temperature=1.0):
        distribution = self.next_distribution(prefix)
        tokens, counts = zip(*distribution.items())
        weights = [float(count) ** (1.0 / max(temperature, 1e-6)) for count in counts]
        return random.choices(tokens, weights=weights, k=1)[0]

    def generate(self, max_tokens=256, temperature=1.0):
        generated = [START_TOKEN]
        while len(generated) < max_tokens:
            token = self.sample(generated, temperature=temperature)
            generated.append(token)
            if token == END_TOKEN:
                break
        return generated

    def perplexity(self, sequences, smoothing=1.0):
        vocab = set(self.unigram)
        vocab_size = max(1, len(vocab))
        total_log_prob = 0.0
        total_tokens = 0
        prefix_len = self.n - 1
        for sequence in sequences:
            padded = [START_TOKEN] * prefix_len + sequence
            for idx in range(prefix_len, len(padded)):
                prefix = tuple(padded[idx - prefix_len : idx])
                token = padded[idx]
                counts = self.counts.get(prefix, Counter())
                denom = sum(counts.values()) + smoothing * vocab_size
                prob = (counts[token] + smoothing) / denom
                total_log_prob += math.log(prob)
                total_tokens += 1
        return math.exp(-total_log_prob / max(1, total_tokens))


def load_sequences(path):
    sequences = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            sequences.append(json.loads(line)["tokens"])
    return sequences


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate an n-gram baseline.")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args()

    model = NGramModel(n=args.n)
    model.fit(load_sequences(args.train_jsonl))
    ppl = model.perplexity(load_sequences(args.val_jsonl))
    print(json.dumps({"n": args.n, "validation_perplexity": ppl}, indent=2))


if __name__ == "__main__":
    main()
