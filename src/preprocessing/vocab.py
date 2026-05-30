import json
from pathlib import Path


SPECIAL_TOKENS = ["<PAD>", "<START>", "<END>", "<BAR>", "<UNK>"]
PAD_TOKEN = "<PAD>"
START_TOKEN = "<START>"
END_TOKEN = "<END>"
BAR_TOKEN = "<BAR>"
UNK_TOKEN = "<UNK>"


class Vocabulary:
    def __init__(self, token_to_id):
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    @classmethod
    def build(cls, sequences):
        tokens = list(SPECIAL_TOKENS)
        seen = set(tokens)
        for sequence in sequences:
            for token in sequence:
                if token not in seen:
                    seen.add(token)
                    tokens.append(token)
        return cls({token: idx for idx, token in enumerate(tokens)})

    @classmethod
    def load(cls, path):
        with Path(path).open("r", encoding="utf-8") as f:
            return cls(json.load(f))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, indent=2, sort_keys=True)

    def encode(self, sequence):
        unk = self.token_to_id[UNK_TOKEN]
        return [self.token_to_id.get(token, unk) for token in sequence]

    def decode(self, ids):
        return [self.id_to_token.get(int(idx), UNK_TOKEN) for idx in ids]

    def __len__(self):
        return len(self.token_to_id)
