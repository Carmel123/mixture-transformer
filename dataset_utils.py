import torch

from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# -----------------------------
# Synthetic Dataset
# -----------------------------

class RandomTokenDataset(Dataset):
    """
    Generates random token sequences:
    input_ids[:-1] -> input_ids[1:]
    """
    def __init__(self, num_samples, block_size, vocab_size):
        self.num_samples = num_samples
        self.block_size = block_size
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        tokens = torch.randint(
            0, self.vocab_size, (self.block_size + 1,)
        )
        return tokens[:-1], tokens[1:]

# -------------------------------------------------
# Tokenizer
# -------------------------------------------------

def build_tokenizer(dataset, vocab_size):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizers = pre_tokenizers.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<pad>"],
    )

    def gen_text():
        for x in dataset:
            yield x["text"]

    tokenizer.train_from_iterator(gen_text(), trainer)
    return tokenizer

# -------------------------------------------------
# WikiText Dataset
# -------------------------------------------------

class WikiTextDataset(Dataset):
    def __init__(self, dataset, split: str = "train", tokenizer=None, block_size: int = 128):
        assert tokenizer is not None

        self.block_size = block_size
        self.tokenizer = tokenizer


        # Concatenate all text and tokenize
        text = "\n\n".join(dataset["text"])
        enc = tokenizer.encode(text)
        token_ids = enc.ids

        # Split into blocks
        self.examples = []
        for i in range(0, len(token_ids) - block_size, block_size):
            block = token_ids[i : i + block_size]
            self.examples.append(torch.tensor(block, dtype=torch.long))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x = self.examples[idx]

        input_ids = x[:-1]
        labels = x[1:].clone()

        return {
            "input_ids": input_ids,
            "labels": labels,
        }