# evaluate.py
import time
import math
import torch
import wandb

from torch.utils.data import Dataset, DataLoader

from transformer import Transformer, TransformerConfig

# -----------------------------
# Configuration
# -----------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

VOCAB_SIZE = 32000
BLOCK_SIZE = 1024
BATCH_SIZE = 4

TRAIN_STEPS = 200
EVAL_STEPS = 50
LOG_EVERY = 20

LR = 3e-4

torch.manual_seed(0)


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


# -----------------------------
# Training
# -----------------------------

def train(model, dataloader, optimizer):
    model.train()
    start_time = time.perf_counter()

    total_tokens = 0
    total_loss = 0.0

    for step, (x, y) in enumerate(dataloader):
        if step >= TRAIN_STEPS:
            break

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        loss = model(x, labels=y)
        loss.backward()
        optimizer.step()

        tokens = x.numel()
        total_tokens += tokens
        total_loss += loss.item()

        if step % LOG_EVERY == 0:
            print(
                f"[train] step={step:4d} "
                f"loss={loss.item():.4f}"
            )

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = total_tokens / elapsed

    return {
        "train_loss": total_loss / TRAIN_STEPS,
        "train_time_sec": elapsed,
        "tokens_per_sec": tokens_per_sec,
    }


# -----------------------------
# Evaluation (Perplexity)
# -----------------------------

@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()

    start_time = time.perf_counter()
    total_loss = 0.0
    total_tokens = 0

    for step, (x, y) in enumerate(dataloader):
        if step >= EVAL_STEPS:
            break

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        loss = model(x, labels=y)

        total_loss += loss.item() * x.numel()
        total_tokens += x.numel()

    elapsed = time.perf_counter() - start_time
    avg_nll = total_loss / total_tokens
    perplexity = math.exp(avg_nll)

    return {
        "eval_nll": avg_nll,
        "perplexity": perplexity,
        "eval_time_sec": elapsed,
    }


# -----------------------------
# Main
# -----------------------------

def main():

    run = wandb.init(
    entity="anemia-pred",
    project="mixture-transformer",
    config={
        "learning_rate": LR,
        "architecture": "Transformer",
        "dataset": "Random",
        "train_steps": TRAIN_STEPS,
    })

    # Model config
    config = TransformerConfig(
        block_size=BLOCK_SIZE,
        vocab_size=VOCAB_SIZE,
        n_layer=6,
        n_head=8,
        dim=512,
        use_fused_ops=False,
    )

    model = Transformer(config).to(DEVICE)
    model.setup_cache(device=DEVICE)

    if DEVICE == "cuda":
        model = model.to(dtype=DTYPE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # Data
    train_ds = RandomTokenDataset(
        num_samples=10_000,
        block_size=BLOCK_SIZE,
        vocab_size=VOCAB_SIZE,
    )
    eval_ds = RandomTokenDataset(
        num_samples=2_000,
        block_size=BLOCK_SIZE,
        vocab_size=VOCAB_SIZE,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
    )

    eval_loader = DataLoader(
        eval_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
    )

    # Warmup
    print("Warming up...")
    x, y = next(iter(train_loader))
    x, y = x.to(DEVICE), y.to(DEVICE)
    for _ in range(5):
        loss = model(x, labels=y)
        loss.backward()
        optimizer.zero_grad(set_to_none=True)
        run.log({"loss": loss})

    torch.cuda.synchronize() if DEVICE == "cuda" else None

    # Training benchmark
    print("\nStarting training benchmark...")
    train_stats = train(model, train_loader, optimizer)
    run.log(train_stats)

    # Evaluation benchmark
    print("\nStarting evaluation...")
    eval_stats = evaluate(model, eval_loader)
    run.log(eval_stats)

    # Results
    print("\n================ RESULTS ================")
    print(f"Train loss           : {train_stats['train_loss']:.4f}")
    print(f"Train time (s)       : {train_stats['train_time_sec']:.2f}")
    print(f"Tokens / sec         : {train_stats['tokens_per_sec']:.2f}")
    print("-----------------------------------------")
    print(f"Eval NLL             : {eval_stats['eval_nll']:.4f}")
    print(f"Perplexity           : {eval_stats['perplexity']:.2f}")
    print(f"Eval time (s)        : {eval_stats['eval_time_sec']:.2f}")
    print("=========================================")


if __name__ == "__main__":
    main()
