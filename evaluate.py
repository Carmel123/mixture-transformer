# evaluate.py
import time
import math
import torch
import wandb
import argparse

from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

from torch.utils.data import Dataset, DataLoader

from transformer import Transformer, TransformerConfig

from attn_transformer import MixTransformer, MixTransformerConfig

from ffn_transformer import FFNTransformer, FFNTransformerConfig

# -----------------------------
# Configuration
# -----------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

PROJECT = "mixture-transformer"

VOCAB_SIZE = 32000
BLOCK_SIZE = 1024
# BATCH_SIZE = 4
BATCH_SIZE = 32

# TRAIN_STEPS = 250
TRAIN_STEPS = 300000
WARMUP_STEPS = 100
EVAL_STEPS = 50
LOG_EVERY = 100

LR = 3e-4

GEN_TOKENS = 256
GEN_WARMUP = 8

PATH = '/scratch/clp9358/mixer-mixture/'

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
# -----------------------------
# Training
# -----------------------------

def train(model, dataloader, optimizer, n_epochs, arch):
    model.train()
    start_time = time.perf_counter()

    total_tokens = 0
    total_loss = 0.0

    global_step = 0

    for epoch in range(n_epochs):
        for step, batch in enumerate(dataloader):
            if global_step >= TRAIN_STEPS:
                break

            global_step += 1

            x = batch['input_ids'].to(DEVICE)
            y = batch['labels'].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            if arch == 1:
                loss = model(x, labels=y)
            else:
                loss, loss_dict = model(x, labels=y)
            loss.backward()
            optimizer.step()

            tokens = x.numel()
            total_tokens += tokens
            total_loss += loss.item()

            if arch:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        'train/epoch': epoch,
                        'train/tokens_seen': total_tokens
                    }
                )
            else:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        'train/aux_loss': loss_dict['aux_loss'],
                        'train/lm_loss': loss_dict['lm_loss'],
                        'train/epoch': epoch,
                        'train/tokens_seen': total_tokens
                    }
                )

            if step % LOG_EVERY == 0:
                print(
                    f'epoch = {epoch} '
                    f"[train] step={step:4d} "
                    f"loss={loss.item():.4f}"
                )

        elapsed = time.perf_counter() - start_time
        tokens_per_sec = total_tokens / elapsed

    out = {
        # "train_loss": total_loss / TRAIN_STEPS,
        'train_avg_loss': total_loss / global_step,
        'train_avg_nll': total_loss / total_tokens,
        "train_time_sec": elapsed,
        "tokens_per_sec": tokens_per_sec,
    }
    if not arch:
        out['train_lm_loss'] = loss_dict['lm_loss'].item()
        out['train_aux_loss'] = loss_dict['aux_loss'].item()

    return out


# --------------
# Evaluation 
# --------------

@torch.no_grad()
def evaluate(model, dataloader, arch):
    model.eval()

    start_time = time.perf_counter()
    total_loss = 0.0
    total_tokens = 0

    for step, batch in enumerate(dataloader):
        if step >= EVAL_STEPS:
            break

        x = batch['input_ids'].to(DEVICE)
        y = batch['labels'].to(DEVICE)

        if arch == 1:
            loss = model(x, labels=y)
        else:
            loss, _ = model(x, labels=y)

        total_loss += loss.item() * x.numel()
        total_tokens += x.numel()

    elapsed = time.perf_counter() - start_time
    avg_nll = total_loss / total_tokens
    perplexity = math.exp(avg_nll)

    return {
        "eval_avg_nll": avg_nll,
        "perplexity": perplexity,
        "eval_time_sec": elapsed,
    }

# -------------------------------------------------
# Generation Throughput
# -------------------------------------------------

@torch.no_grad()
def benchmark_generation(model, tokenizer):
    model.eval()

    input_ids = torch.randint(
        0, tokenizer.get_vocab_size(), (1, GEN_WARMUP), device=DEVICE
    )

    model.setup_kv_cache(
        max_batch_size=1,
        dtype=DTYPE,
        device=torch.device(DEVICE),
    )

    input_pos = torch.arange(GEN_WARMUP, device=DEVICE)
    logits = model(input_ids, input_pos=input_pos)

    torch.cuda.synchronize() if DEVICE == "cuda" else None
    decode_positions = torch.arange(
        GEN_WARMUP,
        GEN_WARMUP + GEN_TOKENS,
        device=DEVICE
    )

    start = time.perf_counter()

    cur_token = logits[:, -1].argmax(dim=-1, keepdim=True)
    for i in range(GEN_TOKENS):
        pos = decode_positions[i].unsqueeze(0)
        logits = model(cur_token, input_pos=pos)
        cur_token = torch.argmax(logits[:, -1:], dim=-1)

    torch.cuda.synchronize() if DEVICE == "cuda" else None
    elapsed = time.perf_counter() - start

    tps = GEN_TOKENS / elapsed
    ms_per_token = 1000 * elapsed / GEN_TOKENS

    return {
        "gen_tokens_per_sec": tps,
        "gen_ms_per_token": ms_per_token,
    }


# -----------------------------
# Main
# -----------------------------

def main(arch, data, n_epochs):

    architecture = "Mix-Transformer"
    if arch == 1:
        architecture = "Transformer"
    if arch == 2:
        architecture = "FFN-Transformer"
    

    run = wandb.init(
    entity="anemia-pred",
    project=PROJECT,
    config={
        "learning_rate": LR,
        "architecture": architecture,
        "dataset": data,
        "train_steps": TRAIN_STEPS,
    },)

    # Model config
    if arch == 1: 
        config = TransformerConfig(
        block_size=BLOCK_SIZE,
        vocab_size=VOCAB_SIZE,
        n_layer=6,
        n_head=8,
        dim=768,
        use_fused_ops=False,
        )
        model = Transformer(config).to(DEVICE)

    elif arch == 0:
        config = MixTransformerConfig(
            block_size=BLOCK_SIZE,
            vocab_size=VOCAB_SIZE,
            n_layer=6,
            n_head=8,
            dim = 384, # for mix transformer
            use_fused_ops=False,
            n_expert=2 # for mix transformer
        )
        model = MixTransformer(config).to(DEVICE)
    
    else:
        config = FFNTransformerConfig(
            block_size=BLOCK_SIZE,
            vocab_size=VOCAB_SIZE,
            n_layer=6,
            n_head=8,
            dim = 768,
            use_fused_ops=False,
            n_expert=2 # for mix transformer
        )
        model = MixTransformer(config).to(DEVICE)
    
    model.setup_cache(device=DEVICE)

    if DEVICE == "cuda":
        model = model.to(dtype=DTYPE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # Random Data
    if data == 'rand':
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
    
    # Wiki Data
    if data == 'wiki':
        dataset = load_dataset('wikitext', "wikitext-103-raw-v1")

        tokenizer = build_tokenizer(
            dataset['train'], vocab_size=VOCAB_SIZE
        )
        train_ds = WikiTextDataset(
            dataset['train'],
            tokenizer=tokenizer,
            block_size=BLOCK_SIZE
        )

        eval_ds = WikiTextDataset(
            dataset['validation'],
            tokenizer=tokenizer,
            block_size=BLOCK_SIZE
        )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE)

    # Warmup
    print("Warming up...")
    batch = next(iter(train_loader))

    x = batch["input_ids"].to(DEVICE)
    y = batch["labels"].to(DEVICE)

    for _ in range(WARMUP_STEPS):
        if arch == 1:
            loss = model(x, labels=y)
        else:
            loss, _ = model(x, labels=y)
        loss.backward()
        optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize() if DEVICE == "cuda" else None

    # Training benchmark
    print("\nStarting training benchmark...")
    train_stats = train(model, train_loader, optimizer, n_epochs, arch)
    run.log(train_stats)

    # Evaluation benchmark
    print("\nStarting evaluation...")
    eval_stats = evaluate(model, eval_loader, arch)
    run.log(eval_stats)

    # Generation benchmark
    print('\nStarting generation...')
    gen_stats = benchmark_generation(model, tokenizer)

    # Results
    wandb.log(
        {
            "summary/train_avg_loss": train_stats["avg_loss"],
            'summary/train_avg_nll': train_stats['avg_nll'],
            "summary/tokens_per_sec": train_stats["tokens_per_sec"],
            "summary/eval_nll": eval_stats["eval_nll"],
            "summary/eval_perplexity": eval_stats["perplexity"],
            "summary/gen_tokens_per_sec": gen_stats["gen_tokens_per_sec"],
            "summary/gen_ms_per_token": gen_stats["gen_ms_per_token"],
        }
    )

    print("\n============== FINAL RESULTS ==============")
    print(f"Train loss        : {train_stats['avg_floss']:.4f}")
    print(f"Tokens / sec      : {train_stats['tokens_per_sec']:.0f}")
    print(f"Eval PPL          : {eval_stats['perplexity']:.2f}")
    print(f"Gen tokens / sec  : {gen_stats['gen_tokens_per_sec']:.2f}")
    print(f"Gen ms / token    : {gen_stats['gen_ms_per_token']:.2f}")
    print("==========================================")

    # Save model
    torch.save(model.state_dict(), f'{PATH}-{arch[:3]}-{data}-mod.pt')
    wandb.save(f'{PATH}-{arch[:3]}-{data}-mod.pt')
    
    wandb.finish()

    # print("\n================ RESULTS ================")
    # print(f"Train loss           : {train_stats['train_loss']:.4f}")
    # print(f"Train time (s)       : {train_stats['train_time_sec']:.2f}")
    # print(f"Tokens / sec         : {train_stats['tokens_per_sec']:.2f}")
    # print("-----------------------------------------")
    # print(f"Eval NLL             : {eval_stats['eval_nll']:.4f}")
    # print(f"Perplexity           : {eval_stats['perplexity']:.2f}")
    # print(f"Eval time (s)        : {eval_stats['eval_time_sec']:.2f}")
    # print("=========================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset creation")

    parser.add_argument("--arch", default=0 , type=int, 
                        help="0 - attn transformer, 1 - transformer, 2 - ffn transformer")

    parser.add_argument("--data", default='wiki', type=str,
                        help='wiki or rand')
    
    parser.add_argument('--epochs', default=5, type=int)
    
    args = parser.parse_args()

    main(args.arch, args.data, args.epochs)
