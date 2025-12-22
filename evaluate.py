# evaluate.py
import time
import math
import torch
import wandb
import argparse

from datasets import load_dataset
from dataset_utils import RandomTokenDataset, build_tokenizer, WikiTextDataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from train import train_model

from torch.utils.data import Dataset, DataLoader

from transformer import Transformer, TransformerConfig

from attn_transformer import MixTransformer, MixTransformerConfig

from ffn_transformer import FFNTransformer, FFNTransformerConfig

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.backends.cuda.enable_flash_sdp = True
torch.backends.cuda.enable_mem_efficient_sdp = True
torch.backends.cuda.enable_math_sdp = False

# -----------------------------
# Configuration
# -----------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

PROJECT = "mixture-project"

VOCAB_SIZE = 32000
BLOCK_SIZE = 256
# BATCH_SIZE = 4
BATCH_SIZE = 32

# TRAIN_STEPS = 250
TRAIN_STEPS = 120000
WARMUP_STEPS = 200
TOT_STEPS = TRAIN_STEPS + WARMUP_STEPS
EVAL_STEPS = 50
LOG_EVERY = 100
LAYERS = 3
# MAX_SEQ_LEN = 256

LR = 3e-4

GEN_TOKENS = 128
GEN_WARMUP = 8

PATH = '/scratch/clp9358/mixer-mixture/'

torch.manual_seed(0)



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
            loss, _ = model(x, labels=y, global_step=step, is_warmup=False)

        total_loss += loss.item() * x.numel()
        total_tokens += x.numel()

    elapsed = time.perf_counter() - start_time
    avg_nll = total_loss / total_tokens
    perplexity = math.exp(avg_nll)

    return {
        "avg_nll": avg_nll,
        "perplexity": perplexity,
        "time_sec": elapsed,
    }

# -------------------------------------------------
# Generation Throughput
# -------------------------------------------------

@torch.no_grad()
def benchmark_generation(model, tokenizer, arch):
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
    print(f"Benchmark input ids: {input_ids}")
    print(f'Benchmark input pos: {input_pos}')
    if arch == 0:
        # logits = model(input_ids, input_pos=input_pos, 
        #                   is_warmup=True, global_step=1)
        logits = model(input_ids, is_warmup=True, global_step=1)
    elif arch == 1:
        # logits = model(input_ids, input_pos=input_pos)
        logits = model(input_ids)

    torch.cuda.synchronize() if DEVICE == "cuda" else None
    decode_positions = torch.arange(
        GEN_WARMUP,
        GEN_WARMUP + GEN_TOKENS,
        device=DEVICE
    )
    print('Warmup complete')
    start = time.perf_counter()

    cur_token = logits[:, -1].argmax(dim=-1, keepdim=True)
    for i in range(GEN_TOKENS):
        print(f'Generating token {i}')
        pos = decode_positions[i].unsqueeze(0)
        if arch == 0:
            logits = model(cur_token, input_pos=pos, 
                            is_warmup=True, global_step=1)
        elif arch == 1:
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

def main(arch, data, n_epochs, evaluate_only, model_path, use_fused_ops):

    architecture = "Mix-Transformer"
    if arch == 1:
        architecture = "Transformer"
    if arch == 2:
        architecture = "FFN-Transformer"
    

    run = wandb.init(entity="anemia-pred", project=PROJECT,
    config={"learning_rate": LR, "architecture": architecture,
        "dataset": data, "train_steps": TRAIN_STEPS,
        "evaluate_only": evaluate_only,
        "use_fused_ops": use_fused_ops,
        "n_layer": LAYERS},)

    # Model config
    if arch == 1: 
        config = TransformerConfig(
        block_size=BLOCK_SIZE,
        vocab_size=VOCAB_SIZE,
        n_layer=LAYERS,
        # n_layer=6,
        n_head=8,
        dim=768,
        use_fused_ops=use_fused_ops,
        )
        model = Transformer(config).to(DEVICE)

    elif arch == 0:
        config = MixTransformerConfig(
            block_size=BLOCK_SIZE,
            vocab_size=VOCAB_SIZE,
            n_layer=LAYERS,
            # n_layer=6,
            n_head=8,
            dim = 384, # for mix transformer
            use_fused_ops=use_fused_ops,
            n_expert=2 # for mix transformer
        )
        model = MixTransformer(config).to(DEVICE)
    
    else:
        config = FFNTransformerConfig(
            block_size=BLOCK_SIZE,
            vocab_size=VOCAB_SIZE,
            n_layer=LAYERS,
            # n_layer=6,
            n_head=8,
            dim = 768,
            use_fused_ops=use_fused_ops,
            n_expert=2 # for mix transformer
        )
        model = MixTransformer(config).to(DEVICE)
    
    model.setup_cache(device=DEVICE)

    if DEVICE == "cuda":
        model = model.to(dtype=DTYPE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # Random Data - might not work
    if data == 'rand':
        if evaluate_only == 0:
            train_ds = RandomTokenDataset(num_samples=10_000, block_size=BLOCK_SIZE, vocab_size=VOCAB_SIZE)
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
        eval_ds = RandomTokenDataset(num_samples=2_000, block_size=BLOCK_SIZE, vocab_size=VOCAB_SIZE)
        eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    
    # Wiki Data
    if data == 'wiki':
        # dataset = load_dataset('wikitext', "wikitext-103-raw-v1")
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

        tokenizer = build_tokenizer(
            dataset['train'], vocab_size=VOCAB_SIZE
        )
        # tokenizer.model_max_length = MAX_SEQ_LEN
        train_ds = WikiTextDataset(dataset['train'], tokenizer=tokenizer, block_size=BLOCK_SIZE)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        eval_ds = WikiTextDataset(dataset['validation'], tokenizer=tokenizer, block_size=BLOCK_SIZE)
        eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE)

    # Warmup
    print("Warming up...")
    batch = next(iter(train_loader))

    x = batch["input_ids"].to(DEVICE)
    y = batch["labels"].to(DEVICE)

    for i in range(WARMUP_STEPS):
        if arch == 1:
            loss = model(x, labels=y)
        else:
            loss, _ = model(x, global_step=i, is_warmup=True, labels=y)
        loss.backward()
        optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize() if DEVICE == "cuda" else None

    # Training benchmark
    if evaluate_only == 0:
        print("\nStarting training...")
        train_stats = train_model(model, train_loader, optimizer, n_epochs, arch,
                                  WARMUP_STEPS, TOT_STEPS, LOG_EVERY, DEVICE)
        run.log(train_stats)

        wandb.log({"summary/train_avg_loss": train_stats["avg_loss"],
            'summary/train_avg_nll': train_stats['avg_nll'],
            "summary/tokens_per_sec": train_stats["tokens_per_sec"]})

        # Save model
        torch.save(model.state_dict(), f'{PATH}{architecture[:3]}-{data}-mod.pt')
        wandb.save(f'{PATH}{architecture[:3]}-{data}-mod.pt')
    
    # Load model
    if evaluate_only == 1:
        print("\nLoading model...")
        model.load_state_dict(torch.load(f'{PATH}{architecture[:3]}-{data}-mod.pt', weights_only=True))

    # Evaluation benchmark
    print("\nStarting evaluation...")
    eval_stats = evaluate(model, eval_loader, arch)
    run.log(eval_stats)

    wandb.log(
        {
            "summary/eval_nll": eval_stats["avg_nll"],
            "summary/eval_perplexity": eval_stats["perplexity"]}
    )
    # Generation benchmark
    print('\nStarting generation...')
    gen_stats = benchmark_generation(model, tokenizer, arch)

    # Results
    wandb.log(
        {
            "summary/gen_tokens_per_sec": gen_stats["gen_tokens_per_sec"],
            "summary/gen_ms_per_token": gen_stats["gen_ms_per_token"],
        }
    )

    print("\n============== FINAL RESULTS ==============")
    if not evaluate_only:
        print(f"Train loss        : {train_stats['avg_loss']:.4f}")
        print(f"Tokens / sec      : {train_stats['tokens_per_sec']:.0f}")
    print(f"Eval PPL          : {eval_stats['perplexity']:.2f}")
    print(f"Gen tokens / sec  : {gen_stats['gen_tokens_per_sec']:.2f}")
    print(f"Gen ms / token    : {gen_stats['gen_ms_per_token']:.2f}")
    print("==========================================")

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset creation")

    parser.add_argument("--arch", default=0 , type=int, 
                        help="0 - attn transformer, 1 - transformer, 2 - ffn transformer")

    parser.add_argument("--data", default='wiki', type=str,
                        help='wiki or rand')
    
    parser.add_argument('--epochs', default=5, type=int)

    parser.add_argument('--evaluate_only', default = 0, type=int,
                        help='0 - train and evaluate, 1 - evaluate (need to include path to model weights)')
    
    parser.add_argument('--model_path', default= '', type=str,
                        help='path to model weights, only used if evaluate_only is true')
    
    parser.add_argument('--use_fused_ops', default=0, type=int,
                        help="0 - use pytorch ops, 1 - use liger fused ops")
    
    args = parser.parse_args()

    main(args.arch, args.data, args.epochs, args.evaluate_only, args.model_path, args.use_fused_ops)
