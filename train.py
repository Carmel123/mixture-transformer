import time
import math
import torch
import wandb


# -----------------------------
# Training
# -----------------------------

def train_model(model, dataloader, optimizer, n_epochs, arch,
          warmup_steps, tot_steps, log_every, device):
    model.train()
    start_time = time.perf_counter()

    total_tokens = 0
    total_loss = 0.0

    global_step = warmup_steps
    
    get_shape = True

    for epoch in range(n_epochs):
        for step, batch in enumerate(dataloader):
            if global_step >= tot_steps:
                break

            global_step += 1
            is_warmup = global_step < warmup_steps

            x = batch['input_ids'].to(device)
            y = batch['labels'].to(device)
        
            if get_shape:
                print(f'x shape: {x.shape}')
                get_shape = False

            optimizer.zero_grad(set_to_none=True)

            if arch == 1:
                loss = model(x, labels=y)
            else:
                loss, loss_dict = model(x, labels=y, 
                                        is_warmup = is_warmup, 
                                        global_step = global_step)
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

            if step % log_every == 0:
                print(
                    f'epoch = {epoch} '
                    f"[train] step={step:4d} "
                    f"loss={loss.item():.4f}"
                )

        elapsed = time.perf_counter() - start_time
        tokens_per_sec = total_tokens / elapsed

    out = {
        # "train_loss": total_loss / TRAIN_STEPS,
        'avg_loss': total_loss / global_step,
        'avg_nll': total_loss / total_tokens,
        "time_sec": elapsed,
        "tokens_per_sec": tokens_per_sec,
    }
    if not arch:
        out['lm_loss'] = loss_dict['lm_loss'].item()
        out['aux_loss'] = loss_dict['aux_loss'].item()

    return out