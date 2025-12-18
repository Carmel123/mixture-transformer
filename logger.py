import torch
import wandb


def log_moe_stats(
    *,
    gate_probs: torch.Tensor,   # [B, T, E]
    layer_idx: int,
    step: int,
    is_warmup: bool,
    log_interval: int = 100,
):
    if step % log_interval != 0:
        return

    B, T, E = gate_probs.shape
    device = gate_probs.device

    # -----------------------
    # 1. Per-layer expert load (weight mass)
    # -----------------------
    expert_mass = gate_probs.sum(dim=(0, 1))          # [E]
    expert_frac = expert_mass / expert_mass.sum()     # [E]

    # -----------------------
    # 2. Expert usage over sequence position
    # -----------------------
    # Average over batch → [T, E]
    pos_expert_mass = gate_probs.mean(dim=0)

    # -----------------------
    # 3. Routing entropy (sanity check)
    # -----------------------
    entropy = -(gate_probs * (gate_probs + 1e-9).log()).sum(dim=-1)
    mean_entropy = entropy.mean()

    # -----------------------
    # WandB logging
    # -----------------------
    log_dict = {
        f"moe/layer_{layer_idx}/expert_0_frac": expert_frac[0].item(),
        f"moe/layer_{layer_idx}/expert_1_frac": expert_frac[1].item(),
        f"moe/layer_{layer_idx}/entropy": mean_entropy.item(),
        f"moe/is_router_warmup": int(is_warmup),
    }

    # Sequence position curves (these render as line plots)
    if not is_warmup:
    # Build table rows
        data = []
        for pos in range(T):
            row = [pos]
            for e in range(E):
                row.append(pos_expert_mass[pos, e].item())
            data.append(row)

        # Column names
        columns = ["pos"] + [f"expert_{e}" for e in range(E)]

        table = wandb.Table(data=data, columns=columns)

        wandb.log(
            {
                f"moe/layer_{layer_idx}/experts_by_position": wandb.plot.line(
                    table,
                    x="pos",
                    ys=[f"expert_{e}" for e in range(E)],
                    title=f"Layer {layer_idx} – Expert Routing by Position",
                )
            },
            step=step,
        )

    wandb.log(log_dict, step=step)
