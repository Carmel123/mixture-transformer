"""
A vanilla transformer implementation. 
Nothing fancy here :)

Parts of it were taken from:
1. https://github.com/pytorch-labs/gpt-fast
2. https://github.com/Lightning-AI/litgpt
3. https://medium.com/@prateeksikdar/understanding-mixture-of-experts-building-a-moe-model-with-pytorch-dd373d9db81c

"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from logger import log_moe_stats

from transformer import apply_rope_emb, build_rope_cache, _init_weights, KVCache, RMSNorm, LigerSwiGLUMLP, LLaMAMLP, Attention

from liger_kernel.transformers import LigerRMSNorm, liger_rotary_pos_emb, LigerFusedLinearCrossEntropyLoss
from liger_kernel.ops.swiglu import LigerSiLUMulFunction

def find_multiple(n: int, k: int) -> int:
    '''find n such that n is a multiple of k'''
    if n % k == 0:
        return n
    return n + k - (n % k)


from torch.nn.attention.flex_attention import flex_attention, BlockMask, _mask_mod_signature
_flex_attention_compiled = torch.compile(flex_attention, dynamic=False, mode="default")

@torch.compiler.disable(recursive=False)
def flex_attention_compiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: Optional[BlockMask] = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    return _flex_attention_compiled(q, k, v, block_mask=block_mask, enable_gqa=enable_gqa)


# used during generation to shift the mask
def get_mask_mod(mask_mod: _mask_mod_signature, offset: int):
    def _mask_mod(b, h, q, kv):
        return mask_mod(b, h, q + offset, kv)
    return _mask_mod

@dataclass
class MixTransformerConfig:
    block_size: int = 2048
    vocab_size: int = 32000
    padded_vocab_size: Optional[int] = None

    n_layer: int = 6
    n_head: int = 12
    dim: int = 768
    intermediate_size: Optional[int] = None
    n_local_heads: int = -1
    head_dim: Optional[int] = None

    rope_base: float = 10000
    norm_eps: float = 1e-5
    rope_n_elem: Optional[int] = None
    
    # for gating
    n_expert: int = 2
    dropout_rate: float = 0.1
    aux_weight: float = 0.01
    aux_warmup_weight: float = 0.01

    # optional
    use_fused_ops: bool = False
    use_qk_norm: bool = False

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)

        assert self.dim % self.n_head == 0
        self.head_dim = self.dim // self.n_head

        self.padded_vocab_size = find_multiple(self.vocab_size, 256)
        self.rope_n_elem = self.head_dim


class MixTransformer(nn.Module):
    def __init__(self, config: MixTransformerConfig) -> None: 
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.padded_vocab_size, config.dim)
        self.layers = nn.ModuleList(MixtureTransformerBlock(config, layer_idx=i) for i in range(config.n_layer))
        
        if self.config.use_fused_ops:
            self.norm = LigerRMSNorm(config.dim, eps=config.norm_eps)
            print('using liger')
        else:
            self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.padded_vocab_size, bias=False)

        if self.config.use_fused_ops:
            self.fused_linear_cross_entropy = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)
            print('using liger')
        
        # initialize weights
        self.apply(lambda m: _init_weights(m, self.config.n_layer, self.config.dim))
        self.get_mask_mod = get_mask_mod

    def setup_cache(self, device=None):
        # force in fp32
        # this happens after the model has been created and move to respective device

        cos, sin = build_rope_cache(
            self.config.block_size, self.config.rope_n_elem, device=device, base=self.config.rope_base
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        print("created cos and sin cache ...")
        print("cos shape:", self.cos.shape)
        print("cos dtype:", self.cos.dtype)

    # used for generation
    def setup_kv_cache(self, max_batch_size: int, dtype, device: torch.device):
        print("Setting up kv cache ...")
        for block in self.layers:
            for expert in block.experts:
                expert.kv_cache = KVCache(
                    max_batch_size, self.config.block_size, self.config.n_local_heads, self.config.head_dim, dtype, device
                )

    def forward(
        self,
        input_ids: torch.LongTensor,
        is_warmup: bool,
        global_step: int, 
        labels: Optional[torch.LongTensor] = None, 
        input_pos: Optional[Tensor] = None, 
        mask: Optional[BlockMask] = None
    ) -> Tensor:
        bsz, seqlen = input_ids.shape

        if (mask is not None) and (input_pos is not None):
            # doing generation
            mask.mask_mod = self.get_mask_mod(mask.mask_mod, input_pos[0])

        if input_pos is not None:
            cos = self.cos[:, input_pos]
            sin = self.sin[:, input_pos]
        else:
            # trim cos, sin
            cos = self.cos[:, :seqlen]
            sin = self.sin[:, :seqlen]

        x = self.wte(input_ids)
        tot_aux_loss = torch.tensor(0.0, device=x.device)
        for i, layer in enumerate(self.layers):
            x, aux_loss = layer(x, cos, sin, mask=mask, 
                                input_pos=input_pos, global_step = global_step,
                                is_warmup = is_warmup)
            tot_aux_loss += aux_loss
        
        x = self.norm(x)
        tot_aux_loss = tot_aux_loss / len(self.layers)

        if labels is not None:
            if self.config.use_fused_ops:
                loss = self.fused_linear_cross_entropy(
                    self.output.weight, x.view(-1, x.size(-1)), labels.view(-1)
                ) # need to reshape to x to (B*N, D) and labels to (B*N)
                return loss
            else:
                logits = self.output(x)
                lm_loss = F.cross_entropy(logits.view(-1, 
                                        logits.size(-1)),
                                        labels.view(-1),
                                        ignore_index=-100)
                
                lb_weight = self.config.aux_warmup_weight if is_warmup else self.config.aux_weight
                loss = lm_loss + lb_weight * tot_aux_loss
                return loss, {'lm_loss': lm_loss.detach(), 'aux_loss': tot_aux_loss.detach()}
        
        logits = self.output(x)
        return logits


class MixtureTransformerBlock(nn.Module):
    def __init__(self, config: MixTransformerConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.router = Router(config, layer_idx)
        # make more modular by replacing Attention
        self.experts = nn.ModuleList(Attention(config, layer_idx=i) for i in range(config.n_expert))

        if config.use_fused_ops:
            self.feed_forward = LigerSwiGLUMLP(config)
        else:
            self.feed_forward = LLaMAMLP(config)

        if self.config.use_fused_ops:
            self.ffn_norm = LigerRMSNorm(config.dim, eps=config.norm_eps)
            self.attention_norm = LigerRMSNorm(config.dim, eps=config.norm_eps)
        else:
            self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
            self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor,
                is_warmup: bool,
                global_step: int,
                is_causal: Optional[bool] = True, 
                mask: Optional[BlockMask] = None, 
                input_pos: Optional[Tensor] = None) -> Tensor:
        aux_loss = torch.tensor(0.0, device=x.device)

        # weights from router
        weights = self.router(x, is_warmup)
        usage = weights.mean(dim=(0, 1))
        #  print(usage.detach().cpu())
        aux_loss =  -torch.sum(usage * torch.log(usage + 1e-9))

        log_moe_stats(
            gate_probs=weights,
            layer_idx=self.layer_idx,
            step=global_step,
            log_interval=100,
            is_warmup=is_warmup
        )
        
        # expert outputs
        exp_out = torch.stack(
            [expert(x, cos, sin, is_causal, mask, input_pos) for expert in self.experts], dim=2)
        
        # adjust size to match expert outputs
        # weights = weights.unsqueeze(1).expand_as(exp_out)
        weights = weights.unsqueeze(-1)
        h = x + torch.sum(exp_out * weights, dim=2)
        out = h + self.feed_forward(self.ffn_norm(h))

        return out, aux_loss


class Router(nn.Module):
    def __init__(self, config: MixTransformerConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config

        # Layers
        self.layer1 = nn.Linear(config.dim, 128)
        self.dropout1 = nn.Dropout(config.dropout_rate)

        self.layer2 = nn.Linear(128, 256)
        self.leaky_relu1 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(config.dropout_rate)

        self.layer3 = nn.Linear(256, 128)
        self.leaky_relu2 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(config.dropout_rate)

        self.layer4 = nn.Linear(128, config.n_expert)

        if config.use_fused_ops:
            self.feed_forward = LigerSwiGLUMLP(config)
        else:
            self.feed_forward = LLaMAMLP(config)


    def forward(self, x: Tensor, is_warmup: bool) -> Tensor:
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)

        x = self.layer2(x)
        x = self.leaky_relu1(x)
        x = self.dropout2(x)

        x = self.layer3(x)
        x = self.leaky_relu2(x)
        x = self.dropout3(x)
        
        temperature = 2.0 if is_warmup else 1.0
        weights = torch.softmax(self.layer4(x) / temperature, dim=-1)
        return weights
   

if __name__ == "__main__":
    # test the model

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = MixTransformerConfig(
        block_size=2048,
        n_layer=12,

        n_head=12,
        dim=768/2,
        # use_fused_ops=True,
    )
    model = MixTransformer(config)
    model.to(device)
    model.setup_cache(device=device) # setup RoPE cache

    input_ids = torch.randint(0, config.vocab_size, (1, config.block_size), device=device)
    print("input_ids.shape:", input_ids.shape, input_ids.dtype)

    logits = model(input_ids)
    print("logits.shape:", logits.shape, logits.dtype)
