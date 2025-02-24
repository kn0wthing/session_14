import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Local Imports
from deepseek_config import DeepSeekConfig, LatentAttentionConfig

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5): # the number of features/dimensions/embeddings in the input, eps is a small number to prevent division by zero
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size)) # weight is a learnable parameter that scales the input
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).sqrt() + self.eps # compute the norm of the input
        return x / norm * self.weight # normalize the input by dividing by the norm and scale it by the weight parameter

class LlamaRotaryEmbedding(nn.Module):
    """Modified RoPE implementation for latent attention"""
    def __init__(self, dim, config: LatentAttentionConfig):
        super().__init__()
        self.dim = dim
        self.base = config.base
        self.scaling_factor = config.scaling_factor
        
        # Rotary embeddings with scaling
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, seq_len, device):
        # Create position embeddings
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        
        # Create rotation matrices [seq_len, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # [1, seq_len, 1, dim]
        cos = emb.cos().view(1, seq_len, 1, self.dim)
        sin = emb.sin().view(1, seq_len, 1, self.dim)
        
        return cos, sin

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def apply_rotary_pos_emb(self, x, cos, sin):
        # Expand cos/sin to match batch size and heads
        cos = cos.expand(x.shape[0], -1, x.shape[2], -1)
        sin = sin.expand(x.shape[0], -1, x.shape[2], -1)
        
        return (x * cos) + (self.rotate_half(x) * sin)


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.hidden_size = config.hidden_size # also dim
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.compression_ratio = config.compression_ratio
        self.latent_dim = self.hidden_size // self.compression_ratio

        # Projection layers with exact dimension matching
        self.kv_proj_d = nn.Linear(self.hidden_size, self.latent_dim, bias=False)
        self.q_proj_d = nn.Linear(self.hidden_size, self.latent_dim, bias=False)
        
        # Decompression projections must output exactly self.head_dim // 2 for k and q
        half_head_dim = self.head_dim // 2
        self.k_proj_u = nn.Linear(self.latent_dim, self.num_heads * half_head_dim, bias=False)
        self.q_proj_u = nn.Linear(self.latent_dim, self.num_heads * half_head_dim, bias=False)
        self.v_proj_u = nn.Linear(self.latent_dim, self.hidden_size, bias=False)

        # Rotary embeddings
        self.rope_k = nn.Linear(self.hidden_size, self.hidden_size // 2, bias=False)
        self.rope_q = nn.Linear(self.latent_dim, self.hidden_size // 2, bias=False)

        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.rotary_emb = LlamaRotaryEmbedding(dim=half_head_dim, config=LatentAttentionConfig())
        # self.rotary_emb = LatentAttention(self.head_dim, self.head_dim, config.latent_attention_config)

    def forward(self, x, attn_mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Project to latent space
        kv_latent = self.kv_proj_d(x) # [batch_size, seq_len, latent_dim]
        q_latent = self.q_proj_d(x) # [batch_size, seq_len, latent_dim]
        
        half_head_dim = self.head_dim // 2 
        
        k = self.k_proj_u(kv_latent) # [batch_size, seq_len, hidden_size // 2]
        q = self.q_proj_u(q_latent) # [batch_size, seq_len, hidden_size // 2]
        v = self.v_proj_u(kv_latent) # [batch_size, seq_len, hidden_size]
        
        q_rope = self.rope_q(q_latent) # [batch_size, seq_len, hidden_size // 2]
        k_rope = self.rope_k(x) # [batch_size, seq_len, hidden_size // 2]

        q_proj_2 = q.view(batch_size, seq_len, self.num_heads, half_head_dim) # [batch_size, seq_len, num_heads, half_head_dim]
        k_proj_2 = k.view(batch_size, seq_len, self.num_heads, half_head_dim) # [batch_size, seq_len, num_heads, half_head_dim]
        q_rope_2= q_rope.view(batch_size, seq_len, self.num_heads, half_head_dim) # [batch_size, seq_len, num_heads, half_head_dim]
        k_rope_2 = k_rope.view(batch_size, seq_len, self.num_heads, half_head_dim) # [batch_size, seq_len, num_heads, half_head_dim]
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim) # [batch_size, seq_len, num_heads, head_dim]

        rotatory_emd_cos, rotatory_emd_sin = self.rotary_emb(seq_len, x.device)
        k_rope_2 = self.rotary_emb.apply_rotary_pos_emb(k_rope_2, rotatory_emd_cos, rotatory_emd_sin) # [batch_size, seq_len, num_heads, half_head_dim]
        q_rope_2 = self.rotary_emb.apply_rotary_pos_emb(q_rope_2, rotatory_emd_cos, rotatory_emd_sin) # [batch_size, seq_len, num_heads, half_head_dim]

        # Concatenate KV vectors with LV_RoPE vectors
        k = torch.cat([k_proj_2, k_rope_2], dim=-1) # [batch_size, seq_len, num_heads, head_dim]
        q = torch.cat([q_proj_2, q_rope_2], dim=-1) # [batch_size, seq_len, num_heads, head_dim]

        # Reshape to match the expected input for scaled_dot_product_attention
        k = k.transpose(1, 2) # [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2) # [batch_size, num_heads, seq_len, head_dim]
        v = v.transpose(1, 2) # [batch_size, num_heads, seq_len, head_dim]

        # Scaled dot product attention
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=True) # [batch_size, num_heads, seq_len, head_dim
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size) # [batch_size, seq_len, hidden_size]
        return self.o_proj(y) # [batch_size, seq_len, hidden_size]


class DeepSeekExpert(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class DeepSeekMoE(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.num_shared_experts = config.num_shared_experts
        self.num_routed_experts = self.num_experts - self.num_shared_experts
        self.top_k = config.top_k
        # self.intermediate_size = intermediate_size 
        self.intermediate_size = int(config.hidden_size * config.mlp_ratio)

        # Shared Experts
        self.shared_experts = nn.ModuleList([
            DeepSeekExpert(self.hidden_size, self.intermediate_size) 
            for _ in range(self.num_shared_experts)
        ])

        # Routed Experts
        self.routed_experts = nn.ModuleList([
            DeepSeekExpert(self.hidden_size, self.intermediate_size)
            for _ in range(self.num_routed_experts)
        ])
        
        # Routing
        self.router = nn.Linear(self.hidden_size, self.num_routed_experts, bias=False)
        self.routing_bias = nn.Parameter(torch.zeros(self.num_routed_experts))
        self.expert_load = None

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape

        # Shared experts
        shared_out = sum(expert(x) for expert in self.shared_experts) / self.num_shared_experts
        if self.num_shared_experts > 1:
            shared_out = shared_out / self.num_shared_experts
        
        # Calculate routing probabilities (EXACT reference implementation)
        routing_logits = self.router(x) + self.routing_bias

        # Get the top k experts per token
        routing_probs = torch.sigmoid(routing_logits)
        scores, indices = torch.topk(routing_probs, self.top_k, dim=-1)

        # Normalize the top k scores
        scores = scores / scores.sum(dim=-1, keepdim=True)

        # Process through the selected experts
        combined_out = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_indices = indices[..., k]
            expert_scores = scores[..., k:k+1]

            # process each expert
            for expert_idx in range(self.num_routed_experts):
                mask = (expert_indices == expert_idx)
                if mask.any():
                    expert_in = x[mask]
                    expert_out = self.routed_experts[expert_idx](expert_in)
                    combined_out[mask] += expert_out * expert_scores[mask]
        
        # Add the shared experts
        final_out = shared_out + combined_out
        return final_out

    def update_bias_terms(self, expert_load):
        """EXACT reference implementation of bias update"""
        target_load = 1.0 / self.num_routed_experts
        load_diff = expert_load - target_load
        update_rate = 0.1 * torch.abs(load_diff)
        with torch.no_grad():
            self.routing_bias.data -= update_rate * load_diff


class LlamaMLP(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.moe = DeepSeekMoE(config)
        
    def forward(self, x):
        return self.moe(x)


class DeepSeekBlock(nn.Module):
    """
    Transformer block
    """
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.self_attn = MultiHeadLatentAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size)
    
    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class LlamaModel(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            DeepSeekBlock(config) for _ in range(config.n_layer)
        ])
        self.norm = LlamaRMSNorm(config.hidden_size)

    def forward(self, x):
        x = self.embed_tokens(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class LlamaForCausalLM(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        
        # Enable memory efficient attention if available
        if hasattr(F, 'scaled_dot_product_attention'):
            self.use_flash_attention = True
        else:
            self.use_flash_attention = False

        # Share weights between embedding and lm_head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Tie weights
        self.lm_head.weight = self.model.embed_tokens.weight
        
        # Weight initialization
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.model(x)
        return self.lm_head(x)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.04)

def get_model(config: DeepSeekConfig):
    return LlamaForCausalLM(config)

if __name__ == "__main__":
    config = DeepSeekConfig()  # Create default config
    model = get_model(config)  # Pass config to get_model
    print(model)
# class DeepSeekLM(nn.Module):
#     """
#     DeepSeekLM model
#     """
#     def __init__(self, config=DeepSeekConfig()):
#         super().__init__()
#         self.config = config
        
#         self.transformer = nn.ModuleDict(dict(
#             wte = nn.Embedding(config.vocab_size, config.hidden_size),
#             h = nn.ModuleList([DeepSeekBlock(config) for _ in range(config.n_layer)]),
#             ln_f = nn.LayerNorm(config.hidden_size, bias=False),
#         ))
        
#         self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

#         # weight sharing
#         self.transformer.wte.weight = self.lm_head.weight

#         # weight initialization
#         self.apply(self._init_weights)
        
#         # Enable memory efficient attention if available
#         if hasattr(F, 'scaled_dot_product_attention'):
#             self.use_flash_attention = True
#         else:
#             self.use_flash_attention = False

#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             std = 0.02
#             if hasattr(module, 'NANGPT_SCALE_INIT'):
#                 std *= (2 * self.config.n_layer) ** -0.5
#             torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
#             if module.bias is not None:
#                 torch.nn.init.zeros_(module.bias)
#         elif isinstance(module, nn.Embedding):
#             torch.nn.init.normal_(module.weight, mean=0.0, std = 0.04)

#     def forward(self, idx, targets=None):
#         # idx is of shape (B, T)
#         B, T = idx.size()
#         assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
#         # forward the token and posisition embeddings
#         tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, hidden_size)
#         x = tok_emb
#         # forward the blocks of the transformer
#         for block in self.transformer.h:
#             x = block(x)
#         # forward the final layernorm and the classifier
#         x = self.transformer.ln_f(x)
#         logits = self.lm_head(x) # (B, T, vocab_size)
#         loss = None
#         if targets is not None:
#             loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
#         return logits, loss