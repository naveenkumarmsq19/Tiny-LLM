"""Small decoder-only Transformer using causal self-attention."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class TinyConfig:
    vocab_size: int
    context_size: int = 64
    embed_dim: int = 128
    heads: int = 4
    layers: int = 2
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.vocab_size < 1 or self.context_size < 1:
            raise ValueError("vocab_size and context_size must be positive.")
        if self.embed_dim < 1 or self.heads < 1 or self.layers < 1:
            raise ValueError("embed_dim, heads, and layers must be positive.")
        if self.embed_dim % self.heads != 0:
            raise ValueError("embed_dim must be divisible by heads.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")


class TransformerBlock(nn.Module):
    def __init__(self, config: TinyConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.attn = nn.MultiheadAttention(
            config.embed_dim,
            config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            nn.GELU(),
            nn.Linear(4 * config.embed_dim, config.embed_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        normalized = self.ln1(x)
        attended, _ = self.attn(
            normalized, normalized, normalized,
            attn_mask=mask,
            need_weights=False,
        )
        x = x + attended
        return x + self.ffn(self.ln2(x))


class TinyLLM(nn.Module):
    def __init__(self, config: TinyConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.context_size, config.embed_dim)
        self.blocks = nn.Sequential(
            *(TransformerBlock(config) for _ in range(config.layers))
        )
        self.final_norm = nn.LayerNorm(config.embed_dim)
        self.output = nn.Linear(config.embed_dim, config.vocab_size)

    def forward(
        self, tokens: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len = tokens.shape
        if seq_len == 0 or seq_len > self.config.context_size:
            raise ValueError("Sequence length must be between 1 and context_size.")
        positions = torch.arange(seq_len, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        logits = self.output(self.final_norm(self.blocks(x)))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(batch_size * seq_len, -1),
                targets.reshape(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative.")
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError("Provide at least one initial token per sequence.")

        for _ in range(max_new_tokens):
            context = tokens[:, -self.config.context_size:]
            logits, _ = self(context)
            next_logits = logits[:, -1, :] / temperature
            if top_k is not None and top_k > 0:
                cutoff = torch.topk(
                    next_logits, min(top_k, next_logits.size(-1))
                ).values[:, -1, None]
                next_logits = next_logits.masked_fill(
                    next_logits < cutoff, float("-inf")
                )
            probabilities = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            tokens = torch.cat((tokens, next_token), dim=1)
        return tokens
