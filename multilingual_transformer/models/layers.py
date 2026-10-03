from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from multilingual_transformer.models.attention import MultiHeadAttention


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        # Non-persistent: recomputed, so checkpoints don't depend on max_len.
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.size(1), :]


class TokenEmbedding(nn.Module):
    """Embedding + positional encoding (same dynamics as the original model: no scaling, no extra dropout)."""

    def __init__(self, vocab_size: int, d_model: int, max_len: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len)

    def forward(self, x: Tensor) -> Tensor:
        return self.pos(self.emb(x))


# class FeedForward(nn.Module):
#     def __init__(self, d_model: int, d_ff: int) -> None:
#         super().__init__()
#         # Project to 2 * d_ff to create both the gate and the value simultaneously
#         self.fc1 = nn.Linear(d_model, d_ff * 2)
#         self.fc2 = nn.Linear(d_ff, d_model)

#     def forward(self, x: Tensor) -> Tensor:
#         # Split the projection in half along the last dimension
#         gate, value = self.fc1(x).chunk(2, dim=-1)
        
#         # SwiGLU: SiLU(gate) * value
#         return self.fc2(F.silu(gate) * value)


# class FeedForward(nn.Module):
#     def __init__(self, d_model: int, d_ff: int) -> None:
#         super().__init__()
#         self.fc1 = nn.Linear(d_model, d_ff)
#         self.fc2 = nn.Linear(d_ff, d_model)

#     def forward(self, x: Tensor) -> Tensor:
#         # Squared ReLU activation: max(0, x)^2
#         return self.fc2(F.relu(self.fc1(x)) ** 2)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        # SiLU (Swish) activation: x * sigmoid(x)
        return self.fc2(F.silu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = self.norm1(x + self.drop(self.self_attn(x, x, x, mask)))
        return self.norm2(x + self.drop(self.ffn(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        enc_out: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> Tensor:
        x = self.norm1(x + self.drop(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.drop(self.cross_attn(x, enc_out, enc_out, src_mask)))
        return self.norm3(x + self.drop(self.ffn(x)))
