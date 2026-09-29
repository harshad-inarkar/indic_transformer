from __future__ import annotations

import math
from typing import cast
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        fill_val = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(mask == 0, fill_val)
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.size()
        return x.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
        q_s = self._split_heads(self.w_q(q))
        k_s = self._split_heads(self.w_k(k))
        v_s = self._split_heads(self.w_v(v))

        out, _ = scaled_dot_product_attention(q_s, k_s, v_s, mask)
        out = out.transpose(1, 2).contiguous().view(q.size(0), -1, self.d_model)
        return cast(Tensor, self.w_o(out))