from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn


class MultiHeadAttention(nn.Module):
    """Multi-head attention using fused SDPA (flash / mem-efficient kernels).

    `mask` is boolean and broadcastable to (B, H, T_q, T_k); True = may attend.
    """

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
        return x.view(x.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
        q_s = self._split_heads(self.w_q(q))
        k_s = self._split_heads(self.w_k(k))
        v_s = self._split_heads(self.w_v(v))
        out = F.scaled_dot_product_attention(q_s, k_s, v_s, attn_mask=mask)
        out = out.transpose(1, 2).reshape(q.size(0), -1, self.d_model)
        return self.w_o(out)
