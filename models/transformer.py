from __future__ import annotations

from typing import cast
import torch
from torch import Tensor, nn
from indic_transformer.models.layers import DecoderLayer, EncoderLayer, PositionalEncoding


def make_src_mask(src: Tensor, pad_idx: int) -> Tensor:
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: Tensor, pad_idx: int) -> Tensor:
    seq_len = tgt.size(1)
    pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=tgt.device)).bool()
    return pad_mask & causal_mask


class TransformerEncoder(nn.Module):
    def __init__(
        self, vocab_size: int, max_len: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, dropout: float
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = self.pos(self.emb(x))
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self, vocab_size: int, max_len: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, dropout: float
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(
        self, x: Tensor, enc_out: Tensor, src_mask: Tensor | None = None, tgt_mask: Tensor | None = None
    ) -> Tensor:
        x = self.pos(self.emb(x))
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return cast(Tensor, self.fc_out(x))


class MultilingualTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        max_len: int = 80,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert torch.cuda.is_available(), "CUDA is required for this pipeline"
        self.encoder = TransformerEncoder(src_vocab_size, max_len, d_model, num_layers, num_heads, d_ff, dropout)
        self.decoder = TransformerDecoder(tgt_vocab_size, max_len, d_model, num_layers, num_heads, d_ff, dropout)

    def forward(
        self, src: Tensor, tgt: Tensor, src_mask: Tensor | None = None, tgt_mask: Tensor | None = None
    ) -> Tensor:
        enc_out = self.encoder(src, src_mask)
        return cast(Tensor, self.decoder(tgt, enc_out, src_mask, tgt_mask))