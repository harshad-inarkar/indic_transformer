from __future__ import annotations

from typing import Any
import torch
import torch.nn.functional as F
from torch import Tensor
from multilingual_transformer.data.dataset import numericalize, pad_sequence
from multilingual_transformer.models.transformer import MultilingualTransformer, make_src_mask, make_tgt_mask


class TranslationGenerator:
    def __init__(
        self,
        model: MultilingualTransformer,
        src_tokenizer: Any,
        tgt_tokenizer: Any,
        max_len: int,
        device: torch.device,
    ) -> None:

        self.src_tok = src_tokenizer
        self.tgt_tok = tgt_tokenizer
        self.max_len = max_len
        self.device = device
        self.pad_src = src_tokenizer.token_to_id("<pad>")
        self.pad_tgt = tgt_tokenizer.token_to_id("<pad>")
        self.sos_tgt = tgt_tokenizer.token_to_id("<sos>")
        self.eos_tgt = tgt_tokenizer.token_to_id("<eos>")


        # Enable Multi-GPU Inference (specifically accelerates batched_beam_decode)
        if torch.cuda.device_count() > 1 and device.type == "cuda":
            self.model = nn.DataParallel(model).to(self.device)
        else:
            self.model = model.to(self.device)


    def _to_text(self, ids: list[int]) -> str:
        return self.tgt_tok.decode(ids, skip_special_tokens=True)

    @torch.inference_mode()
    def greedy_decode(self, sentence: str) -> str:
        self.model.eval()
        src_ids = numericalize(sentence, self.src_tok)
        src = pad_sequence(src_ids, self.max_len, self.pad_src).unsqueeze(0).to(self.device)
        src_mask = make_src_mask(src, self.pad_src)

        tgt_ids = [self.sos_tgt]
        for _ in range(self.max_len - 1):
            tgt = torch.tensor([tgt_ids], device=self.device)
            tgt_mask = make_tgt_mask(tgt, self.pad_tgt)
            with torch.amp.autocast(self.device.type, dtype=torch.float16):
                logits = self.model(src, tgt, src_mask, tgt_mask)
            next_token = int(logits[0, -1].argmax().item())
            tgt_ids.append(next_token)
            if next_token == self.eos_tgt:
                break
        return self._to_text(tgt_ids)

    @torch.inference_mode()
    def batched_greedy_decode(self, sentences: list[str], batch_size: int = 128) -> list[str]:
        self.model.eval()
        all_decoded: list[str] = []
        for i in range(0, len(sentences), batch_size):
            chunk = sentences[i : i + batch_size]
            b_size = len(chunk)
            src_ids = [numericalize(t, self.src_tok) for t in chunk]
            src = torch.stack([pad_sequence(s, self.max_len, self.pad_src) for s in src_ids]).to(self.device)
            src_mask = make_src_mask(src, self.pad_src)

            tgt = torch.full((b_size, 1), self.sos_tgt, dtype=torch.long, device=self.device)
            unfinished = torch.ones(b_size, dtype=torch.bool, device=self.device)

            for _ in range(self.max_len - 1):
                tgt_mask = make_tgt_mask(tgt, self.pad_tgt)
                with torch.amp.autocast(self.device.type, dtype=torch.float16):
                    logits = self.model(src, tgt, src_mask, tgt_mask)
                next_tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                tgt = torch.cat([tgt, next_tokens], dim=1)
                unfinished = unfinished & (next_tokens.squeeze(1) != self.eos_tgt)
                if not unfinished.any():
                    break

            for row in tgt.tolist():
                all_decoded.append(self._to_text(row))
        return all_decoded

    @torch.inference_mode()
    def batched_beam_decode(self, sentences: list[str], beam_size: int = 5, batch_size: int = 128) -> list[str]:
        self.model.eval()
        vocab_size = self.tgt_tok.get_vocab_size()
        all_decoded: list[str] = []

        for i in range(0, len(sentences), batch_size):
            chunk = sentences[i : i + batch_size]
            b = len(chunk)
            src_ids = [numericalize(t, self.src_tok) for t in chunk]
            src = torch.stack([pad_sequence(s, self.max_len, self.pad_src) for s in src_ids]).to(self.device)
            src_mask = make_src_mask(src, self.pad_src)

            with torch.amp.autocast(self.device.type, dtype=torch.float16):
                enc_out = self.model.encoder(src, src_mask)

            enc_out = enc_out.repeat_interleave(beam_size, dim=0)
            src_mask = src_mask.repeat_interleave(beam_size, dim=0)
            tgt = torch.full((b * beam_size, 1), self.sos_tgt, dtype=torch.long, device=self.device)

            scores = torch.full((b, beam_size), float("-inf"), device=self.device)
            scores[:, 0] = 0.0
            finished = torch.zeros((b, beam_size), dtype=torch.bool, device=self.device)

            for _ in range(self.max_len - 1):
                tgt_mask = make_tgt_mask(tgt, self.pad_tgt)
                with torch.amp.autocast(self.device.type, dtype=torch.float16):
                    logits = self.model.decoder(tgt, enc_out, src_mask, tgt_mask)

                log_probs = F.log_softmax(logits[:, -1, :], dim=-1).view(b, beam_size, vocab_size)
                for bi in range(b):
                    for bm in range(beam_size):
                        if finished[bi, bm]:
                            log_probs[bi, bm, :] = float("-inf")
                            log_probs[bi, bm, self.eos_tgt] = 0.0

                next_scores = (scores.unsqueeze(-1) + log_probs).view(b, beam_size * vocab_size)
                top_scores, top_indices = torch.topk(next_scores, beam_size, dim=1)
                scores = top_scores

                beam_idx = top_indices // vocab_size
                token_idx = top_indices % vocab_size

                curr_len = tgt.size(1)
                tgt_reshaped = tgt.view(b, beam_size, curr_len)
                new_tgt = torch.zeros((b, beam_size, curr_len + 1), dtype=torch.long, device=self.device)
                new_finished = torch.zeros((b, beam_size), dtype=torch.bool, device=self.device)

                for bi in range(b):
                    for bm in range(beam_size):
                        p_beam = beam_idx[bi, bm]
                        new_tgt[bi, bm, :curr_len] = tgt_reshaped[bi, p_beam]
                        new_tgt[bi, bm, curr_len] = token_idx[bi, bm]
                        new_finished[bi, bm] = finished[bi, p_beam] | (token_idx[bi, bm] == self.eos_tgt)

                tgt = new_tgt.view(b * beam_size, curr_len + 1)
                finished = new_finished
                if finished.all():
                    break

            tgt_results = tgt.view(b, beam_size, -1)
            for bi in range(b):
                all_decoded.append(self._to_text(tgt_results[bi, 0].tolist()))
        return all_decoded