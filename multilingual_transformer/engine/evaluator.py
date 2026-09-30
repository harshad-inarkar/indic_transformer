from __future__ import annotations

import time
import numpy as np
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.chrf_score import sentence_chrf
from multilingual_transformer.engine.decoder import TranslationGenerator


class TranslationEvaluator:
    def __init__(self, generator: TranslationGenerator) -> None:
        self.generator = generator

    def evaluate(
        self,
        sources: list[str],
        references: list[str],
        method: str = "beam",
        beam_size: int = 5,
        batch_size: int = 128,
        sample_size: int = 300,
    ) -> tuple[float, float, list[dict[str, str]]]:
        n = min(sample_size, len(sources))
        src_subset = sources[:n]
        ref_subset = references[:n]

        if method == "beam":
            preds = self.generator.batched_beam_decode(src_subset, beam_size=beam_size, batch_size=batch_size)
        else:
            preds = self.generator.batched_greedy_decode(src_subset, batch_size=batch_size)

        smooth = SmoothingFunction().method1
        bleu_scores = [
            sentence_bleu([ref.split()], pred.split(), smoothing_function=smooth)
            for pred, ref in zip(preds, ref_subset)
        ]
        chrf_scores = [sentence_chrf(ref, pred) for pred, ref in zip(preds, ref_subset)]

        samples = [
            {"src": src_subset[i], "ref": ref_subset[i], "pred": preds[i]}
            for i in range(min(5, n))
        ]
        return float(np.mean(bleu_scores) * 100), float(np.mean(chrf_scores) * 100), samples