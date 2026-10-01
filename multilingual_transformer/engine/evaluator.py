from __future__ import annotations

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
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
        src_subset, ref_subset = sources[:n], references[:n]

        if method == "beam":
            preds = self.generator.batched_beam_decode(src_subset, beam_size=beam_size, batch_size=batch_size)
        else:
            preds = self.generator.batched_greedy_decode(src_subset, batch_size=batch_size)

        # Corpus-level BLEU (standard definition), smoothed for short test sets.
        bleu = corpus_bleu(
            [[r.split()] for r in ref_subset],
            [p.split() for p in preds],
            smoothing_function=SmoothingFunction().method1,
        )
        chrf = float(np.mean([sentence_chrf(r, p) if p.strip() else 0.0 for r, p in zip(ref_subset, preds)]))

        samples = [{"src": src_subset[i], "ref": ref_subset[i], "pred": preds[i]} for i in range(min(5, n))]
        return float(bleu * 100), chrf * 100, samples
