from __future__ import annotations

import unicodedata
from typing import TYPE_CHECKING

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.chrf_score import sentence_chrf

if TYPE_CHECKING:  # avoids importing torch just to score text
    from multilingual_transformer.engine.decoder import TranslationGenerator


def tokenize_for_metrics(text: str) -> list[str]:
    """Script-aware tokenisation used for BLEU and chrF.

    - NFKC-normalises so equivalent Unicode forms (e.g. precomposed vs decomposed nukta letters) match.
    - Drops invisible format characters (ZWJ/ZWNJ/BOM), which models and references use inconsistently.
    - Splits punctuation/symbols (Unicode categories P* and S*, which includes the Hindi danda "।")
      into separate tokens, so "है।" and "है ।" score identically.
    - Never splits inside a word: Indic vowel signs and viramas are combining marks (M*), not punctuation.
    """
    out: list[str] = []
    for ch in unicodedata.normalize("NFKC", text):
        cat = unicodedata.category(ch)
        if cat == "Cf":
            continue
        out.append(f" {ch} " if cat[0] in "PS" else ch)
    return "".join(out).split()


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

        ref_toks = [tokenize_for_metrics(r) for r in ref_subset]
        hyp_toks = [tokenize_for_metrics(p) for p in preds]

        # Corpus-level BLEU (standard definition) on script-aware tokens, smoothed for small test sets.
        bleu = corpus_bleu(
            [[r] for r in ref_toks],
            hyp_toks,
            smoothing_function=SmoothingFunction().method1,
        )
        # chrF (character n-grams, whitespace ignored) on the same normalised text; mean of sentence scores.
        chrf = float(
            np.mean(
                [
                    sentence_chrf(" ".join(r), " ".join(h)) if h else 0.0
                    for r, h in zip(ref_toks, hyp_toks)
                ]
            )
        )

        samples = [{"src": src_subset[i], "ref": ref_subset[i], "pred": preds[i]} for i in range(min(5, n))]
        return float(bleu * 100), chrf * 100, samples
