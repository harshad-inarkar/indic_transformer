from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.chrf_score import sentence_chrf

if TYPE_CHECKING:
    from multilingual_transformer.engine.decoder import TranslationGenerator

# Indic numerals (Devanagari, Bengali, Gujarati, Odia, Gurmukhi, Tamil, Telugu, Kannada, Malayalam)
INDIC_DIGITS = (
    "०१२३४५६७८९"  # Devanagari / Hindi / Marathi / Sanskrit
    "০১২৩৪৫৬৭৮৯"  # Bengali / Assamese
    "૦૧૨૩૪૫૬૭૮૯"  # Gujarati
    "୦୧୨୩୪୫୬୭୮୯"  # Odia
    "੦੧੨੩੪੫੬੭੮੯"  # Gurmukhi / Punjabi
    "௦௧௨௩௪௫௬௭௮௯"  # Tamil
    "౦౧౨౩౪౫౬౭౮౯"  # Telugu
    "೦೧೨೩೪೫೬೭೮೯"  # Kannada
    "൦൧൨൩൪൫൬൭൮൯"  # Malayalam
)
ARABIC_DIGITS = "0123456789" * 9
DIGIT_TRANSLATION_TABLE = str.maketrans(INDIC_DIGITS, ARABIC_DIGITS)


def normalize_indic_text(text: str) -> str:
    """Canonical text pre-processing applied equally to hypotheses and references.
    
    1. Lowercases Latin script tokens (English loanwords/acronyms).
    2. NFKC-normalises decomposed/precomposed glyphs and nuktas.
    3. Normalises pipe '|' and double pipes '||' to Devanagari danda '।' and double danda '॥'.
    4. Canonicalises Indic script digits to standard Arabic numerals (0-9).
    5. Strips redundant whitespace and removes control formatting characters (ZWJ, ZWNJ, BOM).
    """
    if not text:
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Canonical Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # 3. Replace ASCII pipes with authentic Indic danda / double danda
    text = text.replace("||", "॥").replace("|", "।")

    # 4. Canonicalize numerals
    text = text.translate(DIGIT_TRANSLATION_TABLE)

    # 5. Remove zero-width & formatting control chars (Unicode category 'Cf')
    cleaned = [ch for ch in text if unicodedata.category(ch) != "Cf"]
    text = "".join(cleaned)

    # 6. Normalize whitespace
    return re.sub(r"\s+", " ", text).strip()


def tokenize_for_metrics(text: str) -> list[str]:
    """Script-aware tokenisation for BLEU and chrF over normalized text.

    - Splits punctuation and symbols (categories P* and S*, including danda '।') into distinct tokens.
    - Preserves combining marks (vowel signs/matras and viramas - category M*).
    """
    norm_text = normalize_indic_text(text)
    out: list[str] = []
    for ch in norm_text:
        cat = unicodedata.category(ch)
        # Pad punctuation/symbols with spaces so punctuation splits cleanly without fragmenting words
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

        # Both reference and hypothesis pass through identical normalization and tokenization
        ref_toks = [tokenize_for_metrics(r) for r in ref_subset]
        hyp_toks = [tokenize_for_metrics(p) for p in preds]

        # Corpus-level BLEU on script-aware tokens with smoothing
        bleu = corpus_bleu(
            [[r] for r in ref_toks],
            hyp_toks,
            smoothing_function=SmoothingFunction().method1,
        )

        # chrF on normalized strings (whitespace ignored)
        chrf = float(
            np.mean(
                [
                    sentence_chrf(" ".join(r), " ".join(h)) if h else 0.0
                    for r, h in zip(ref_toks, hyp_toks)
                ]
            )
        )

        samples = [
            {"src": src_subset[i], "ref": ref_subset[i], "pred": preds[i]}
            for i in range(min(5, n))
        ]
        return float(bleu * 100), chrf * 100, samples