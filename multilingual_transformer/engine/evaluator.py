from __future__ import annotations

import unicodedata
from typing import TYPE_CHECKING

from sacrebleu.metrics import BLEU, CHRF

if TYPE_CHECKING:  # avoids importing torch just to score text
    from multilingual_transformer.engine.decoder import TranslationGenerator


class TranslationEvaluator:
    """Corpus-level BLEU and chrF via sacreBLEU, the de-facto reference implementation.

    Metric choices (all reported in `self.signatures` after each `evaluate` call, so scores are reproducible):
      * BLEU   - corpus-level, sacreBLEU default smoothing ("exp"), tokenize="intl". The default "13a"
                 tokenizer only splits ASCII punctuation, so the Hindi danda "।" stays glued to the
                 previous word ("है।" != "है ।"). "intl" splits Unicode punctuation/symbols and keeps
                 Indic combining marks (matras, viramas) inside words.
      * chrF   - corpus-level character n-grams (sacreBLEU default: char order 6, beta 2). Set
                 `chrf_word_order=2` for chrF++ (adds word uni/bi-grams).
      * lowercase=True - the model is trained on lowercased text, so scoring is case-insensitive.
                 Pass lowercase=False for case-sensitive scoring.
      * Unicode NFC applied to hypotheses and references only. This merges canonically-equivalent
        encodings (e.g. a precomposed nukta letter vs base letter + nukta); no other rewriting is done.
        Notably there is NO mapping of digit scripts, danda variants, etc.: a hypothesis that writes
        "५०" where the reference has "50" is a genuine surface mismatch and is scored as one.
    """

    def __init__(
        self,
        generator: TranslationGenerator,
        bleu_tokenize: str = "intl",
        chrf_word_order: int = 0,
        lowercase: bool = True,
        normalize_unicode: bool = True,
    ) -> None:
        self.generator = generator
        self.bleu_tokenize = bleu_tokenize
        self.chrf_word_order = chrf_word_order
        self.lowercase = lowercase
        self.normalize_unicode = normalize_unicode
        self.signatures: dict[str, str] = {}

    def _prep(self, text: str) -> str:
        return unicodedata.normalize("NFC", text) if self.normalize_unicode else text

    def evaluate(
        self,
        sources: list[str],
        references: list[str],
        method: str = "beam",
        beam_size: int = 5,
        batch_size: int = 128,
        sample_size: int = 500,
    ) -> tuple[float, float, list[dict[str, str]]]:
        n = min(sample_size, len(sources))
        src_subset, ref_subset = sources[:n], references[:n]

        if method == "beam":
            preds = self.generator.batched_beam_decode(src_subset, beam_size=beam_size, batch_size=batch_size)
        else:
            preds = self.generator.batched_greedy_decode(src_subset, batch_size=batch_size)

        hyps = [self._prep(p) for p in preds]
        refs = [self._prep(r) for r in ref_subset]

        # Fresh metric objects per call: sacreBLEU metrics keep per-evaluation state for the signature.
        bleu_metric = BLEU(tokenize=self.bleu_tokenize, lowercase=self.lowercase)
        chrf_metric = CHRF(word_order=self.chrf_word_order, lowercase=self.lowercase)
        bleu = bleu_metric.corpus_score(hyps, [refs]).score
        chrf = chrf_metric.corpus_score(hyps, [refs]).score
        self.signatures = {
            "bleu": str(bleu_metric.get_signature()),
            "chrf": str(chrf_metric.get_signature()),
        }

        samples = [{"src": src_subset[i], "ref": ref_subset[i], "pred": preds[i]} for i in range(min(5, n))]
        return float(bleu), float(chrf), samples