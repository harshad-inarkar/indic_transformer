from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from multilingual_transformer.configs.config import AppConfig
from multilingual_transformer.data.dataset import DataPipeline
from multilingual_transformer.engine.decoder import TranslationGenerator
from multilingual_transformer.engine.evaluator import TranslationEvaluator
from multilingual_transformer.engine.loader import load_model, load_tokenizers
import random


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_config = script_dir.parent / "configs" / "transformer_config.toml"

    parser = argparse.ArgumentParser(description="Score a trained checkpoint (BLEU / chrF) on the cached test split")
    parser.add_argument("--config", type=str, default=str(default_config))
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Number of test sentences to score (default: training.bleu_sample from the config)",
    )
    parser.add_argument("--method", choices=["both", "greedy", "beam"], default="both")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument(
        "--no-repeat-ngram", type=int, default=None,
        help="Beam search n-gram repeat blocking (0 = off). Default: inference.no_repeat_ngram_size from the config",
    )
    parser.add_argument("--show-samples", action="store_true", help="Print a few predictions next to the references")
    args = parser.parse_args()

    cfg = AppConfig.from_toml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = cfg.checkpoint_path()

    # Guardrail: Ensure train.py was run first
    if not ckpt_path.exists():
        print(f"\n[Error] Checkpoint not found at: {ckpt_path}")
        print("Please run 'python -m multilingual_transformer.scripts.train' first (and check training.save_best).\n")
        sys.exit(1)

    pipeline = DataPipeline(
        dataset_name=cfg.language.dataset_name,
        src_lang=cfg.language.src_code,
        tgt_lang=cfg.language.tgt_lang,
        lang_pair=cfg.language.lang_pair,
        data_dir=cfg.data.data_dir,
        max_len=cfg.data.max_len,
        max_ratio=cfg.data.max_length_ratio,
        seed=cfg.project.seed,
    )
    try:
        te_src, te_tgt = pipeline.load_cached_split("test", cfg.data.test_size)
    except FileNotFoundError as err:
        print(f"\n[Error] {err}\n")
        sys.exit(1)

    print("Loading tokenizers and model... (this takes a few seconds)")
    tok_src, tok_tgt = load_tokenizers(cfg)
    model = load_model(cfg, tok_src, tok_tgt, device)
    generator = TranslationGenerator(
        model, tok_src, tok_tgt, cfg.data.max_len, device,
        no_repeat_ngram_size=cfg.inference.no_repeat_ngram_size,
    )
    if args.no_repeat_ngram is not None:
        generator.no_repeat_ngram_size = args.no_repeat_ngram
    evaluator = TranslationEvaluator(generator)

    sample_size = args.sample_size if args.sample_size is not None else cfg.training.bleu_sample
    n_eval = min(sample_size, len(te_src))
    gen_bs = cfg.training.gen_batch_size

    results: dict[str, tuple[float, float]] = {}
    samples: list[dict[str, str]] = []
    if args.method in ("both", "greedy"):
        bleu, chrf, samples = evaluator.evaluate(
            te_src, te_tgt, method="greedy", batch_size=gen_bs, sample_size=sample_size
        )
        results["Greedy Decoding"] = (bleu, chrf)
    if args.method in ("both", "beam"):
        bleu, chrf, samples = evaluator.evaluate(
            te_src, te_tgt, method="beam", beam_size=args.beam_size, batch_size=gen_bs, sample_size=sample_size
        )
        results["Beam Decoding"] = (bleu, chrf)

    print("\n" + "=" * 70)
    print("                    EVALUATION RESULTS")
    print("=" * 70)
    print(f"Target Language   : {cfg.language.target_lang} ({cfg.language.tgt_lang})")
    print(f"Checkpoint        : {ckpt_path}")
    print(f"Tokenizers Used   : SRC = {cfg.tokenizer.algo_src.upper()} | TGT = {cfg.tokenizer.algo_tgt.upper()}")
    print(f"Test Sentences    : {n_eval:,} (of {len(te_src):,} in the test split)")
    print(f"Device            : {device.type}")
    block = generator.no_repeat_ngram_size
    print(f"No-Repeat N-gram  : {block if block >= 2 else 'off'} (beam search)")
    print("-" * 70)
    for name, (bleu, chrf) in results.items():
        label = f"{name} (k={args.beam_size})" if name == "Beam Decoding" else name
        print(f"{label:<20}-> BLEU: {bleu:5.2f} | CHRF: {chrf:5.2f}")
    print("=" * 70 + "\n")

    if args.show_samples:
        k = min(10, len(te_src))
        idx = random.Random(cfg.project.seed).sample(range(len(te_src)), k)
        sents = [te_src[i] for i in idx]
        if args.method == "greedy":
            preds = generator.batched_greedy_decode(sents, batch_size=gen_bs)
        else:
            preds = generator.batched_beam_decode(sents, beam_size=args.beam_size, batch_size=gen_bs)

        print(f"=== Test Sample Translations ({cfg.language.lang_pair}) ===")
        for n, (i, pred) in enumerate(zip(idx, preds), 1):
            print(f"[{n}] EN   : {te_src[i]}")
            print(f"    Ref    : {te_tgt[i]}")
            print(f"    Pred   : {pred}\n")


if __name__ == "__main__":
    main()
