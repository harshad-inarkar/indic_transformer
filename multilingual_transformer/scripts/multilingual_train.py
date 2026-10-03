from __future__ import annotations

import argparse
from pathlib import Path
import tomllib
import warnings

import torch

from multilingual_transformer.configs.config import AppConfig
from multilingual_transformer.data.dataset import TranslationDataset
from multilingual_transformer.data.multilingual_dataset import MultilingualDataPipeline
from multilingual_transformer.engine.decoder import TranslationGenerator
from multilingual_transformer.engine.evaluator import TranslationEvaluator
from multilingual_transformer.engine.trainer import Trainer
from multilingual_transformer.models.transformer import MultilingualTransformer
from multilingual_transformer.utils.helpers import free_memory, set_seed
from multilingual_transformer.data.multilingual_tokenizer import MultilingualTokenizerManager


def main() -> None:
    assert torch.cuda.is_available(), "CUDA device required."
    
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir.parent / "configs" / "multilingual_config.toml"
    
    with open(config_path, "rb") as f:
        raw_cfg = tomllib.load(f)
        
    # Use existing AppConfig to feed standard variables to the Trainer
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = AppConfig.from_toml(config_path)

    set_seed(cfg.project.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    pipeline = MultilingualDataPipeline(
        dataset_map=raw_cfg["multilingual"]["dataset_map"],
        data_dir=cfg.data.data_dir,
        max_len=cfg.data.max_len,
        max_ratio=cfg.data.max_length_ratio,
        seed=cfg.project.seed,
    )

    corpus = pipeline.acquire_multilingual_corpus(
        pairs_per_lang=raw_cfg["multilingual"]["pairs_per_lang"],
        val_per_lang=raw_cfg["multilingual"]["val_size_per_lang"],
        test_per_lang=raw_cfg["multilingual"]["test_size_per_lang"],
    )

    print("Training shared multilingual tokenizer...")

    combined_texts = corpus["train_src"] + corpus["train_tgt"]
    shared_tok = MultilingualTokenizerManager.train(
        combined_texts, 
        cfg.tokenizer.algo_tgt, 
        cfg.tokenizer.max_vocab_size,
        lang_tokens=raw_cfg["multilingual"]["target_tokens"]
    )
    
    # Define path and create directory before saving
    cfg.data.tokenizer_dir.mkdir(parents=True, exist_ok=True)
    shared_tok_path = cfg.data.tokenizer_dir / f"multilingual_shared_{cfg.tokenizer.max_vocab_size}.json"
    
    # Save uses the inherited method from the original TokenizerManager
    MultilingualTokenizerManager.save(shared_tok, shared_tok_path)

    tr_ds = TranslationDataset(corpus["train_src"], corpus["train_tgt"], shared_tok, shared_tok, cfg.data.max_len)
    va_ds = TranslationDataset(corpus["val_src"], corpus["val_tgt"], shared_tok, shared_tok, cfg.data.max_len)
    tr_loader = pipeline.create_loader(tr_ds, cfg.training.batch_size, shuffle=True)
    val_loader = pipeline.create_loader(va_ds, cfg.training.batch_size, shuffle=False)
    free_memory()

    vocab_sz = shared_tok.get_vocab_size()
    model = MultilingualTransformer(
        src_vocab_size=vocab_sz, tgt_vocab_size=vocab_sz, max_len=cfg.data.max_len,
        d_model=cfg.model.d_model, num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads, d_ff=cfg.model.d_ff, dropout=cfg.model.dropout,
    )

    trainer = Trainer(model, cfg, tr_loader, val_loader, shared_tok, shared_tok)
    trainer.fit()
    trainer.restore()
    trainer.release()
    free_memory()

    generator = TranslationGenerator(
        model, shared_tok, shared_tok, cfg.data.max_len, torch.device("cuda"),
        no_repeat_ngram_size=cfg.inference.no_repeat_ngram_size,
    )
    evaluator = TranslationEvaluator(generator)

    print("\n" + "=" * 70)
    print("      MULTILINGUAL EVALUATION BENCHMARKS (EN <-> HI / MR / SA)")
    print("=" * 70)
    for pair_key in ["en-hi", "hi-en", "en-mr", "en-sa"]:
        split = corpus["eval_splits"].get(pair_key)
        if split:
            bleu, chrf, _ = evaluator.evaluate(
                split["test_src"], split["test_tgt"], method="beam", beam_size=5,
                batch_size=cfg.training.gen_batch_size, sample_size=cfg.training.bleu_sample,
            )
            print(f"Direction {pair_key.upper():<7} -> Beam BLEU: {bleu:5.2f} | Beam CHRF: {chrf:5.2f}")
    print("=" * 70)

if __name__ == "__main__":
    main()