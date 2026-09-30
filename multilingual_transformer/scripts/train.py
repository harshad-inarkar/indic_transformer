from __future__ import annotations

import argparse
import sys
import torch
from multilingual_transformer.configs.config import AppConfig
from multilingual_transformer.data.dataset import DataPipeline, TranslationDataset
from multilingual_transformer.data.tokenizer import TokenizerManager
from multilingual_transformer.engine.decoder import TranslationGenerator
from multilingual_transformer.engine.evaluator import TranslationEvaluator
from multilingual_transformer.engine.trainer import Trainer
from multilingual_transformer.models.transformer import MultilingualTransformer
from pathlib import Path
from multilingual_transformer.utils.helpers import set_seed, free_memory


def print_stats_table(
    config: AppConfig,
    n_params: int,
    train_loader_len: int,
    total_time: float,
    peak_mem: float,
    peak_res: float,
    bleu_greedy: float,
    chrf_greedy: float,
    bleu_beam: float,
    chrf_beam: float,
    src_vocab_sz: int,
    tgt_vocab_sz: int,
) -> None:
    print("\n" + "=" * 70)
    print("              CONSOLIDATED FINAL STATISTICS")
    print("=" * 70)
    print(f"Target Language   : {config.language.target_language} ({config.language.tgt_lang})")
    print(f"Dataset Source    : {config.language.dataset_name}")
    print(f"Tokenizers Used   : SRC = {config.tokenizer.algo_src.upper()} | TGT = {config.tokenizer.algo_tgt.upper()}")
    print(f"Vocab Sizes       : SRC = {src_vocab_sz:,} | TGT = {tgt_vocab_sz:,}")
    print(f"Model Parameters  : {n_params:,} ({n_params * 4 / 1024**2:.1f} MB fp32)")
    print("-" * 70)
    print(f"Epochs            : {config.training.epochs}")
    print(f"Train / Val Size  : {config.data.train_size:,} / {config.data.test_size:,}")
    print(f"Batch Size        : {config.training.batch_size} (Train) | {config.training.gen_batch_size} (Eval)")
    print(f"Steps per Epoch   : {train_loader_len:,}")
    print(f"Total Train Time  : {total_time / 60:.2f} minutes")
    print(f"Peak GPU          : {peak_mem:.2f} GB")
    print(f"Peak Reserved GPU : {peak_res:.2f} GB")
    print("-" * 70)
    print(f"Greedy Decoding   -> BLEU: {bleu_greedy:5.2f} | CHRF: {chrf_greedy:5.2f}")
    print(f"Beam Decoding     -> BLEU: {bleu_beam:5.2f} | CHRF: {chrf_beam:5.2f}")
    print("=" * 70 + "\n")


def main() -> None:
    assert torch.cuda.is_available(), "CUDA device required. CPU execution is disabled."

    script_dir = Path(__file__).resolve().parent
    default_config = script_dir.parent / "configs" / "transformer_config.toml"

    parser = argparse.ArgumentParser(description="Multilingual Transformer Engine")
    parser.add_argument("--config", type=str, default=str(default_config))
    args = parser.parse_args()
    

    cfg = AppConfig.from_toml(args.config)


    # --- NEW: Print Key Config Information at Start ---
    print("\n" + "=" * 50)
    print("          INITIALIZING TRAINING PIPELINE")
    print("=" * 50)
    print(f"Dataset Name : {cfg.language.dataset_name}")
    print(f"Language     : {cfg.language.target_language} ({cfg.language.tgt_lang})")
    print(f"Tokenizers   : SRC = {cfg.tokenizer.algo_src.upper()} | TGT = {cfg.tokenizer.algo_tgt.upper()}")
    print(f"Epochs       : {cfg.training.epochs}")
    print(f"Train Size   : {cfg.data.train_size}")
    print(f"Test Size    : {cfg.data.test_size}")


    print("=" * 50 + "\n")


    set_seed(cfg.project.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    pipeline = DataPipeline(
        dataset_name=cfg.language.dataset_name,
        tgt_lang=cfg.language.tgt_lang,
        lang_pair=cfg.language.lang_pair,
        data_dir=cfg.data.data_dir,
        max_len=cfg.data.max_len,
        max_ratio=cfg.data.max_length_ratio,
        oversample=cfg.data.oversample,
        seed=cfg.project.seed,
    )
    tr_src, tr_tgt, val_src, val_tgt = pipeline.acquire_corpus(
        cfg.data.train_size, cfg.data.test_size, cfg.data.force_download
    )

    print("Training tokenizers...")
    tok_src = TokenizerManager.train(tr_src, cfg.tokenizer.algo_src, cfg.tokenizer.max_vocab_size)
    tok_tgt = TokenizerManager.train(tr_tgt, cfg.tokenizer.algo_tgt, cfg.tokenizer.max_vocab_size)

    tr_ds = TranslationDataset(tr_src, tr_tgt, tok_src, tok_tgt, cfg.data.max_len)
    val_ds = TranslationDataset(val_src, val_tgt, tok_src, tok_tgt, cfg.data.max_len)

    tr_loader = pipeline.create_loader(tr_ds, cfg.training.batch_size, shuffle=True)
    val_loader = pipeline.create_loader(val_ds, cfg.training.batch_size, shuffle=False)

    free_memory()

    model = MultilingualTransformer(
        src_vocab_size=tok_src.get_vocab_size(),
        tgt_vocab_size=tok_tgt.get_vocab_size(),
        max_len=cfg.data.max_len,
        d_model=cfg.model.d_model,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        d_ff=cfg.model.d_ff,
        dropout=cfg.model.dropout,
    )
    n_params = sum(p.numel() for p in model.parameters())

    trainer = Trainer(model, cfg, tr_loader, val_loader, tok_src, tok_tgt)
    train_time, peak_mem, peak_res = trainer.fit()

    generator = TranslationGenerator(model, tok_src, tok_tgt, cfg.data.max_len, torch.device("cuda"))
    evaluator = TranslationEvaluator(generator)

    bleu_greedy, chrf_greedy, _ = evaluator.evaluate(
        val_src, val_tgt, method="greedy", sample_size=cfg.training.bleu_sample
    )
    bleu_beam, chrf_beam, _ = evaluator.evaluate(
        val_src, val_tgt, method="beam", beam_size=5, sample_size=cfg.training.bleu_sample
    )

    print_stats_table(
        cfg,
        n_params,
        len(tr_loader),
        train_time,
        peak_mem,
        peak_res,
        bleu_greedy,
        chrf_greedy,
        bleu_beam,
        chrf_beam,
        tok_src.get_vocab_size(),
        tok_tgt.get_vocab_size(),
    )

    print(f"=== Config Sample Translations ({cfg.language.lang_pair}) ===")
    for i, sent in enumerate(cfg.inference.sample_sentences, 1):
        g_pred = generator.greedy_decode(sent)
        b_pred = generator.batched_beam_decode([sent], beam_size=5)[0]
        print(f"[{i}] EN   : {sent}")
        print(f"    Greedy : {g_pred}")
        print(f"    Beam   : {b_pred}\n")


if __name__ == "__main__":
    main()