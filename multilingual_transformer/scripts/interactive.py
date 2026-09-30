import argparse
import sys
import torch
from config import AppConfig
from multilingual_transformer.data.dataset import DataPipeline
from multilingual_transformer.data.tokenizer import TokenizerManager
from multilingual_transformer.engine.decoder import TranslationGenerator
from multilingual_transformer.models.transformer import MultilingualTransformer

def main():
    parser = argparse.ArgumentParser(description="Interactive Translation Tester")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = AppConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = cfg.language.lang_pair.lower().replace("-", "_")
    ckpt_path = cfg.data.checkpoint_dir / f"transformer_{tag}_best.pt"

    # Guardrail: Ensure train.py was run first
    if not ckpt_path.exists():
        print(f"\n[Error] Checkpoint not found at: {ckpt_path}")
        print("Please run 'python scripts/train.py' first to generate the model.\n")
        sys.exit(1)

    print("Loading tokenizers and model... (this takes a few seconds)")
    
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
    tr_src, tr_tgt, _, _ = pipeline.acquire_corpus(cfg.data.train_size, cfg.data.test_size)
    tok_src = TokenizerManager.train(tr_src, cfg.tokenizer.algo_src, cfg.tokenizer.max_vocab_size)
    tok_tgt = TokenizerManager.train(tr_tgt, cfg.tokenizer.algo_tgt, cfg.tokenizer.max_vocab_size)

    model = MultilingualTransformer(
        src_vocab_size=tok_src.get_vocab_size(),
        tgt_vocab_size=tok_tgt.get_vocab_size(),
        max_len=cfg.data.max_len,
        d_model=cfg.model.d_model,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        d_ff=cfg.model.d_ff,
        dropout=cfg.model.dropout,
    ).to(device)

    # Load trained checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    generator = TranslationGenerator(model, tok_src, tok_tgt, cfg.data.max_len, device)
    tgt_code = cfg.language.tgt_lang.upper()

    print("\n" + "=" * 50)
    print(f"Interactive Translator (EN -> {tgt_code})")
    print("Type 'q' or 'exit' to quit.")
    print("=" * 50 + "\n")

    while True:
        try:
            text = input("Enter English: ").strip()
            if not text:
                continue
            if text.lower() in ["q", "exit"]:
                print("Exiting.")
                break

            greedy_out = generator.greedy_decode(text)
            beam_out = generator.batched_beam_decode([text], beam_size=5)[0]

            print(f"[{tgt_code} Greedy] : {greedy_out}")
            print(f"[{tgt_code} Beam]   : {beam_out}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

if __name__ == "__main__":
    main()