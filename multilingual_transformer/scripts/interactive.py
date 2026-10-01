import argparse
import sys
from pathlib import Path

import torch

from multilingual_transformer.configs.config import AppConfig
from multilingual_transformer.engine.decoder import TranslationGenerator
from multilingual_transformer.engine.loader import load_model, load_tokenizers


def main():
    script_dir = Path(__file__).resolve().parent
    default_config = script_dir.parent / "configs" / "transformer_config.toml"

    parser = argparse.ArgumentParser(description="Interactive Translation Tester")
    parser.add_argument("--config", type=str, default=str(default_config))
    args = parser.parse_args()

    cfg = AppConfig.from_toml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = cfg.checkpoint_path()

    # Guardrail: Ensure train.py was run first
    if not ckpt_path.exists():
        print(f"\n[Error] Checkpoint not found at: {ckpt_path}")
        print("Please run 'python scripts/train.py' first to generate the model.\n")
        sys.exit(1)

    print("Loading tokenizers and model... (this takes a few seconds)")
    tok_src, tok_tgt = load_tokenizers(cfg)
    model = load_model(cfg, tok_src, tok_tgt, device)

    generator = TranslationGenerator(
        model, tok_src, tok_tgt, cfg.data.max_len, device,
        no_repeat_ngram_size=cfg.inference.no_repeat_ngram_size,
    )
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
