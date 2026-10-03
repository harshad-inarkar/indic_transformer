import argparse
from pathlib import Path
import tomllib
import warnings

import torch

from multilingual_transformer.configs.config import AppConfig
from multilingual_transformer.data.tokenizer import TokenizerManager
from multilingual_transformer.engine.decoder import TranslationGenerator
from multilingual_transformer.models.transformer import MultilingualTransformer

def main():
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir.parent / "configs" / "multilingual_config.toml"
    
    with open(config_path, "rb") as f:
        raw_cfg = tomllib.load(f)
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = AppConfig.from_toml(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok_path = cfg.data.tokenizer_dir / f"multilingual_shared_{cfg.tokenizer.max_vocab_size}.json"
    ckpt_path = cfg.checkpoint_path("best" if cfg.training.save_best else "last")

    if not ckpt_path.exists():
        print(f"\n[Error] Checkpoint not found at: {ckpt_path}\n")
        return

    shared_tok = TokenizerManager.load(tok_path)
    vocab_sz = shared_tok.get_vocab_size()

    model = MultilingualTransformer(
        src_vocab_size=vocab_sz, tgt_vocab_size=vocab_sz, max_len=cfg.data.max_len,
        d_model=cfg.model.d_model, num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads, d_ff=cfg.model.d_ff, dropout=cfg.model.dropout,
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    generator = TranslationGenerator(
        model, shared_tok, shared_tok, cfg.data.max_len, device,
        no_repeat_ngram_size=cfg.inference.no_repeat_ngram_size,
    )

    print("\n" + "=" * 60)
    print("Universal Translation Matrix (Zero-Shot Enabled)")
    print("Supported Target Codes: en, hi, mr, sa")
    print("Type 'q' to quit.")
    print("=" * 60 + "\n")

    while True:
        try:
            tgt_lang = input("\nEnter Target Language code (e.g., mr) [or 'q']: ").strip().lower()
            if tgt_lang in ["q", "exit"]: break
            if f"<2{tgt_lang}>" not in raw_cfg["multilingual"]["target_tokens"]:
                print("Unsupported language code.")
                continue

            text = input("Enter sentence to translate: ").strip()
            if not text: continue

            prefixed_text = f"<2{tgt_lang}> {text}"
            beam_out = generator.batched_beam_decode([prefixed_text], beam_size=5)[0]
            print(f"[{tgt_lang.upper()} Beam]: {beam_out}")
            
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

if __name__ == "__main__":
    main()