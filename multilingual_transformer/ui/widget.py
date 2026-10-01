from __future__ import annotations

from pathlib import Path

import ipywidgets as widgets
import torch

from multilingual_transformer.configs.config import AppConfig
from multilingual_transformer.engine.decoder import TranslationGenerator
from multilingual_transformer.engine.loader import load_model, load_tokenizers


class TranslationWidget:
    def __init__(self, generator: TranslationGenerator, tgt_code: str) -> None:
        self.generator = generator
        self.tgt_code = tgt_code.upper()

    def render(self) -> widgets.VBox:
        text_box = widgets.Text(
            value="education is essential for every child",
            description="EN:",
            layout=widgets.Layout(width="500px"),
        )
        method_dropdown = widgets.Dropdown(options=["greedy", "beam"], value="greedy", description="Method:")
        translate_btn = widgets.Button(description="Translate", button_style="primary")
        out = widgets.Output()

        def on_click(_: widgets.Button) -> None:
            out.clear_output()
            with out:
                if method_dropdown.value == "greedy":
                    res = self.generator.greedy_decode(text_box.value)
                else:
                    res = self.generator.batched_beam_decode([text_box.value], beam_size=5)[0]
                print(f"EN : {text_box.value}")
                print(f"{self.tgt_code}: {res}")

        translate_btn.on_click(on_click)
        return widgets.VBox([text_box, method_dropdown, translate_btn, out])


def launch(config_path: str | None = None) -> widgets.VBox | None:
    """Loads trained checkpoint and renders the interactive widget in Colab/Jupyter."""
    if config_path is None:
        # ui/ -> parent is the package root
        config_path = str(Path(__file__).resolve().parent.parent / "configs" / "transformer_config.toml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = AppConfig.from_toml(config_path)
    ckpt_path = cfg.checkpoint_path()

    if not ckpt_path.exists():
        print(f"[Error] Checkpoint not found at: {ckpt_path}")
        print("Please run training first: !python scripts/train.py")
        return None

    print("Loading cached dataset and tokenizers (this takes a few seconds)...")
    tok_src, tok_tgt = load_tokenizers(cfg)

    print("Loading model weights...")
    model = load_model(cfg, tok_src, tok_tgt, device)

    print("Ready! Rendering widget...\n")
    generator = TranslationGenerator(model, tok_src, tok_tgt, cfg.data.max_len, device)
    return TranslationWidget(generator, cfg.language.tgt_lang).render()
