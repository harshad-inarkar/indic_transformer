from __future__ import annotations

from pathlib import Path
import ipywidgets as widgets
from IPython.display import display
import torch
from pathlib import Path

# # Automatically enable Colab widget manager if running in Google Colab
# try:
#     from google.colab import output
#     output.enable_custom_widget_manager()
# except ImportError:
#     pass

from multilingual_transformer.configs.config  import AppConfig
from multilingual_transformer.data.dataset import DataPipeline
from multilingual_transformer.data.tokenizer import TokenizerManager
from multilingual_transformer.engine.decoder import TranslationGenerator
from multilingual_transformer.models.transformer import MultilingualTransformer



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
        method_dropdown = widgets.Dropdown(
            options=["greedy", "beam"], value="greedy", description="Method:"
        )
        translate_btn = widgets.Button(
            description="Translate", button_style="primary"
        )
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
        
        # Return the VBox instead of just displaying it
        return widgets.VBox([text_box, method_dropdown, translate_btn, out])


def launch(config_path: str | None = None) -> widgets.VBox | None:
    """Loads trained checkpoint and renders the interactive widget in Colab/Jupyter."""
    if config_path is None:
        # Dynamically resolve relative to widget.py (ui/ -> parent is package root)
        config_path = str(Path(__file__).resolve().parent.parent / "configs" / "transformer_config.toml")


    assert torch.cuda.is_available(), "CUDA device required. CPU execution is disabled."
    device = torch.device("cuda")


    cfg = AppConfig.from_toml(config_path)

    tag = cfg.language.lang_pair.lower().replace("-", "_")
    ckpt_path = cfg.data.checkpoint_dir / f"transformer_{tag}_best.pt"

    if not ckpt_path.exists():
        print(f"[Error] Checkpoint not found at: {ckpt_path}")
        print("Please run training first: !python scripts/train.py")
        return None

    print("Loading cached dataset and tokenizers (this takes a few seconds)...")
    
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

    print("Loading model weights...")
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

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    print("Ready! Rendering widget...\n")
    generator = TranslationGenerator(model, tok_src, tok_tgt, cfg.data.max_len, device)
    ui = TranslationWidget(generator, cfg.language.tgt_lang)
    
    # Return the widget directly to the cell output
    return ui.render()