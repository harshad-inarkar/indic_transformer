"""
colab_widget_demo.py
Copy and paste this code into a Google Colab cell to launch the interactive translator.
Ensure you have run training at least once so a checkpoint exists.
"""
import torch
from indic_transformer.config import AppConfig
from indic_transformer.data.dataset import DataPipeline
from indic_transformer.data.tokenizer import TokenizerManager
from indic_transformer.engine.decoder import TranslationGenerator
from indic_transformer.models.transformer import MultilingualTransformer
from indic_transformer.ui.widget import TranslationWidget

def launch_widget(config_path="configs/default.yaml"):
    cfg = AppConfig.from_yaml(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Rebuild tokenizers from the training data split
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

    # 2. Instantiate the model
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

    # 3. Load the best checkpoint
    tag = cfg.language.lang_pair.lower().replace("-", "_")
    ckpt_path = cfg.data.checkpoint_dir / f"transformer_{tag}_best.pt"
    
    if not ckpt_path.exists():
        print(f"Checkpoint not found at {ckpt_path}. Please run training first.")
        return

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # 4. Render the UI
    generator = TranslationGenerator(model, tok_src, tok_tgt, cfg.data.max_len, device)
    ui = TranslationWidget(generator, cfg.language.tgt_lang)
    ui.render()

if __name__ == "__main__":
    launch_widget()