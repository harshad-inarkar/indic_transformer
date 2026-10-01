# Multilingual Transformer

A from-scratch implementation of the Transformer (Vaswani et al., 2017) in PyTorch for **English → Indic machine translation**. Train one model per target language on public parallel corpora, then evaluate with BLEU and chrF and translate interactively with greedy or beam search decoding.

## Features

- Encoder-decoder Transformer built from scratch (no `nn.Transformer`), fused SDPA attention
- 12 target languages: Hindi, Marathi, Bengali, Gujarati, Kannada, Malayalam, Odia, Punjabi, Tamil, Telugu, Assamese, Sanskrit
- Five tokenizer types (BPE, Unigram, WordPiece, Whitespace, Regex), selectable independently for source and target; tokenizers are cached and reused at inference
- Batched greedy decoding and vectorised batched beam search (encoder runs once per batch, length-sorted batching)
- Mixed-precision training (`torch.amp.autocast`), fused AdamW, dynamic padding, pre-tokenised data
- TOML-configured: choose languages **by name**; codes and `lang_pair` are derived from `configs/language_config.toml`
- Built-in evaluation (BLEU, chrF), terminal REPL and Jupyter/Colab widget

## Supported Languages and Datasets

| Language(s) | Dataset |
|---|---|
| Hindi | [`cfilt/iitb-english-hindi`](https://huggingface.co/datasets/cfilt/iitb-english-hindi) or Samanantar |
| Marathi, Bengali, Gujarati, Kannada, Malayalam, Odia, Punjabi, Tamil, Telugu, Assamese (and Hindi) | [`ai4bharat/samanantar`](https://huggingface.co/datasets/ai4bharat/samanantar) |
| Sanskrit | [`acomquest/Saamayik`](https://huggingface.co/datasets/acomquest/Saamayik) |

Datasets are downloaded automatically from Hugging Face on first run. A language/dataset mismatch (e.g. Tamil with Saamayik) raises a clear error.

## Requirements

- Python 3.9+
- NVIDIA GPU with CUDA for training (T4 or better recommended); inference also runs on CPU
- PyTorch ≥ 2.3 ([install guide](https://pytorch.org/get-started/locally/))
- `datasets<4` (Samanantar/Saamayik use loading scripts)

## Installation

```bash
git clone https://github.com/harshad-inarkar/multilingual_transformer.git
cd multilingual_transformer
pip install -e .
```

## Quick Start

Run from a working directory; data, tokenizers and checkpoints are written there.

```bash
mkdir -p work_dir && cd work_dir
python -m multilingual_transformer.scripts.train          # data -> tokenizers -> train -> BLEU/chrF
python -m multilingual_transformer.scripts.interactive    # terminal REPL
```

Notebook:

```python
from multilingual_transformer.ui.widget import launch
launch()
```

On Colab: enable a GPU runtime, then clone, `pip install -q -e .`, `mkdir work_dir`, `%cd work_dir` and run the same commands.

## Configuration

Edit `multilingual_transformer/configs/transformer_config.toml`. Only three keys select the language pair:

```toml
[language]
src_lang = "English"
target_lang = "Tamil"
dataset_name = "ai4bharat/samanantar"
```

`tgt_lang` (`ta`) and `lang_pair` (`EN-TA`) are derived from `configs/language_config.toml` (name → code, case-insensitive). To add a language, add one line there.

## Project Structure

```
multilingual_transformer/
├── configs/
│   ├── config.py                  # dataclasses + TOML loading, artifact paths
│   ├── transformer_config.toml    # runtime settings
│   └── language_config.toml       # language name -> code registry
├── data/        dataset.py, tokenizer.py
├── models/      attention.py, layers.py, transformer.py
├── engine/      trainer.py, decoder.py, evaluator.py, loader.py
├── scripts/     train.py, interactive.py
├── ui/          widget.py
└── utils/       helpers.py
```

Artifacts (in the working directory): `data/`, `tokenizers/`, `checkpoints/`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Lower `batch_size` in the config |
| Widget or REPL cannot find a checkpoint/tokenizer | Run from the same `work_dir` used for training |
| Old checkpoint fails or translates poorly | Architecture changed (embedding scaling); retrain |
| Dataset download fails | Check internet access, Hugging Face availability, and `datasets<4` |

## References

- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), 2017
- Ramesh et al., [*Samanantar*](https://arxiv.org/abs/2104.05596), 2021
- Kunchukuttan et al., [*The IIT Bombay English-Hindi Parallel Corpus*](https://arxiv.org/abs/1710.02855), 2018
- Maheshwari et al., [*Sāmayik*](https://arxiv.org/abs/2305.14004), 2024

## License

The source code is released under the [MIT License](LICENSE).
