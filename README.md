# Multilingual Transformer

A from-scratch implementation of the Transformer (Vaswani et al., 2017) in PyTorch for **English → Indic machine translation**. Train one model per target language on public parallel corpora, then evaluate with BLEU and chrF and translate interactively with greedy or beam search decoding.

## Features

- Encoder-decoder Transformer built from scratch (no `nn.Transformer`)
- 12 target languages: Hindi, Marathi, Bengali, Gujarati, Kannada, Malayalam, Odia, Punjabi, Tamil, Telugu, Assamese, Sanskrit
- Five tokenizer types (BPE, Unigram, WordPiece, Whitespace, Regex), selectable independently for source and target
- Batched greedy decoding and batched beam search with cross-attention caching
- Mixed-precision training (`torch.amp.autocast`), fused AdamW, pinned-memory non-blocking transfers
- Fully TOML-configured: no code changes needed to switch language, tokenizer or hyperparameters
- Built-in evaluation (BLEU, chrF), terminal REPL and Jupyter/Colab widget

## Supported Languages and Datasets

| Language(s) | Code(s) | Dataset |
|---|---|---|
| Hindi | `hi` | [`cfilt/iitb-english-hindi`](https://huggingface.co/datasets/cfilt/iitb-english-hindi) |
| Marathi, Bengali, Gujarati, Kannada, Malayalam, Odia, Punjabi, Tamil, Telugu, Assamese | `mr`, `bn`, `gu`, `kn`, `ml`, `or`, `pa`, `ta`, `te`, `as` | [`ai4bharat/samanantar`](https://huggingface.co/datasets/ai4bharat/samanantar) |
| Sanskrit | `sa` | [`acomquest/Saamayik`](https://huggingface.co/datasets/acomquest/Saamayik) |

Datasets are downloaded automatically from Hugging Face on first run.

## Default Configuration

| Category | Setting | Value |
|---|---|---|
| Model | Encoder / decoder layers | 6 / 6 |
| | Attention heads | 8 |
| | `d_model` | 512 |
| | `d_ff` | 2048 |
| | Dropout | 0.1 |
| Data | Train samples | 50,000 |
| | Test samples | 10,000 |
| | Vocabulary size | 22,400 |
| Training | Epochs | 10 |
| | Batch size | 224 |
| | Learning rate | 5e-4 |
| | Optimizer | AdamW (fused) |
| Inference | Decoding | Greedy, Beam Search |
| Evaluation | Metrics | BLEU, chrF |

All values are set in `configs/transformer_config.toml`.

## Requirements

- Python 3.9+
- NVIDIA GPU with CUDA support (T4 or better recommended)
- PyTorch with CUDA build ([install guide](https://pytorch.org/get-started/locally/))

## Installation

```bash
git clone https://github.com/harshad-inarkar/multilingual_transformer.git
cd multilingual_transformer
pip install -e .
```

## Quick Start

All commands are run from a working directory, where tokenizers, checkpoints and logs are written.

```bash
mkdir -p work_dir && cd work_dir
```

**1. Train** (downloads data, trains tokenizers, trains the model, reports BLEU/chrF, saves the best checkpoint to `checkpoints/`):

```bash
python -m multilingual_transformer.scripts.train
```

**2. Translate in the terminal:**

```bash
python -m multilingual_transformer.scripts.interactive
```

Enter an English sentence to get both greedy and beam search translations.

**3. Translate in a notebook** (run in a Python cell):

```python
from multilingual_transformer.ui.widget import launch
launch()
```

The widget loads the trained checkpoint and cached tokenizers automatically.

## Google Colab

**0. Enable a GPU runtime (Runtime → Change runtime type), then setup cell:**

```python
%cd /content
!rm -rf multilingual_transformer
!git clone https://github.com/harshad-inarkar/multilingual_transformer.git
%cd multilingual_transformer
!pip install -q -e .
!mkdir -p work_dir
```

**1. Train** (downloads data, trains tokenizers, trains the model, reports BLEU/chrF, saves the best checkpoint to `checkpoints/`):

```bash
%cd work_dir

python -m multilingual_transformer.scripts.train
```

**2. Translate in the terminal:**

```bash
python -m multilingual_transformer.scripts.interactive
```

Enter an English sentence to get both greedy and beam search translations.

**3. Translate in a notebook** (run in a Python cell):

```python
from multilingual_transformer.ui.widget import launch
launch()
```


## Configuration

Edit `configs/transformer_config.toml`. Example, training English → Hindi:

```toml
[language]
target_language = "Hindi"
dataset_name = "ai4bharat/samanantar"
tgt_lang = "hi"
lang_pair = "EN-HI"

[model]
d_model = 512
num_layers = 6
num_heads = 8
d_ff = 2048
dropout = 0.1

[training]
epochs = 10
batch_size = 224
lr = 5.0e-4
```

To train a different language, change `target_language`, `tgt_lang`, `lang_pair` and, if needed, `dataset_name` (see the dataset table above).

## Project Structure

```
multilingual_transformer/
├── configs/
│   └── transformer_config.toml    # all runtime settings
├── multilingual_transformer/
│   ├── scripts/
│   │   ├── train.py               # end-to-end training + evaluation
│   │   └── interactive.py         # terminal translation REPL
│   └── ui/
│       └── widget.py              # notebook translation widget
└── pyproject.toml
```

The package is organised into separate modules for data loading, tokenization, model architecture, learning-rate scheduling and inference.


## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Lower `batch_size` in the config |
| Widget or REPL cannot find a checkpoint | Run from the same `work_dir` used for training |
| Dataset download fails | Check internet access and Hugging Face availability |

## References

- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), 2017
- Ramesh et al., [*Samanantar: The Largest Publicly Available Parallel Corpora Collection for 11 Indic Languages*](https://arxiv.org/abs/2104.05596), 2021
- Kunchukuttan et al., [*The IIT Bombay English-Hindi Parallel Corpus*](https://arxiv.org/abs/1710.02855), 2018
- Ayush Maheshwari et al., [*Sāmayik: A Benchmark and Dataset for English-Sanskrit Translation*](https://arxiv.org/abs/2305.14004), 2024


## License

The source code is released under the [MIT License](LICENSE).
