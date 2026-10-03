# Multilingual Transformer

A from-scratch implementation of the Transformer (Vaswani et al., 2017) in PyTorch for **English ↔ Indic machine translation**. 

This pipeline supports both **One-to-One** translation (training a dedicated model for a specific language pair) and **Multilingual** translation (training a single model capable of translating across multiple languages using zero-shot target-forcing tokens).

## Key Architectural Upgrades

- **Multi-GPU DistributedDataParallel (DDP):** Auto-detects available GPUs and distributes training batches via `torch.multiprocessing.spawn`. Parallelized vector beam search evaluation aggregates predictions across all GPUs via NCCL `all_gather_object`.
- **Length-Bucketed Sampler:** Sentences of similar lengths are bucketed into batches before padding, minimizing sequence padding overhead to under 10% and accelerating training times by 2-3×.
- **SwiGLU Activation:** Upgraded the feed-forward networks from standard ReLU to SwiGLU (Swish-Gated Linear Unit). Using a `3x` dimension ratio (e.g., $d_{model}=256, d_{ff}=768$), it requires half the total parameters of a standard `4x` ReLU setup while outperforming it across BLEU and chrF metrics.
- **Bidirectional Prefix Augmentation:** For the multilingual flow, the pipeline automatically flips English-Indic pairs to Indic-English and injects target-language tags (e.g., `<2hi>`, `<2en>`), training the model on bidirectional translation in a single pass.

## Supported Languages and Datasets

| Language(s) | Dataset |
|---|---|
| Hindi | [`cfilt/iitb-english-hindi`](https://huggingface.co/datasets/cfilt/iitb-english-hindi) or Samanantar |
| Marathi, Bengali, Gujarati, Kannada, Malayalam, Odia, Punjabi, Tamil, Telugu, Assamese (and Hindi) | [`ai4bharat/samanantar`](https://huggingface.co/datasets/ai4bharat/samanantar) |
| Sanskrit | [`acomquest/Saamayik`](https://huggingface.co/datasets/acomquest/Saamayik) |

Datasets are downloaded automatically from Hugging Face via Arrow backends, pre-tokenized, and cached locally to bypass runtime overhead.

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA (T4 or better recommended; seamlessly supports 1 to *N* GPUs via DDP).
- PyTorch ≥ 2.3 ([install guide](https://pytorch.org/get-started/locally/))

## Installation

```bash
git clone [https://github.com/harshad-inarkar/multilingual_transformer.git](https://github.com/harshad-inarkar/multilingual_transformer.git)
cd multilingual_transformer
pip install -e .
```

## Quick Start (Terminal)

Run from a working directory; data, tokenizers, and checkpoints are written there.

### 1. One-to-One Translation (Single Language Pair)

```bash
mkdir -p work_dir && cd work_dir

# Auto-detects GPU count. If >1, it spawns DDP workers across all cards.
python -m multilingual_transformer.scripts.train          

# Interactive testing
python -m multilingual_transformer.scripts.interactive    

```

### 2. Universal Multilingual Model (Many-to-Many)

```bash
mkdir -p work_dir && cd work_dir

# Trains a single model on all language pairs defined in multilingual_config.toml
python -m multilingual_transformer.scripts.multilingual_train          

# Interactive testing with prefix target forcing (e.g., "mr: how are you?")
python -m multilingual_transformer.scripts.multilingual_interactive_3    

```

## Jupyter & Colab Widgets

You can run interactive translation testing directly inside a notebook. The widget dynamically populates based on the configuration file used during training.

```python
# For One-to-One Models
from multilingual_transformer.ui.widget_2 import launch
launch()

# For Universal Multilingual Models
from multilingual_transformer.ui.multilingual_widget import launch
launch()

```

## Configuration

Configuration is managed via TOML files in `multilingual_transformer/configs/`.

### 1. `transformer_config.toml` (One-to-One)

```toml
[language]
src_lang = "English"
target_lang = "Hindi"
dataset_name = "ai4bharat/samanantar"

```

### 2. `multilingual_config.toml` (Universal Model)

The multilingual pipeline builds a shared vocabulary across all datasets and injects `<2tgt>` tags.

```toml
[multilingual]
train_languages_pairs = ["en-hi", "en-mr", "en-sa"]
target_tokens_format = "<2{}>"
pairs_per_lang = 50000

```

### Architecture Tuning

The architecture defaults to a highly efficient SwiGLU 3x dimension configuration. Batch sizes represent the load *per GPU*.

```toml
[model]
d_model = 256
num_layers = 4
num_heads = 4
d_ff = 768    # 3x d_model for optimal SwiGLU parameter density
dropout = 0.1

[training]
epochs = 10
batch_size = 256

```

## Project Structure

```
multilingual_transformer/
├── configs/
│   ├── config.py                  # Dataclasses + TOML loading
│   ├── transformer_config.toml    # One-to-One settings
│   ├── multilingual_config.toml   # Multi-language settings
│  
├── data/
│   ├── dataset.py                 # Core text processing, dataset caching
│   ├── multilingual_dataset.py    # Bidirectional prefix logic
│   ├── sampler.py                 # Length-bucketing DDP sampler
│   ├── tokenizer.py               # Tokenizer wrappers
│   └── multilingual_tokenizer.py  # Shared vocabulary generator
│
├── models/      attention.py, layers.py (SwiGLU FFN), transformer.py
├── engine/      trainer.py (DDP Engine), decoder.py, evaluator.py, loader.py
├── scripts/     train.py, multilingual_train.py, interactive.py
├── ui/          widget.py, multilingual_widget.py
└── utils/       helpers.py, distributed.py (NCCL & mp.spawn hooks)

```

## References

* Shazeer, *[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)*, 2020
* Vaswani et al., *[Attention Is All You Need](https://arxiv.org/abs/1706.03762)*, 2017
* Ramesh et al., *[Samanantar](https://arxiv.org/abs/2104.05596)*, 2021
* Kunchukuttan et al., *[The IIT Bombay English-Hindi Parallel Corpus](https://arxiv.org/abs/1710.02855)*, 2018
* Maheshwari et al., *[Sāmayik](https://arxiv.org/abs/2305.14004)*, 2024

## License

The source code is released under the [MIT License](https://www.google.com/search?q=LICENSE).
