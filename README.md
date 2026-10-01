# Multilingual Transformer

A from-scratch implementation of the Transformer (Vaswani et al., 2017) in PyTorch for **English → Indic machine translation**. Train one model per target language on public parallel corpora, then evaluate with BLEU and chrF and translate interactively with greedy or beam search decoding.

## Features

- Encoder-decoder Transformer built from scratch (no `nn.Transformer`), fused SDPA attention
- 12 target languages: Hindi, Marathi, Bengali, Gujarati, Kannada, Malayalam, Odia, Punjabi, Tamil, Telugu, Assamese, Sanskrit
- Five tokenizer types (BPE, Unigram, WordPiece, Whitespace, Regex), selectable independently for source and target; tokenizers are cached and reused at inference
- Disjoint **train / validation / test** splits: validation drives per-epoch loss, BLEU/chrF are computed on the test split only
- Batched greedy decoding and vectorised batched beam search (encoder runs once per batch, length-sorted batching)
- Mixed-precision training (`torch.amp.autocast`), fused AdamW, dynamic padding, pre-tokenised data
- Choose between keeping the **latest** or the **best-validation** checkpoint (`save_best`)
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

- Python 3.11+
- NVIDIA GPU with CUDA for training (T4 or better recommended); inference also runs on CPU
- PyTorch ≥ 2.3 ([install guide](https://pytorch.org/get-started/locally/))

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

## Google Colab Setup

You can use the ready-made notebook, which clones the repo and runs the train/translate steps:
https://colab.research.google.com/drive/1umfH2Zis4b6Hzya5qL5g0eMsComwdU-f

Or build your own notebook with the cells below. Run each block as a separate cell.

**0. Enable a GPU runtime:** *Runtime → Change runtime type → T4 GPU* (or better).

**1. Clone and install**

```python
%cd /content
!rm -rf multilingual_transformer
!git clone https://github.com/harshad-inarkar/multilingual_transformer.git
%cd multilingual_transformer
!pip install -q -e .
!mkdir -p work_dir
```

**2. (Optional) Edit the configuration from a cell**

You can open `multilingual_transformer/configs/transformer_config.toml` in the Colab file browser and edit it directly.


**3. Train** (downloads data, trains tokenizers, trains the model, reports BLEU/chrF on the test split)

```python
%cd /content/multilingual_transformer/work_dir
!python -m multilingual_transformer.scripts.train
```

**4. Translate in terminal**
Run from the same directory as training, because the checkpoint and tokenizers are found relative to it:


```python
%cd /content/multilingual_transformer/work_dir
python -m multilingual_transformer.scripts.interactive
```

**5. Translate in the notebook**

Run from the same directory as training, because the checkpoint and tokenizers are found relative to it:

```python
%cd /content/multilingual_transformer/work_dir
from multilingual_transformer.ui.widget import launch
launch()
```


## Configuration

Edit `multilingual_transformer/configs/transformer_config.toml`. Only three keys select the language pair:

```toml
[language]
src_lang = "English"
target_lang = "Hindi"
dataset_name = "ai4bharat/samanantar"
```

`tgt_lang` (`ta`) and `lang_pair` (`EN-TA`) are derived from `configs/language_config.toml` (name → code, case-insensitive). To add a language, add one line there.

### Data splits

```toml
[data]
train_size = 50000
val_size = 5000     # per-epoch validation loss
test_size = 5000    # final BLEU/chrF and test samples
```

The three splits are disjoint, and the English side is de-duplicated across them. Validation loss is computed on the validation split after every epoch; BLEU and chrF are computed on the **test split only**.

### Checkpoints

```toml
[training]
save_best = false
bleu_sample = 300
```

| `save_best` | Behaviour | Checkpoint file |
|---|---|---|
| `false` (default) | Saves the latest weights after every epoch; training and inference use the final epoch | `checkpoints/transformer_en_xx_last.pt` |
| `true` | Saves only when validation loss improves; the best weights are restored before the final evaluation and used for inference | `checkpoints/transformer_en_xx_best.pt` |

The REPL and widget read the same flag, so keep it the same for training and inference. `bleu_sample` is the number of test sentences used for BLEU/chrF.

### Sample translations

```toml
[inference]
sample_source = "config"   # "config" or "test"
num_samples = 5            # used when sample_source = "test"
sample_sentences = ["this is a wonderful day", "education is essential for every child"]
```

| `sample_source` | What is printed after training |
|---|---|
| `"config"` | Greedy and beam translations of `sample_sentences` |
| `"test"` | The first `num_samples` test pairs, with the reference translation shown next to each prediction |

The training banner at the start of a run shows `Val Size`, `Save Best` and `Samples From` along with the other key settings.

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

Artifacts (in the working directory): `data/` (`train_*`, `val_*`, `test_*` files), `tokenizers/`, `checkpoints/`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Lower `batch_size` in the config |
| Widget or REPL cannot find a checkpoint/tokenizer | Run from the same `work_dir` used for training |
| Checkpoint not found although training finished | `save_best` differs between training and inference (`_last.pt` vs `_best.pt`); set it to the value used for training |
| Old checkpoint fails to load | Layer names changed (`embed.emb`); retrain |
| Data is downloaded again after changing sizes | Cache filenames include the split sizes; this is expected |
| Dataset download fails | Check internet access, Hugging Face availability, and `datasets<4` |

## References

- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), 2017
- Ramesh et al., [*Samanantar*](https://arxiv.org/abs/2104.05596), 2021
- Kunchukuttan et al., [*The IIT Bombay English-Hindi Parallel Corpus*](https://arxiv.org/abs/1710.02855), 2018
- Maheshwari et al., [*Sāmayik*](https://arxiv.org/abs/2305.14004), 2024

## License

The source code is released under the [MIT License](LICENSE).
