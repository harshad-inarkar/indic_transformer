# multilingual_transformer

High-performance, from-scratch Sequence-to-Sequence Transformer (Vaswani et al., 2017) built with PyTorch, optimized exclusively for CUDA hardware[cite: 2]. Supports English to 12+ Indic languages via Hugging Face parallel corpora (`ai4bharat/samanantar`, `cfilt/iitb-english-hindi`, `acomquest/Saamayik`)[cite: 2].

---

## Project Specifications

* **Languages:** Hindi_IITB, Hindi, Marathi, Bengali, Gujarati, Kannada, Malayalam, Odia, Punjabi, Tamil, Telugu, Assamese, Sanskrit
* **Datasets:**
  * Hindi_IITB: `cfilt/iitb-english-hindi`
  * Other 11 languages: `ai4bharat/samanantar` (IITM)
  * Sanskrit: `acomquest/Saamayik` (IITB)
* **Tokenizers:** BPE, Unigram, WordPiece, Whitespace, Regex (Source and target languages can use different tokenizers)
* **Vocabulary Size:** 22,400
* **Decoding Strategies:** Greedy, Beam Search
* **Evaluation Metrics:** BLEU, chrF

### Training & Architecture Parameters

* **Training Splits:**
  * Train Size: 50,000 samples
  * Test Size: 10,000 samples
  * Epochs: 10
* **Transformer Architecture (Encoder/Decoder):**
  * Layers: 6
  * Attention Heads: 8
  * Embedding Dimension (`d_model`): 512
  * Feed Forward Dimension (`d_ff`): 2048
  * Dropout: 0.1

---

## Architecture & Features

* **Strict OOP Modular Design:** Separation of concerns between Data, Tokenization, Architecture, Scheduling, and Inference[cite: 2].
* **CUDA Optimized:** Uses `torch.amp.autocast`, fused AdamW, non-blocking page-locked host-to-device transfers, and numerical underflow guardrails[cite: 2].
* **Vectorized Generation:** Batched greedy decoding and batched beam search decoding with cross-attention state caching[cite: 2].
* **Config-Driven:** Fully parameterizable execution via TOML (`configs/transformer_config.toml`)[cite: 2].

---

## Google Colab Setup & Execution

Run these commands in a Google Colab notebook with a GPU runtime (T4, V100, or A100) enabled[cite: 2].

### 1. Setup & Install
Clone the repository and install the package in editable mode[cite: 2].
```bash
# Verify GPU availability
!nvidia-smi

# # 1. Always start from the Colab root workspace
%cd /content

# # 2. Safely wipe the old directory
!rm -rf multilingual_transformer

# 3. Clone fresh
!git clone https://github.com/harshad-inarkar/multilingual_transformer.git

# 4. Enter the clean directory
%cd multilingual_transformer

# 5. Install
!pip install -q -e .

# Go to internal project dir
%cd multilingual_transformer

```

### 2. Train the Model

Execute the end-to-end pipeline using the default TOML configuration. This will download the dataset, train the tokenizers, run the training loop, calculate BLEU/chrF metrics, and save the best weights to the `checkpoints/` directory.

```bash
!python scripts/train.py --config configs/transformer_config.toml

```

### 3. Interactive Translation Widget

Render a UI text box and dropdown directly in your notebook to test translations instantly. This automatically loads your trained checkpoint and cached tokenizers.

Run this in a standard Python cell (not a bash cell):

```python
from multilingual_transformer.ui.widget import launch

launch(config_path="configs/transformer_config.toml")

```

### 4. Interactive Terminal REPL (Alternative)

If you prefer a pure text-based loop without UI widgets, you can run the interactive terminal script. Type your English sentences and get immediate greedy and beam search translations.

```bash
!python scripts/interactive.py --config configs/transformer_config.toml

```

---

## Local Installation & Configuration

If you are running this locally on a Linux/macOS/Windows machine with a dedicated NVIDIA GPU:

```bash
git clone [https://github.com/harshad-inarkar/multilingual_transformer.git](https://github.com/harshad-inarkar/multilingual_transformer.git)
cd multilingual_transformer
pip install -e .

```

### Modifying the Configuration

Modify `configs/transformer_config.toml` to specify target languages, model dimensions, training hyperparameters, or test prompts:

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