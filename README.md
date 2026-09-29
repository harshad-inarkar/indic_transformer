# Indic-Transformer

High-performance, from-scratch Sequence-to-Sequence Transformer (Vaswani et al., 2017) built with PyTorch, optimized exclusively for CUDA hardware. Supports English to 12+ Indic languages via Hugging Face parallel corpora (`ai4bharat/samanantar`, `cfilt/iitb-english-hindi`, `acomquest/Saamayik`).

---

## Architecture & Features

* **Strict OOP Modular Design:** Separation of concerns between Data, Tokenization, Architecture, Scheduling, and Inference.
* **CUDA Optimized:** Uses `torch.amp.autocast`, fused AdamW, non-blocking page-locked host-to-device transfers, and numerical underflow guardrails.
* **Vectorized Generation:** Batched greedy decoding and batched beam search decoding with cross-attention state caching.
* **Config-Driven:** Fully parameterizable execution via YAML (`configs/default.yaml`).

---

## Google Colab Setup & Execution

Run these commands in a Google Colab notebook with a GPU runtime (T4, V100, or A100) enabled.

### 1. Setup & Install
Clone the repository and install the package in editable mode.
```bash
# Verify GPU availability
!nvidia-smi

# Clone and install
!git clone [https://github.com/](https://github.com/)<your-username>/indic-transformer.git
%cd indic-transformer
!pip install -q -e .



### 2. Train the Model

Execute the end-to-end pipeline using the default YAML configuration. This will download the dataset, train the tokenizers, run the training loop, calculate BLEU/chrF metrics, and save the best weights to the `checkpoints/` directory.

```bash
!python scripts/train.py --config configs/default.yaml

```

### 3. Interactive Translation Widget

Render a UI text box and dropdown directly in your notebook to test translations instantly. This automatically loads your trained checkpoint and cached tokenizers.

Run this in a standard Python cell (not a bash cell):

```python
from indic_transformer.ui.widget import launch

launch(config_path="configs/default.yaml")

```

### 4. Interactive Terminal REPL (Alternative)

If you prefer a pure text-based loop without UI widgets, you can run the interactive terminal script. Type your English sentences and get immediate greedy and beam search translations.

```bash
!python scripts/interactive.py --config configs/default.yaml

```

---

## Local Installation & Configuration

If you are running this locally on a Linux/macOS/Windows machine with a dedicated NVIDIA GPU:

```bash
git clone [https://github.com/](https://github.com/)<your-username>/indic-transformer.git
cd indic-transformer
pip install -e .

```

### Modifying the Configuration

Modify `configs/default.yaml` to specify target languages, model dimensions, training hyperparameters, or test prompts:

```yaml
language:
  target_language: "Hindi"
  dataset_name: "ai4bharat/samanantar"
  tgt_lang: "hi"
  lang_pair: "EN-HI"

model:
  d_model: 512
  num_layers: 6
  num_heads: 8
  d_ff: 2048
  dropout: 0.1

training:
  epochs: 10
  batch_size: 224
  lr: 5.0e-4

```