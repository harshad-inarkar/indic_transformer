# Indic-Transformer

High-performance, from-scratch Sequence-to-Sequence Transformer (Vaswani et al., 2017) built with PyTorch, optimized exclusively for CUDA hardware. Supports English to 12+ Indic languages via Hugging Face parallel corpora (`ai4bharat/samanantar`, `cfilt/iitb-english-hindi`, `acomquest/Saamayik`).

---

## Architecture & Features

- **Strict OOP Modular Design:** Separation of concerns between Data, Tokenization, Architecture, Scheduling, and Inference.
- **CUDA Optimized:** Uses `torch.amp.autocast`, fused AdamW, non-blocking page-locked host-to-device transfers, and numerical underflow guardrails.
- **Vectorized Generation:** Batched greedy decoding and batched beam search decoding with cross-attention state caching.
- **Config-Driven:** Fully parameterizable execution via YAML (`configs/default.yaml`).

---

## Installation

```bash
git clone [https://github.com/](https://github.com/)<your-username>/indic-transformer.git
cd indic-transformer
pip install -e .