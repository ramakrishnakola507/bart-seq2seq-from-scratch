# 🤖 BART — Denoising Sequence-to-Sequence Transformer (From Scratch)

> A from-scratch PyTorch implementation of **BART** (Bidirectional and Auto-Regressive Transformers), the state-of-the-art encoder-decoder LLM for natural language generation, translation, and comprehension — as described in the paper by Lewis et al. (2019).

---

## 🧠 What Is BART?

BART is a denoising autoencoder for pretraining sequence-to-sequence models. It:

1. **Corrupts** input text using a noising function (masking, deletion, permutation, etc.)
2. **Learns** to reconstruct the original text via a bidirectional encoder + autoregressive decoder

It generalises both **BERT** (bidirectional encoder) and **GPT** (left-to-right decoder) into a single unified architecture, achieving state-of-the-art results on:

| Benchmark | Score |
|-----------|-------|
| SQuAD 1.1 F1 | **90.8** |
| MNLI Accuracy | **84.1** |
| CNN/DM Perplexity | **5.41** |

---

## 🏗️ Architecture Overview

```
Input (Corrupted Text)
        │
        ▼
┌───────────────────────┐
│      BART Encoder      │   ← 6 layers of bidirectional self-attention
│  (Bidirectional BERT)  │   ← Learned Positional Embeddings
└───────────┬───────────┘
            │  encoder_hidden_states
            ▼
┌───────────────────────┐
│      BART Decoder      │   ← 6 layers of masked self-attention
│  (Autoregressive GPT) │   ← Cross-attention over encoder output
└───────────┬───────────┘
            │
            ▼
        LM Head (tied weights)
            │
            ▼
    Reconstructed Text
```

### Key components implemented:

| Component | Description |
|-----------|-------------|
| `BARTConfig` | Dataclass-based hyperparameter configuration |
| `LearnedPositionalEmbedding` | Position-aware token embeddings (not sinusoidal) |
| `MultiHeadAttention` | Scaled dot-product attention with `n_heads` |
| `PositionwiseFeedForward` | Two-layer FFN with GELU activation |
| `EncoderLayer` | Self-attention + FFN + LayerNorm + residuals |
| `DecoderLayer` | Masked self-attention + cross-attention + FFN |
| `BARTEncoder` | Stack of 6 encoder layers |
| `BARTDecoder` | Stack of 6 decoder layers |
| `BARTModel` | Full seq2seq model with tied LM head weights |

---

## ⚙️ Model Configuration

```python
BARTConfig(
    vocab_size        = 30000,
    d_model           = 512,      # Hidden dimension
    encoder_layers    = 6,        # Encoder depth
    decoder_layers    = 6,        # Decoder depth
    n_heads           = 8,        # Attention heads
    d_ff              = 2048,     # Feed-forward dimension
    dropout           = 0.1,
    max_position_embeddings = 512,
    pad_token_id      = 0,
)
```

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch
```

### Run the model

```python
import torch
import torch.nn.functional as F
from bart import BARTConfig, BARTModel

# Initialise config
config = BARTConfig(
    vocab_size=100,        # Use 30000 for real training
    d_model=256,
    encoder_layers=3,
    decoder_layers=3,
    n_heads=8,
    d_ff=1024,
    max_position_embeddings=128,
)

# Build model
model = BARTModel(config)
model.train()

# Dummy forward pass
batch_size, src_len, tgt_len = 2, 10, 12
src     = torch.randint(4, config.vocab_size, (batch_size, src_len))
tgt_inp = torch.randint(4, config.vocab_size, (batch_size, tgt_len))
tgt_out = torch.randint(4, config.vocab_size, (batch_size, tgt_len))

src_pad_mask = src.eq(config.pad_token_id)
tgt_pad_mask = tgt_inp.eq(config.pad_token_id)

# Forward + loss
logits = model(src, tgt_inp,
               src_key_padding_mask=src_pad_mask,
               tgt_key_padding_mask=tgt_pad_mask)

loss = F.cross_entropy(
    logits.view(-1, config.vocab_size),
    tgt_out.view(-1),
    ignore_index=config.pad_token_id
)

print("Logits shape:", logits.shape)   # (batch, tgt_len, vocab_size)
print("Loss:", loss.item())
```

---

## 📁 Repository Structure

```
bart-seq2seq-from-scratch/
│
├── bart.ipynb                  # Full implementation notebook (all components)
├── mini_bart.py                # Minimal standalone implementation
├── pretrain_bart.py            # Pre-training loop with noising schemes
├── README.md                   # This file
└── requirements.txt            # torch
```

---

## 🔧 Pre-Training Noising Schemes

BART is pre-trained by learning to reconstruct text from corrupted inputs. Five noising strategies are implemented:

| Scheme | Description |
|--------|-------------|
| **Token Masking** | Random tokens replaced with `[MASK]` (like BERT) |
| **Token Deletion** | Tokens removed; model must infer missing positions |
| **Text Infilling** | Spans replaced with single `[MASK]`; model predicts span length |
| **Sentence Permutation** | Document sentences shuffled randomly |
| **Document Rotation** | Document rotated to start at a random token |

> Best performance: **Text Infilling + Sentence Permutation** combined (SQuAD F1: 90.8, CNN/DM PPL: 5.41)

---

## 📊 Training Objective

```
loss = CrossEntropy(decoder_output, original_text)
```

During pre-training, the model minimises the **negative log-likelihood** of the original (uncorrupted) document given the corrupted input. The decoder is only exposed to uncorrupted context during pre-training, minimising the gap between pre-training and generation tasks.

---

## 🔬 Research Paper

This implementation is based on the original BART paper and supported by a co-authored research report submitted at SRM University AP:

> **"BART: A Denoising Sequence-to-Sequence Pre-training Method for Natural Language Generation, Translation, and Comprehension"**  
> Lewis et al., 2019 — [arXiv:1910.13461](https://arxiv.org/abs/1910.13461)  
>
> *Co-authored report:* Dr. Varsha Santosh Sambhaje, Akhilesh Vallabhaneni, Shrijal Shrestha, Govind Mohan Shah, **Kola Ramakrishna**, Devesh Chaudhary — SRM University AP, 2025

---

## 🧪 Results (Reproduced from Paper)

### BART Pre-training Objectives Comparison

| Variant | SQuAD F1 | MNLI Acc | CNN/DM PPL |
|---------|----------|----------|------------|
| Token Masking | 90.4 | 84.1 | 6.10 |
| Token Deletion | 90.4 | 84.1 | 5.87 |
| **Text Infilling** | **90.8** | **84.0** | 5.83 |
| Document Rotation | 77.2 | 75.3 | 10.59 |
| Sentence Shuffle | 85.4 | 81.5 | 7.89 |
| **Text Infill + Shuffle** | **90.8** | 83.8 | **5.41** |

---

## 🛠️ Technical Highlights

- **Tied LM head weights** — `lm_head.weight` shares parameters with `decoder.embed_tokens.weight`, reducing model size and improving training stability
- **Causal masking** — decoder self-attention uses autoregressive masking to prevent attending to future tokens
- **Padding-aware attention** — all attention layers support `key_padding_mask` for variable-length batches
- **Teacher forcing** — training uses ground-truth decoder inputs (shifted right: `BOS + target[:-1]`)
- **GeLU activations** — FFN layers use GeLU (vs ReLU in vanilla Transformer) following the BART paper

---

## 📚 Concepts Covered

If you're learning from this repo, you'll gain hands-on understanding of:

- [ ] Transformer encoder-decoder architecture
- [ ] Multi-head scaled dot-product attention
- [ ] Cross-attention between encoder and decoder
- [ ] Learned vs sinusoidal positional embeddings
- [ ] Denoising pre-training strategies
- [ ] Tied embedding weights
- [ ] Autoregressive decoding with causal masking
- [ ] Sequence-to-sequence training with teacher forcing
- [ ] Padding masks and batch processing

---

## 🔮 Future Work

- [ ] Add tokeniser (BPE/SentencePiece) for real text input
- [ ] Implement greedy and beam search decoding
- [ ] Add fine-tuning scripts for summarisation (CNN/DM) and QA (SQuAD)
- [ ] Train on a real corpus with all 5 noising schemes
- [ ] Add HuggingFace-compatible model export

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

## 🙋 Author

**Kola Ramakrishna**  
B.Tech CSE (AI/ML) — SRM University Andhra Pradesh  
[GitHub](https://github.com/ramakrishnakola507) · [LinkedIn](https://linkedin.com/in/ramakrishna-kola) · [Email](mailto:ramakrishna_kola@srmap.edu.in)

---

> ⭐ If this helped you understand BART or transformer architecture, consider starring the repo!
