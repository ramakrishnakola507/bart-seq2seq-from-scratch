# 🤖 BART — Denoising Seq2Seq Transformer (From Scratch in PyTorch)

> A clean, from-scratch PyTorch implementation of **BART** (Bidirectional and Auto-Regressive Transformers) — the encoder-decoder LLM for natural language generation, translation, and comprehension — based on Lewis et al. (2019).
>
> Built as part of a co-authored research paper at SRM University AP.

---

## 🧠 What Is BART?

BART is a denoising autoencoder for pretraining sequence-to-sequence models:

1. **Corrupt** input text with a noising function
2. **Reconstruct** the original via a bidirectional encoder + autoregressive decoder

It unifies **BERT** (bidirectional encoder) and **GPT** (left-to-right decoder) into one architecture, achieving state-of-the-art results:

| Benchmark | Score |
|-----------|-------|
| SQuAD 1.1 F1 | **90.8** |
| MNLI Accuracy | **84.1** |
| CNN/DM Perplexity (Text Infill + Shuffle) | **5.41** |

---

## 🏗️ Architecture

```
Input (Corrupted Tokens)
         │
         ▼
┌─────────────────────────┐
│       BARTEncoder        │  ← 6 × EncoderLayer
│  token embed + pos embed │  ← LearnedPositionalEmbedding
│  → self-attention + FFN  │  ← MultiHeadAttention + GeLU FFN
│  → LayerNorm + residual  │
└────────────┬────────────┘
             │  encoder_out  (batch, src_len, d_model)
             ▼
┌─────────────────────────┐
│       BARTDecoder        │  ← 6 × DecoderLayer
│  → masked self-attention │  ← causal mask (autoregressive)
│  → cross-attention       │  ← attends to encoder_out
│  → FFN + LayerNorm       │
└────────────┬────────────┘
             │
             ▼
       LM Head (Linear)       ← weights tied to decoder embeddings
             │
             ▼
    Output Logits (batch, tgt_len, vocab_size)
```

---

## 📁 Repository Structure

```
bart-seq2seq-from-scratch/
│
├── bart.ipynb          ← Full implementation (all components + demo)
└── README.md           ← This file
```

> **Note:** `pretrain_bart.py` and `mini_bart.py` are placeholders for future additions — all working code lives in `bart.ipynb`.

---

## ⚙️ Model Configuration

```python
@dataclass
class BARTConfig:
    vocab_size: int = 30000
    d_model: int = 512        # Hidden size
    encoder_layers: int = 6   # Encoder depth
    decoder_layers: int = 6   # Decoder depth
    n_heads: int = 8          # Attention heads
    d_ff: int = 2048          # FFN intermediate size
    dropout: float = 0.1
    max_position_embeddings: int = 512
    pad_token_id: int = 0
```

---

## 🧩 Components Implemented

| Class | Description |
|-------|-------------|
| `BARTConfig` | Dataclass for all model hyperparameters |
| `LearnedPositionalEmbedding` | Trainable position embeddings via `nn.Embedding` |
| `MultiHeadAttention` | Scaled dot-product attention with Q/K/V projections, causal + padding masks |
| `PositionwiseFeedForward` | Two-layer FFN with **GeLU** activation |
| `EncoderLayer` | Self-attention → LayerNorm → FFN → LayerNorm (residuals throughout) |
| `DecoderLayer` | Masked self-attn → cross-attn → FFN (all with LayerNorm + residuals) |
| `BARTEncoder` | Token embed + positional embed + N encoder layers + final LayerNorm |
| `BARTDecoder` | Token embed + positional embed + N decoder layers + final LayerNorm |
| `BARTModel` | Full seq2seq: encoder + decoder + tied LM head + `generate()` |

---

## 🚀 Quick Start

### Install dependency

```bash
pip install torch
```

### Training forward pass

```python
import torch
import torch.nn.functional as F
from bart import BARTConfig, BARTModel  # or run directly in notebook

config = BARTConfig(
    vocab_size=100,       # toy vocab; use 30000 for real training
    d_model=256,
    encoder_layers=3,
    decoder_layers=3,
    n_heads=8,
    d_ff=1024,
    max_position_embeddings=128,
)

model = BARTModel(config)
model.train()

# Dummy batch
src     = torch.randint(4, config.vocab_size, (2, 10))   # (batch, src_len)
tgt_inp = torch.randint(4, config.vocab_size, (2, 12))   # (batch, tgt_len) — shifted right
tgt_out = torch.randint(4, config.vocab_size, (2, 12))   # (batch, tgt_len) — targets

src_pad_mask = src.eq(config.pad_token_id)
tgt_pad_mask = tgt_inp.eq(config.pad_token_id)

logits = model(src, tgt_inp,
               src_key_padding_mask=src_pad_mask,
               tgt_key_padding_mask=tgt_pad_mask)

loss = F.cross_entropy(
    logits.view(-1, config.vocab_size),
    tgt_out.view(-1),
    ignore_index=config.pad_token_id
)

print("Logits shape:", logits.shape)   # → (2, 12, 100)
print("Loss:", loss.item())
```

### Greedy decoding (inference)

```python
model.eval()
with torch.no_grad():
    generated_ids = model.generate(
        src,
        src_key_padding_mask=src_pad_mask,
        max_len=50,
        bos_token_id=2,
        eos_token_id=3,
    )
print("Generated token IDs:", generated_ids)
```

---

## 🔧 Pre-Training Noising Schemes

BART is pre-trained by corrupting input and learning to reconstruct the original. Five strategies from the paper:

| Scheme | Description | SQuAD F1 | CNN/DM PPL |
|--------|-------------|----------|------------|
| Token Masking | Random tokens → `[MASK]` (BERT-style) | 90.4 | 6.10 |
| Token Deletion | Tokens removed; model infers positions | 90.4 | 5.87 |
| **Text Infilling** | Spans → single `[MASK]`; predict span length | **90.8** | 5.83 |
| Document Rotation | Rotate to start at random token | 77.2 | 10.59 |
| Sentence Permutation | Sentences shuffled | 85.4 | 7.89 |
| **Text Infill + Shuffle** | Best combination | **90.8** | **5.41** |

---

## 🛠️ Technical Highlights

- **Tied LM head** — `lm_head.weight` shares parameters with `decoder.embed_tokens.weight`, reducing model size and improving gradient flow
- **Causal masking** — decoder self-attention uses upper-triangular mask via `torch.triu` to prevent future token leakage
- **Padding-aware attention** — all attention layers accept `key_padding_mask` for proper batched training
- **Teacher forcing** — training uses ground-truth decoder inputs shifted right (`BOS + target[:-1]`)
- **GeLU activations** — FFN uses GeLU (not ReLU) following the BART paper for better gradient behaviour
- **Greedy decoding** — `model.generate()` implements autoregressive greedy search with EOS stopping

---

## 📚 Concepts You'll Learn From This Code

- [x] Transformer encoder-decoder architecture end-to-end
- [x] Multi-head scaled dot-product attention (from scratch)
- [x] Cross-attention between encoder and decoder
- [x] Learned positional embeddings vs sinusoidal
- [x] Causal masking for autoregressive decoding
- [x] Tied embedding weights
- [x] Padding masks for variable-length batches
- [x] Teacher forcing during sequence-to-sequence training
- [x] Greedy decoding with EOS stopping

---

## 🔬 Research Paper

This implementation is based on the original BART paper, supported by a co-authored research report:

> **"BART: A Denoising Sequence-to-Sequence Pre-training Method for Natural Language Generation, Translation, and Comprehension"**  
> Lewis et al., 2019 — [arXiv:1910.13461](https://arxiv.org/abs/1910.13461)

**Co-authored research report submitted at SRM University Andhra Pradesh (2025):**  
Dr. Varsha Santosh Sambhaje, Akhilesh Vallabhaneni, Shrijal Shrestha, Govind Mohan Shah, **Kola Ramakrishna**, Devesh Chaudhary

---

## 🔮 Planned Additions

- [ ] `mini_bart.py` — minimal single-file implementation (no notebook dependency)
- [ ] `pretrain_bart.py` — full pre-training loop with all 5 noising schemes
- [ ] Tokeniser integration (BPE via HuggingFace tokenizers)
- [ ] Beam search decoding
- [ ] Fine-tuning script for summarisation (CNN/DailyMail)

---

## 📄 License

MIT License — free to use, modify, and build upon with attribution.

---

## 🙋 Author

**Kola Ramakrishna**  
B.Tech CSE (AI/ML) — SRM University Andhra Pradesh  
[GitHub](https://github.com/ramakrishnakola507) · [LinkedIn](https://linkedin.com/in/ramakrishna-kola) · [Email](mailto:ramakrishna_kola@srmap.edu.in)

---

> ⭐ If this helped you understand BART or Transformer architecture from the ground up, consider starring the repo!
