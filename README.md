# 🌐 AdiVaani NMT System
### *Bridging Hindi ↔ Marathi — Classical Neural Machine Translation*

> **MISN Lab · IIT Delhi | AdiVaani Initiative**



```

---

## Table of Contents

- [Overview](#-overview)
- [Repository Structure](#-repository-structure)
- [Setup & Installation](#-setup--installation)
- [Dataset](#-dataset)
- [Notebook 1 — Random Embeddings](#-notebook-1--random-embeddings-notebookeff294ffad__1_ipynb)
- [Notebook 2 — BERT Embeddings](#-notebook-2--bert-embeddings-bertipynb)
- [Experiments & Results](#-experiments--results)
- [Report](#-report)
- [Reproducibility](#-reproducibility)
- [LLM Usage Disclosure](#-llm-usage-disclosure)
- [Hardware](#-hardware)
- [References](#-references)

---

##  Overview

This repository is the complete implementation for **Part I** of the AdiVaani Hiring Assessment at MISN Lab, IIT Delhi — Classical Neural Machine Translation for Hindi ↔ Marathi using recurrent architectures.

The two notebooks run **in parallel** (designed for T4×2 GPU), each implementing the same LSTM Seq2Seq + Bahdanau Attention framework under a different embedding strategy:

| Notebook | Embedding Strategy | Est. Runtime |

| `notebookeff294ffad__1_.ipynb` | Randomly initialized embeddings (shared BPE vocab) | ~4–5 hours |
| `Bert.ipynb` | Pretrained BERT-based embeddings (Hindi + Marathi BERT) | ~4–5 hours |

Both are evaluated with **BLEU-100** and **CHRF++-100** and produce train/val loss, BLEU, and CHRF++ plots.

> **Prohibited Models**: mT5, NLLB-200, MarianMT, M2M-100, and any off-the-shelf MT framework are strictly **not used**.

---
##  Repository Structure

```
adivaani-nmt/
│
├── notebookeff294ffad__1_.ipynb     # Part I — Experiment A: Random Embeddings
│   │
│   ├── Cell 1  — Install dependencies
│   ├── Cell 2  — Imports & seed setup
│   ├── Cell 3  — Configuration (all hyperparameters in one place)
│   ├── Cell 4  — Load & inspect data
│   ├── Cell 5  — Train SentencePiece (shared BPE, both Hindi + Marathi)
│   ├── Cell 6  — Dataset & DataLoader
│   ├── Cell 7  — Model: Bahdanau Attention + Seq2Seq LSTM
│   ├── Cell 8  — Label-smoothed cross-entropy + optimizer + scheduler
│   ├── Cell 9  — BLEU & CHRF++ evaluation helpers
│   ├── Cell 10 — Training loop
│   ├── Cell 11 — Plots (loss, BLEU-100, CHRF++-100)
│   ├── Cell 12 — Final evaluation on best checkpoint + qualitative examples
│   └── Cell 13 — Hindi→Marathi & Marathi→Hindi inference demo
│
├── Bert.ipynb                       # Part I — Experiment B: BERT Embeddings
│   │
│   ├── Cell 1  — Install dependencies
│   ├── Cell 2  — Imports & seed setup
│   ├── Cell 3  — Configuration (BERT-specific hyperparameters)
│   ├── Cell 4  — Load BERT tokenizers & models (Hindi + Marathi BERT)
│   ├── Cell 5  — Load data
│   ├── Cell 6  — Train SentencePiece for TARGET side (Marathi output)
│   ├── Cell 7  — Pre-cache BERT embeddings (critical for speed)
│   ├── Cell 8  — Dataset & DataLoader (BERT embeddings as input)
│   ├── Cell 9  — Model: BERT-projected LSTM Seq2Seq
│   ├── Cell 10 — Loss, optimizer, scheduler
│   ├── Cell 11 — Evaluation helpers
│   ├── Cell 12 — Training loop
│   ├── Cell 13 — Plots
│   ├── Cell 14 — Final evaluation + save results
│   └── Cell 15 — Inference demo
│
├── data/
│   └── raw/                         # Place downloaded corpus here (train.hi / train.mr)
│
├── checkpoints/                     # Saved during training (best_random.pt, best_bert.pt)
│
├── plots/                           # Exported training curves from both notebooks
│
├── report/
│   └── technical_report.pdf         # Full technical report
│
├── requirements.txt
├── environment.yaml
└── README.md
```

---

##  Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/adivaani-nmt.git
cd adivaani-nmt
```

### 2. Create Environment

**Using Conda (recommended):**
```bash
conda env create -f environment.yaml
conda activate adivaani-nmt
```

**Using pip:**
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Key Dependencies

| Package | Purpose |
|---------|---------|
| `torch >= 2.0` | Core deep learning framework |
| `transformers` | Hindi & Marathi BERT tokenizers and models |
| `sentencepiece` | Shared BPE tokenizer training |
| `sacrebleu` | BLEU-100 & CHRF++ evaluation |
| `matplotlib` | Training curve plots |
| `tqdm` | Progress bars |
| `accelerate` | Mixed precision (AMP) support |

### 4. Launch Notebooks

```bash
jupyter notebook
```

Open both notebooks side by side and run each top-to-bottom. They are designed to run **in parallel** on separate GPUs.

---

##  Dataset

**Hindi–Marathi Parallel Corpus** — [Download here](https://shorturl.at/9Zkkn)

Place `train.hi` and `train.mr` inside `data/raw/`. Both notebooks auto-detect the data path via a glob search (Cell 3/4 respectively) — update `CFG['data_dir']` manually if auto-detection fails.

---

## Notebook 1 — Random Embeddings (`notebookeff294ffad__1_.ipynb`)

This notebook implements the **baseline** experiment: a full LSTM Seq2Seq + Bahdanau Attention system where both source (Hindi) and target (Marathi) use **randomly initialized, fully trainable** embeddings trained from scratch alongside the model.

### Architecture

```
Source (Hindi)                         Target (Marathi)
    │                                       │
[Random Embedding (256-dim)]        [Random Embedding (256-dim)]
    │                                       │
[BiLSTM Encoder × 2]                [LSTM Decoder × 2]
    │                                       │
[Hidden States h₁…hₙ]  ──────►    [Bahdanau Attention]
                                           │
                                    [Context Vector cₜ]
                                           │
                                  [Linear Projection]
                                           │
                                   [Softmax over Vocab]
```

### Cell-by-Cell Walkthrough

**Cell 1 — Install:** Installs `transformers`, `sentencepiece`, `sacrebleu`, `matplotlib`, `tqdm`, `accelerate` via pip.

**Cell 2 — Imports:** Sets global seed (`SEED=42`) across `random`, `numpy`, `torch`. Detects GPU count and prints device info.

**Cell 3 — Configuration:** Single `CFG` dict controlling everything:

```python
CFG = {
    'max_samples': 500_000,     # random subset of corpus
    'max_len': 60,              # token-level length filter
    'val_split': 0.05,
    'vocab_size': 16_000,       # shared BPE vocab
    'emb_dim': 256,
    'hidden_dim': 512,
    'n_layers': 2,
    'dropout': 0.3,
    'batch_size': 256,
    'epochs': 30,
    'lr': 1e-3,
    'clip_grad': 1.0,
    'label_smooth': 0.1,
    'beam_width': 4,
    'len_penalty': 0.6,
    ...
}
```

**Cell 4 — Load & inspect data:** Reads `train.hi` / `train.mr`, filters by length, random-samples up to `max_samples`, splits into train/val.

**Cell 5 — SentencePiece:** Trains a **shared BPE tokenizer** (16K vocab) over both Hindi and Marathi. Shared vocab encourages subword overlap between the two closely related Devanagari-script languages.

**Cell 6 — Dataset & DataLoader:** `NMTDataset` class returning padded tensors. Collate function handles variable-length batching with `pack_padded_sequence`.

**Cell 7 — Model:** Defines three classes:
- `BahdanauAttention` — additive attention: `eᵢⱼ = vₐᵀ·tanh(Wₐ·sᵢ + Uₐ·hⱼ)`
- `Encoder` — 2-layer BiLSTM, projects bidirectional hidden/cell states for decoder init
- `Seq2Seq` — ties encoder and decoder, handles teacher forcing + beam search

**Cell 8 — Loss / optimizer / scheduler:** Label-smoothed cross-entropy (ε=0.1), Adam optimizer, linear warmup + cosine decay scheduler.

**Cell 9 — Evaluation helpers:** `evaluate_bleu_chrf()` runs beam search on a subset and computes `sacrebleu` BLEU-100 and CHRF++ scores.

**Cell 10 — Training loop:** Full epoch loop with AMP (`autocast` + `GradScaler`), `DataParallel` across available GPUs, gradient clipping, best-checkpoint saving.

**Cell 11 — Plots:** Inline `matplotlib` plots of train/val loss, BLEU-100, CHRF++-100 per epoch. Saved to `plots/`.

**Cell 12 — Final evaluation:** Loads best checkpoint, runs full test-set BLEU-100 + CHRF++ evaluation, prints 10 qualitative translation examples with source, reference, and hypothesis.

**Cell 13 — Inference demo:** Interactive Hindi→Marathi and Marathi→Hindi beam search decoding on custom input sentences.

---

##  Notebook 2 — BERT Embeddings (`Bert.ipynb`)

This notebook implements the **BERT embedding** experiment: same LSTM Seq2Seq + Bahdanau Attention architecture, but the **source side (Hindi) uses BERT contextual embeddings** and the **target side (Marathi) uses BERT embeddings for output vocabulary**, with a trainable projection layer bridging BERT's 768-dim space to the LSTM's 256-dim input.

Designed to run **in parallel** with Notebook 1 on a second GPU.

### Architecture

```
Source (Hindi)
    │
[Hindi BERT — l3cube-pune/hindi-bert-v2]   ← frozen or fine-tunable
    │  (768-dim, last 4 layers averaged)
[Linear Projection → 256-dim]
    │
[BiLSTM Encoder × 2]
    │
[Hidden States]  ──────►   [LSTM Decoder × 2]
                                   │
                            [Bahdanau Attention]
                                   │
                             [Context Vector]
                                   │
                    [SentencePiece BPE for target (Marathi)]
                                   │
                             [Softmax over Vocab]
```

### Cell-by-Cell Walkthrough

**Cell 1 — Install:** Same packages as Notebook 1.

**Cell 2 — Imports:** Seeds set identically (`SEED=42`). Detects GPU count.

**Cell 3 — Configuration:** BERT-specific `CFG`:

```python
CFG = {
    'max_samples': 200_000,     # smaller: BERT encoding is slower
    'max_bert_len': 64,         # max BERT tokens (source side)
    'hi_bert': 'l3cube-pune/hindi-bert-v2',
    'mr_bert': 'l3cube-pune/marathi-bert-v2',
    'bert_freeze': True,        # True = frozen (fast); False = fine-tune
    'bert_layers': 4,           # average last 4 hidden layers
    'proj_dim': 256,            # BERT 768 → projected 256 → LSTM
    'hidden_dim': 512,
    'vocab_size': 8000,         # target-side SPM vocab (Marathi only)
    'batch_size': 128,
    'epochs': 20,
    'lr': 1e-3,
    'bert_lr': 2e-5,            # separate LR for BERT when unfrozen
    ...
}
```

**Cell 4 — Load BERT models:** Loads `hindi-bert-v2` and `marathi-bert-v2` via HuggingFace. Auto-detects hidden size (768). Freezes or unfreezes BERT weights based on `bert_freeze` flag.

**Cell 5 — Load data:** Same parallel corpus loading and train/val splitting as Notebook 1.

**Cell 6 — SentencePiece (target only):** Unlike Notebook 1's shared vocab, here SPM is trained only on **Marathi** (target side). Source tokens are handled entirely by the BERT tokenizer.

**Cell 7 — Pre-cache BERT embeddings:** Critical optimization — BERT forward passes are run **once** over the entire dataset and cached to CPU tensors. Avoids re-running BERT every epoch, saving ~5× training time. Averages the last 4 hidden layers per token, strips `[CLS]` and `[SEP]`.

**Cell 8 — Dataset & DataLoader:** `BERTNMTDataset` takes pre-cached embedding tensors (not raw strings) as source input. Custom collate function pads variable-length embedding sequences.

**Cell 9 — Model:** Same `BahdanauAttention` and `Seq2Seq` structure as Notebook 1, but the `Encoder` now accepts pre-computed embedding tensors through a `Linear(bert_hidden → proj_dim)` projection layer before the BiLSTM.

**Cell 10 — Loss / optimizer / scheduler:** Same label-smoothed cross-entropy setup. When `bert_freeze=False`, BERT parameters use a separate lower learning rate (`bert_lr=2e-5`).

**Cell 11 — Evaluation helpers:** Same `sacrebleu` BLEU-100 + CHRF++ evaluation, adapted for BERT-embedding inputs.

**Cell 12 — Training loop:** Same AMP + DataParallel loop. Checkpoint saved to `best_bert.pt`.

**Cell 13 — Plots:** Inline train/val loss, BLEU-100, CHRF++-100 curves. Saved to `plots/`.

**Cell 14 — Final evaluation + save results:** Loads best checkpoint, full test-set evaluation, prints qualitative translation examples for direct comparison against Notebook 1 output.

**Cell 15 — Inference demo:** Beam search inference on custom inputs.

---

## Experiments & Results

### Comparison Summary

| Experiment | Embedding | Vocab Strategy | BLEU-100 (Val) | CHRF++-100 (Val) |

| Notebook 1 | Random (256-dim) | Shared BPE (16K, Hi+Mr) | *see plots* | *see plots* |
| Notebook 2 — Frozen BERT | BERT frozen (768→256) | SPM target-only (8K) | *see plots* | *see plots* |
| Notebook 2 — Tuned BERT | BERT fine-tuned (768→256) | SPM target-only (8K) | *see plots* | *see plots* |

### Training Curves

All plots are exported inline in each notebook and saved to `plots/`:

```
plots/
├── random_loss.png          # Notebook 1: train/val loss
├── random_bleu.png          # Notebook 1: train/val BLEU-100
├── random_chrf.png          # Notebook 1: train/val CHRF++-100
├── bert_loss.png            # Notebook 2: train/val loss
├── bert_bleu.png            # Notebook 2: train/val BLEU-100
└── bert_chrf.png            # Notebook 2: train/val CHRF++-100
```

---

##  Report

The full **technical report** is available at [`report/technical_report.pdf`](report/technical_report.pdf).

**Contents:**

1. Dataset processing methodology
2. Tokenization strategy — shared BPE vs. BERT tokenizer + target-only SPM
3. Architecture design — BiLSTM encoder, Bahdanau attention, beam search
4. Optimization — label smoothing, warmup scheduling, AMP, DataParallel
5. Hyperparameter choices and ablation studies
6. Comparative analysis: random vs. BERT embeddings (convergence, BLEU, CHRF++, qualitative)
7. Failure analysis and training observations
8. Qualitative translation examples
9. Hardware and compute constraints
10. LLM usage documentation

---

##  Reproducibility

1. Download the corpus and place `train.hi` + `train.mr` in `data/raw/`
2. Install dependencies via `environment.yaml` or `requirements.txt`
3. **Run `notebookeff294ffad__1_.ipynb`** top-to-bottom → reproduces random embedding experiment
4. **Run `Bert.ipynb`** top-to-bottom → reproduces BERT embedding experiment

Both notebooks are designed to run simultaneously on separate GPUs. All random seeds are fixed identically in Cell 2 of each:

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

Checkpoints are saved automatically to `checkpoints/` during training.

---

##  LLM Usage Disclosure

In compliance with the assessment guidelines, all LLM-assisted components are documented:

| Component | LLM Used | Usage Description |
|---|---|---|
| BERT embedding caching pipeline | Claude | Initial scaffold for `cache_embeddings()`; reviewed and adapted |
| Beam search decoder | Claude | Draft implementation; debugged and validated independently |
| SentencePiece integration | ChatGPT | API usage reference; logic verified manually |
| AMP + DataParallel training loop | ChatGPT | Boilerplate structure; modified for this codebase |

> All code was critically reviewed, tested, and is independently defensible. No generated code was used without full understanding.

---

## Hardware

| Spec | Details |
|---|---|
| GPU | *[e.g., 2× NVIDIA T4 — fill in your actual hardware]* |
| CUDA Version | *[e.g., 12.1]* |
| RAM | *[e.g., 16 GB]* |
| Training Time (Notebook 1) | *[e.g., ~4–5 hours]* |
| Training Time (Notebook 2) | *[e.g., ~4–5 hours]* |

---

##  References

1. Bahdanau, D., Cho, K., & Bengio, Y. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR 2015.
2. Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS 2017.
3. Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL 2019.
4. L3Cube Pune. *Hindi BERT v2.* [`l3cube-pune/hindi-bert-v2`](https://huggingface.co/l3cube-pune/hindi-bert-v2)
5. L3Cube Pune. *Marathi BERT v2.* [`l3cube-pune/marathi-bert-v2`](https://huggingface.co/l3cube-pune/marathi-bert-v2)
6. Post, M. (2018). *A Call for Clarity in Reporting BLEU Scores.* WMT 2018. [`sacrebleu`]

---

<div align="center">

**Built for the AdiVaani Initiative · MISN Lab, IIT Delhi**

*Low-resource NLP for Indian languages — because every language deserves a voice.*

</div>
