# Hindi-Marathi Neural Machine Translation

## Overview
Brief description of your work

## Hardware Used
- GPU: [Specify exact model, e.g., Tesla T4, RTX 3090]
- VRAM: [e.g., 16GB]
- Training time: [Approximate hours for each model]

## LLM Usage Declaration
- Used ChatGPT/Claude for: [Be specific - code debugging, understanding RoPE, etc.]
- All architectural decisions justified independently

## Setup Instructions
```bash
pip install -r requirements.txt
```

## Data Preparation
```bash
python data/preprocessing.py --input_path ... --output_path ...
```

## Training

### Part I: LSTM Seq2Seq
```bash
# Random embeddings
python training/train_lstm.py --config configs/lstm_random_emb.yaml

# BERT embeddings
python training/train_lstm.py --config configs/lstm_bert_emb.yaml
```

### Part II: Pretraining
```bash
# BERT pretraining (110M params)
python training/train_bert.py --config configs/bert_pretrain.yaml

# GPT-2 pretraining (124M params)
python training/train_gpt2.py --config configs/gpt2_pretrain.yaml
```

## Evaluation
```bash
python evaluation/evaluate.py --checkpoint path/to/checkpoint
```

## Results Summary
| Model | BLEU-100 | CHRF++-100 |
|-------|----------|------------|
| LSTM Random | XX.X | XX.X |
| LSTM BERT | XX.X | XX.X |
| Pretrained | XX.X | XX.X |

## Checkpoints
Due to size constraints, model checkpoints are available at: [Google Drive/Hugging Face link]
