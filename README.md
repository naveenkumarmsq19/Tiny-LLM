# Tiny-LLM

A minimal **character-level, decoder-only Transformer** built with Python and PyTorch. This is an educational model trained **from scratch**, not a pretrained chatbot or a production-scale LLM.

## Requirements

- Python 3.10+
- PyTorch 2.3+

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train.py --steps 300
python generate.py --prompt "tiny" --max-new-tokens 160
```

Training uses `data/sample.txt` by default. For more meaningful outputs, replace it with a larger text corpus you have permission to use, then retrain. The trained checkpoint is saved at `checkpoints/tinyllm.pt` (ignored by Git).

## Configuration

```bash
python train.py --data data/sample.txt --steps 1000 --context-size 64 --batch-size 16 --embed-dim 128 --heads 4 --layers 2
python generate.py --checkpoint checkpoints/tinyllm.pt --prompt "the " --temperature 0.8
```

Prompts can only contain characters seen in the training data. Start with a short prompt and expect mostly random-looking output after a small amount of training.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Structure

- `tiny_llm/model.py`: causal Transformer
- `tiny_llm/tokenizer.py`: character-level tokenizer
- `train.py`: training and checkpoint saving
- `generate.py`: text generation from a checkpoint
- `data/sample.txt`: small original practice corpus
- `tests/`: basic correctness tests
