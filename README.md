# Tiny-LLM

A small **decoder-only Transformer** written with Python + PyTorch. The neural model starts from **random weights**. No pretrained model, external AI API, fine-tuned checkpoint, or paid service is required.

The repository now supports **company-specific question answering** as an educational experiment, plus the original plain-text language-model training.

> **Reality check:** A tiny character-level model trained from scratch on a handful of company FAQs will usually give poor or incorrect answers to unseen questions. More training examples and training steps can help it memorize known phrasing, but won't make it comparable to ChatGPT. A separate **grounded FAQ mode** gives reliable answers to matched questions by returning the supplied answer *directly from the JSON*, not from the neural network. The two modes are never represented as the same thing.

## 1. Install

Requirements: Python 3.10+ and PyTorch 2.3+.

```bash
# Windows PowerShell:
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# macOS / Linux:
# python3 -m venv .venv
# source .venv/bin/activate
# pip install -r requirements.txt
```

## 2. Feed your company details

Copy `data/company.example.json` to `data/company.json` and replace its **demo values** with your real company facts. Each fact has one authoritative answer and several ways a customer might ask for it:

```json
{
  "name": "Your Company",
  "facts": [
    {
      "key": "services",
      "answer": "We create websites and mobile applications.",
      "questions": [
        "What services do you provide?",
        "What does your company build?",
        "Tell me about your services."
      ]
    }
  ]
}
```

Keep answers **short and single-line**; provide **multiple question variants per fact**. Add facts for location, pricing, contact information, working hours, return policy, etc. Only include information you're allowed to use. `data/company.json` is ignored by Git, but the **repository is public**, so don't commit confidential company data or model checkpoints.

## 3. Train *our own neural model* from scratch

```bash
python train_company.py --company data/company.json --steps 2000
```

Training performs next-character prediction on `Q: ...\nA: ...\n` examples. The **answer** part contributes to the loss; the question prompt is masked. The final question phrasing for each fact is held out for rough validation. The initial checkpoint is saved to `checkpoints/company.pt` and excluded from Git.

To do a fast pipeline smoke test with fewer resources:

```bash
python train_company.py --company data/company.example.json --steps 2 --embed-dim 32 --layers 1 --context-size 192 --batch-size 2
```

A 2-step smoke test checks the pipeline, **not model quality**. If your longer questions/answers exceed the context length, increase `--context-size`. On a CPU, full training can be slow; a CUDA-capable PyTorch installation can use your NVIDIA GPU.

## 4. Chat with your company

**Grounded FAQ mode** (default; matches question variants and returns answers stored in JSON; does not need training):

```bash
python chat_company.py --company data/company.json
python chat_company.py --company data/company.json --question "What services do you provide?"
```

**Neural mode** (uses the model weights you trained from scratch; outputs may be wrong):

```bash
python chat_company.py --company data/company.json --mode neural
python chat_company.py --company data/company.json --mode neural --question "What services do you provide?" --max-new-tokens 120
```

- Grounded mode returns `I don't know from the supplied company details.` for unmatched or ambiguous questions. It is a simple lexical FAQ matcher, not semantic RAG or a learned model.
- Neural mode has a character-level vocabulary; unknown input characters and long questions are reported explicitly. It doesn't have an accuracy guarantee or memory of earlier chat turns.
- If you update the company data, **retrain** to update neural knowledge. Grounded mode reads the JSON every time you start it.

## 5. Run tests

```bash
python -m unittest discover -s tests -v
```

## Existing character-model experiment

You can still train on arbitrary text and generate samples:

```bash
python train.py --data data/sample.txt --steps 300
python generate.py --checkpoint checkpoints/tinyllm.pt --prompt "tiny"
```

## Code layout

- `tiny_llm/model.py` — scratch decoder-only Transformer and causal attention
- `tiny_llm/tokenizer.py` — character tokenizer
- `tiny_llm/company.py` — JSON facts, question examples, transparent FAQ matching
- `train_company.py` — answer-supervised scratch model training
- `chat_company.py` — CLI with explicit grounded and neural modes
- `data/company.example.json` — clearly fictional sample company
- `train.py`, `generate.py`, `data/sample.txt` — original plain-text experiment
- `tests/` — tokenizer, attention, company QA pipeline tests
