# NCM research plan for Tiny-LLM (October 2026)

## What this branch does

Adds an **isolated research data exporter and non-neural comparison baseline**. The existing Tiny-LLM decoder-only character Transformer, Cake Galaxy pricing, chat scripts, and production behavior are unchanged. **No NCM architecture is implemented, trained, benchmarked, or deployed by this PR.**

The last documented Cake Galaxy Tiny-LLM run had ~78,596 parameters and **0/60 exact held-out answers**. The verified-price/grounded FAQ chatbot should remain the source for customer-facing answers.

## What real Next-Concept Modeling needs

Next-Concept Modeling (NCM) from Stanford Hazy Research uses:
1. A trained multiscale residual VQ tokenizer to encode text in hierarchies of discrete codes.
2. A separate coarse-to-fine Transformer generator trained with the tokenizer frozen.
3. Level-by-level parallel prediction, followed by text decoding.

Official sources:
- [Paper](https://arxiv.org/abs/2610.14162)
- [Explanation](https://hazyresearch.stanford.edu/blog/2026-10-12-ncm)
- [Official code](https://github.com/HazyResearch/ncm)

At time of review, the official repo has **released evaluation code/checkpoints but not complete public training scripts**. Its TinyStories and LM1B checkpoints are not Cake Galaxy Q&A assistants. Do **not** claim NCM can be plugged into our existing checkpoint or reused as a production chatbot without training.

## Prepare an experimental dataset

From the Tiny-LLM repo root:

```bash
python experiments/ncm/prepare_dataset.py --company data/cake_galaxy.json --out experiments/ncm/generated
python experiments/ncm/eval_grounded.py --company data/cake_galaxy.json --validation experiments/ncm/generated/validation.jsonl
python -m unittest discover -s tests -v
```

Current Cake Galaxy data should yield **60 facts, 180 training questions, 60 validation phrasings**. Output: `train.jsonl`, `validation.jsonl`, `manifest.json`. Generated files are ignored by Git.

**Important limitations:** the split holds out only the last phrasing of each already-known fact, so it does not test unseen facts. The baseline matcher has access to all variants, including held-out variants, so the lexical baseline is a sanity test only. Exported JSONL is **not** native NCM tokens or a trainable official NCM dataset. The next engineering step must implement/port a proper conditional training pipeline and an independent unseen-question evaluation set.

## Reproduce official NCM separately

```bash
git clone https://github.com/HazyResearch/ncm.git
cd ncm
python -m pip install -r requirements.txt
# Install Hugging Face CLI per its documentation.
hf download hazyresearch/ncm-checkpoints --local-dir artifacts
hf download hazyresearch/ncm-datasets --repo-type dataset --local-dir data
python evaluate.py --config config/eval/tinystories.yaml --checkpoint-dir artifacts/tinystories
```

This evaluates an official pretrained **unconditional TinyStories model**, **not** a company-trained or instruction-following NCM chatbot. Use a separate CUDA environment, check storage, checkpoint sizes, package compatibility and any usage restrictions before downloading.

## GPU, data and costs (planning only)

The paper studies **97M / 317M / 725M**-parameter models. Parameter count alone is not enough to forecast training memory or wall time: VQ tokenizer training, transformer gradients, optimizer, activations, context length, batch size and precision all matter.

| Stage | Planning assumption |
| --- | --- |
| Dataset preparation and experiment scripts | CPU; no GPU needed |
| Official smaller checkpoint evaluation | Start with a CUDA GPU (24GB–80GB range depends on the checkpoint); profile before booking |
| From-scratch 100M+ NCM research training | A100/H100 80GB class GPU(s) is a conservative starting point; determine actual VRAM from a working training implementation |
| Cake Galaxy-quality chatbot | 240 question phrasings are **not sufficient** evidence of generalization; needs much more task-appropriate, licensed data and verified evaluation |

**Cloud compute hourly examples (Runpod Secure Cloud, October 2026, check before renting):**

| GPU | Published pod price | 10 GPU-hours | 100 GPU-hours |
| --- | --- | --- | --- |
| RTX 4090 (24GB) | $0.89/hour | $8.90 | $89 |
| A100 (80GB) | $1.79/hour | $17.90 | $179 |
| H100 SXM (80GB) | $3.99/hour | $39.90 | $399 |

Prices are USD before tax, storage, data transfer or downtime. **10/100 hours are billing scenarios, not claims that an NCM model can be trained in that time.** Source: [Runpod pricing](https://www.runpod.io/pricing).

## Practical milestones / acceptance gates

1. Reproduce official NCM evaluation and log throughput, latency and peak VRAM.
2. Prepare a genuinely unseen Cake Galaxy QA test set, without copying its questions into FAQ reference text or training.
3. Research/implement a **real multiscale VQ tokenizer + conditional NCM training loop**, with versioned data, reproducible checkpoints and GPU monitoring.
4. Compare NCM with the current autoregressive Tiny-LLM, existing grounded FAQ and a small pretrained instruction model on **real answer correctness**, abstention, speed and operating cost.
5. Never use raw generated prices, orders, refunds or delivery promises as authoritative; preserve deterministic backend checks.

Do not merge experimental code into production automatically. Maintain a separate research branch until the model has passed these gates.
