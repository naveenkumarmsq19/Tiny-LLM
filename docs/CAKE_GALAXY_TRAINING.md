# Cake Galaxy — from-scratch Tiny-LLM training

## Source and privacy

- Cake Galaxy app source: `naveenkumarmsq19/Cake-galaxy`, docs and owner-approved catalogue snapshot dated **23 October 2026**.
- `data/cake_galaxy.json`: **60** public/reference facts, **240** question variants, 28 business/operational facts and 32 product price entries.
- `data/cake_galaxy_catalog.json`: 32 approved half-kilogram reference prices extracted from `packages/catalog/approvedCakes.js`. Does not verify product stock or live website.
- **Do not store:** OTP codes, credentials, payment keys, customer data, undisclosed company records or unverified tax/license/support details. This repository is **public**.
- The live website could not be independently inspected during preparation. Payment and service statuses are described conservatively.

## Recommended customer FAQ mode

```bash
python chat_cake_galaxy.py --question "What is the price of Red Velvet Cake?"
python chat_cake_galaxy.py --question "How much is 1 kg eggless Almond Alphonso Cake?"
python chat_cake_galaxy.py
```

This mode returns exact reference prices using the **approved code formula** and cautious answers from verified project facts; it does **not** use a trained neural model.

Calculation for regular cakes:

- Base = 0.5 kg approved price × (weight in kg / 0.5)
- At weights of 1 kg or more, apply 10% off **cake base only**
- If eggless is selected, add ₹50 for every 0.5 kg after the cake discount
- Add-ons, delivery and taxes may change the final checkout total
- Custom design cakes are excluded; the final custom quote is **admin-only**

Example: 1 kg Almond Alphonso base ₹449 × 2 × 0.9 = ₹808.20; eggless would add ₹100 = ₹908.20, before extras. Check website configuration and active products before making any live promise.

## Training the *actual scratch Transformer* locally

```bash
python train_company.py \
  --company data/cake_galaxy.json \
  --save checkpoints/cake_galaxy.pt \
  --steps 1200 --context-size 320 --embed-dim 48 --layers 2 \
  --heads 4 --batch-size 6 --dropout 0 --lr 0.0008 \
  --eval-interval 150
python chat_company.py --company data/cake_galaxy.json \
  --checkpoint checkpoints/cake_galaxy.pt --mode neural
```

For Windows Command Prompt, run the `python train_company.py ...` command on a single line.

The script trains the project's own Transformer from *random weights*; no pretrained model is loaded.

## Actual run completed on 29 October 2026

A locally curated **Cake Galaxy training snapshot** with the same 60 business/product fact topics and 240 question variations was used for the following offline run. It was **not byte-identical** to `data/cake_galaxy.json`; therefore, to reproduce the supplied checkpoint exactly, use the bundled training snapshot shipped alongside that checkpoint, or retrain on the repository's public dataset.

| Metric | Result |
| --- | --- |
| Parameters | 78,596 |
| Architecture | character-level decoder Transformer, 2 blocks, 4 attention heads, embedding size 48 |
| Steps | 1200 |
| Context | 320 characters |
| Batch size | 6 |
| Train / held-out paraphrases | 180 / 60 |
| First train and held-out loss | 4.2874 / 4.2872 |
| Last train and held-out loss | 0.5117 / 1.7125 |
| Held-out exact-answer matches, greedy decoding | **0 / 60** |

**Quality conclusion:** training executed and loss decreased, but free-form model generations were frequently incoherent or gave **incorrect cake prices**. This is **not** a deployable customer-service LLM. Tiny character-level training on 240 examples teaches patterns and partial memorization, not deep language comprehension. Held-out samples test new phrasing of already-known facts, not new facts. Do **not** use neural mode for payments, stock, discounts, delivery commitments or refunds.

Checkpoints are excluded from Git (`.gitignore`). A standalone trained checkpoint and exact matching training snapshot were packaged separately for the user to download and copy into the cloned project. No binary checkpoint has been uploaded to GitHub.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover the exact discount threshold, surcharge order, weight limits, item matching, custom quote refusal, and unknown-question refusal.

## Next research steps

To make the **learned** mode useful, increase high-quality human-written question/answer examples substantially, use a stronger tokenizer and model, evaluate on genuinely unseen held-out questions, and consider retrieval from verified company facts. Keep deterministic pricing/operational policy logic independent of generated language even if a future model is more fluent.
