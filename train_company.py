"""Train the project's own character-level Transformer on company FAQ examples.

The weights start randomly initialized. No pretrained model is downloaded.
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from tiny_llm import CharTokenizer, TinyConfig, TinyLLM
from tiny_llm.company import TrainingExample, load_company, split_examples


def encode_examples(
    examples: list[TrainingExample], tokenizer: CharTokenizer, context_size: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Create next-character labels; ignore prompt and padding tokens."""
    encoded = []
    for example in examples:
        ids = tokenizer.encode(example.text)
        prompt_size = len(tokenizer.encode(example.prompt))
        if len(ids) - 1 > context_size:
            raise ValueError(
                f"Example too long ({len(ids)-1} > {context_size}). "
                f"Increase --context-size. Question: {example.question!r}"
            )
        inputs = torch.tensor(ids[:-1], dtype=torch.long)
        targets = torch.tensor(ids[1:], dtype=torch.long)
        targets[:prompt_size - 1] = -100  # Train only on ' ' + answer + '\n'.
        encoded.append((inputs, targets))
    return encoded


def sample_batch(
    encoded: list[tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    pad_id: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.randint(len(encoded), (batch_size,)).tolist()
    selected = [encoded[i] for i in indices]
    width = max(len(inputs) for inputs, _ in selected)
    x = torch.full((batch_size, width), pad_id, dtype=torch.long)
    y = torch.full((batch_size, width), -100, dtype=torch.long)
    for index, (inputs, targets) in enumerate(selected):
        x[index, : len(inputs)] = inputs
        y[index, : len(targets)] = targets
    return x.to(device), y.to(device)


@torch.no_grad()
def average_loss(
    model: TinyLLM,
    examples: list[tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    pad_id: int,
    device: str,
    repeats: int = 3,
) -> float:
    model.eval()
    losses = []
    for _ in range(repeats):
        x, y = sample_batch(examples, batch_size, pad_id, device)
        _, loss = model(x, y)
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a company Q&A model from scratch.")
    parser.add_argument("--company", type=Path, default=Path("data/company.example.json"))
    parser.add_argument("--save", type=Path, default=Path("checkpoints/company.pt"))
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-size", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.eval_interval) < 1 or args.lr <= 0:
        parser.error("steps, batch-size, eval-interval and lr must be positive.")
    if args.context_size < 1:
        parser.error("context-size must be positive.")

    torch.manual_seed(args.seed)
    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available. Try --device cpu.")
    knowledge = load_company(args.company)
    train, validation = split_examples(knowledge)
    if not train:
        parser.error("Company JSON needs at least one training question.")
    tokenizer = CharTokenizer.from_text("".join(ex.text for ex in train + validation))
    train_samples = encode_examples(train, tokenizer, args.context_size)
    val_samples = encode_examples(validation, tokenizer, args.context_size)
    config = TinyConfig(
        vocab_size=tokenizer.vocab_size,
        context_size=args.context_size,
        embed_dim=args.embed_dim,
        heads=args.heads,
        layers=args.layers,
        dropout=args.dropout,
    )
    model = TinyLLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    pad_id = tokenizer.stoi[" "]
    print(
        f"Training scratch model for {knowledge.name} | device={device} | "
        f"train_examples={len(train)} | held_out_phrasings={len(validation)} | "
        f"parameters={sum(p.numel() for p in model.parameters()):,}",
        flush=True,
    )
    for step in range(1, args.steps + 1):
        model.train()
        x, y = sample_batch(train_samples, args.batch_size, pad_id, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            train_loss = average_loss(model, train_samples, args.batch_size, pad_id, device)
            val_loss = (
                average_loss(model, val_samples, args.batch_size, pad_id, device)
                if val_samples else None
            )
            print(
                f"step={step} train_loss={train_loss:.4f} "
                + (f"held_out_loss={val_loss:.4f}" if val_loss is not None else ""),
                flush=True,
            )
    args.save.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(config),
            "chars": tokenizer.chars,
            "model_state": model.state_dict(),
            "company_name": knowledge.name,
            "steps": args.steps,
        },
        args.save,
    )
    print(f"Saved scratch-trained checkpoint: {args.save}")


if __name__ == "__main__":
    main()
