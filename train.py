"""Train Tiny-LLM on a plain text file."""

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from tiny_llm import CharTokenizer, TinyConfig, TinyLLM


def batch_from_text(
    tokens: torch.Tensor, batch_size: int, context_size: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(tokens) <= context_size:
        raise ValueError("Data split needs at least context_size + 1 tokens.")
    starts = torch.randint(len(tokens) - context_size, (batch_size,))
    x = torch.stack([tokens[i : i + context_size] for i in starts])
    y = torch.stack([tokens[i + 1 : i + context_size + 1] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(
    model: TinyLLM,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    batch_size: int,
    context_size: int,
    device: str,
    batches: int = 5,
) -> dict[str, float]:
    model.eval()
    losses = {}
    for name, tokens in (("train", train_tokens), ("val", val_tokens)):
        values = []
        for _ in range(batches):
            x, y = batch_from_text(tokens, batch_size, context_size, device)
            _, loss = model(x, y)
            values.append(loss.item())
        losses[name] = sum(values) / len(values)
    model.train()
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny character Transformer.")
    parser.add_argument("--data", type=Path, default=Path("data/sample.txt"))
    parser.add_argument("--save", type=Path, default=Path("checkpoints/tinyllm.pt"))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--context-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.steps < 1 or args.batch_size < 1 or args.eval_interval < 1 or args.lr <= 0:
        parser.error("steps, batch-size, eval-interval and lr must be positive.")

    torch.manual_seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available; use --device cpu.")

    text = args.data.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    all_tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split = int(len(all_tokens) * 0.9)
    train_tokens, val_tokens = all_tokens[:split], all_tokens[split:]
    if min(len(train_tokens), len(val_tokens)) <= args.context_size:
        parser.error("Text corpus is too short for a 90/10 split at this context size.")

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
    print(
        f"device={device} | chars={len(text)} | vocab={tokenizer.vocab_size} "
        f"| parameters={sum(p.numel() for p in model.parameters()):,}",
        flush=True,
    )

    for step in range(1, args.steps + 1):
        x, y = batch_from_text(train_tokens, args.batch_size, args.context_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            metrics = evaluate(
                model, train_tokens, val_tokens, args.batch_size,
                args.context_size, device,
            )
            print(
                f"step {step:>5} | train {metrics['train']:.4f} | "
                f"val {metrics['val']:.4f}",
                flush=True,
            )

    args.save.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(config),
            "chars": tokenizer.chars,
            "model_state": model.state_dict(),
            "step": args.steps,
        },
        args.save,
    )
    print(f"Checkpoint saved to {args.save}")


if __name__ == "__main__":
    main()
