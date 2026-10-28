"""Generate text from a trained Tiny-LLM checkpoint."""

import argparse
from pathlib import Path

import torch

from tiny_llm import CharTokenizer, TinyConfig, TinyLLM


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate character-level text.")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/tinyllm.pt")
    )
    parser.add_argument("--prompt", type=str, default="tiny")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available; use --device cpu.")

    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=True
    )
    tokenizer = CharTokenizer(checkpoint["chars"])
    model = TinyLLM(TinyConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    start = torch.tensor(
        [tokenizer.encode(args.prompt)], dtype=torch.long, device=device
    )
    output = model.generate(
        start,
        args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k or None,
    )
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
