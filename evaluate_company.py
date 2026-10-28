"""Evaluate exact-answer accuracy of the scratch-trained company Transformer."""

import argparse
from pathlib import Path

import torch

from chat_company import neural_answer
from tiny_llm import CharTokenizer, TinyConfig, TinyLLM
from tiny_llm.company import load_company, split_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Test held-out company Q&A accuracy.")
    parser.add_argument("--company", type=Path, default=Path("data/cake_galaxy.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/cake_galaxy.pt"))
    parser.add_argument("--examples", type=int, default=5, help="Number of sample answers to show.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.examples < 0:
        parser.error("--examples cannot be negative.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available; use --device cpu.")

    company = load_company(args.company)
    _, validation = split_examples(company)
    if not validation:
        parser.error("At least one fact must have two question variants for validation.")
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    if state.get("company_name") != company.name:
        parser.error("Checkpoint company name does not match dataset.")
    model = TinyLLM(TinyConfig(**state["config"])).to(args.device)
    model.load_state_dict(state["model_state"])
    model.eval()
    tokenizer = CharTokenizer(state["chars"])

    correct = 0
    for i, example in enumerate(validation):
        answer = neural_answer(
            model, tokenizer, example.question, args.device,
            max_new_tokens=max(32, min(400, len(example.answer) + 25)),
            temperature=0.0,
        )
        match = answer.strip().casefold() == example.answer.strip().casefold()
        correct += int(match)
        if i < args.examples:
            print(f"\nQ: {example.question}\nExpected: {example.answer}\nNeural: {answer}\nMatch: {match}")
    print(f"\nExact-answer accuracy on held-out phrasings: {correct}/{len(validation)} "
          f"({100 * correct / len(validation):.2f}%)")
    print("Held-out phrasings may still share factual answers with train examples.")


if __name__ == "__main__":
    main()
