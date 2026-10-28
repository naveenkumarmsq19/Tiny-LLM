"""Interactive company chatbot with explicitly separated grounded/neural modes."""

import argparse
from pathlib import Path

import torch

from tiny_llm import CharTokenizer, TinyConfig, TinyLLM
from tiny_llm.company import grounded_answer, load_company


@torch.no_grad()
def neural_answer(
    model: TinyLLM,
    tokenizer: CharTokenizer,
    question: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    prompt = f"Q: {question.strip()}\nA:"
    if len(prompt) >= model.config.context_size:
        return "[Question too long for this model; shorten it or retrain with a larger context.]"
    try:
        tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    except ValueError:
        return "[Question contains characters absent from the training corpus.]"
    for _ in range(max_new_tokens):
        logits, _ = model(tokens[:, -model.config.context_size:])
        scores = logits[:, -1, :]
        if temperature == 0:
            token_id = torch.argmax(scores, dim=-1, keepdim=True)
        else:
            probabilities = torch.softmax(scores / temperature, dim=-1)
            token_id = torch.multinomial(probabilities, num_samples=1)
        next_char = tokenizer.decode([int(token_id.item())])
        if next_char == "\n":
            break
        tokens = torch.cat((tokens, token_id), dim=1)
    return tokenizer.decode(tokens[0].tolist())[len(prompt):].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about your company.")
    parser.add_argument("--company", type=Path, default=Path("data/company.example.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/company.pt"))
    parser.add_argument("--mode", choices=["grounded", "neural"], default="grounded")
    parser.add_argument("--question", type=str, default="")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.temperature < 0:
        parser.error("max-new-tokens must be positive and temperature non-negative.")
    knowledge = load_company(args.company)
    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    model = tokenizer = None
    if args.mode == "neural":
        if device == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA is not available. Use --device cpu.")
        try:
            checkpoint = torch.load(
                args.checkpoint, map_location=device, weights_only=True
            )
        except FileNotFoundError:
            parser.error("Model checkpoint not found. Run train_company.py first.")
        if checkpoint.get("company_name") != knowledge.name:
            parser.error("Checkpoint company differs from JSON company; retrain.")
        tokenizer = CharTokenizer(checkpoint["chars"])
        model = TinyLLM(TinyConfig(**checkpoint["config"])).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
    else:
        print("Mode: grounded FAQ lookup (uses JSON facts, NOT neural model generation).")

    def respond(question: str) -> str:
        if args.mode == "neural":
            assert model is not None and tokenizer is not None
            return neural_answer(
                model, tokenizer, question, device,
                args.max_new_tokens, args.temperature,
            )
        answer, _ = grounded_answer(knowledge, question)
        return answer or "I don't know from the supplied company details."

    if args.question.strip():
        print(f"AI: {respond(args.question)}")
        return
    print(f"Company: {knowledge.name}. Ask a question (type /quit to exit).")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if question.casefold() in {"/quit", "/exit", "exit"}:
            print("Bye!")
            break
        if question:
            print(f"AI: {respond(question)}")


if __name__ == "__main__":
    main()
