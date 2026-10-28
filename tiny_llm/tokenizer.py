"""Minimal character-level tokenizer (one token per Unicode character)."""


class CharTokenizer:
    def __init__(self, chars: list[str]):
        if not chars:
            raise ValueError("Vocabulary must not be empty.")
        self.chars = list(chars)
        self.stoi = {char: index for index, char in enumerate(self.chars)}
        self.itos = {index: char for index, char in enumerate(self.chars)}
        if len(self.stoi) != len(self.chars):
            raise ValueError("Vocabulary contains duplicate characters.")

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        try:
            return [self.stoi[char] for char in text]
        except KeyError as exc:
            raise ValueError(
                f"Character {exc.args[0]!r} is missing from the training vocabulary."
            ) from exc

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.itos[index] for index in token_ids)
