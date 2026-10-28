"""Quick model correctness and tokenizer tests."""

import unittest

import torch

from tiny_llm import CharTokenizer, TinyConfig, TinyLLM


class TestTinyLLM(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(123)
        self.config = TinyConfig(
            vocab_size=8, context_size=8, embed_dim=16,
            heads=4, layers=1, dropout=0.0,
        )
        self.model = TinyLLM(self.config).eval()

    def test_forward_shape_and_loss(self) -> None:
        tokens = torch.randint(0, 8, (2, 5))
        logits, loss = self.model(tokens, tokens)
        self.assertEqual(logits.shape, (2, 5, 8))
        self.assertTrue(torch.isfinite(loss).item())

    def test_causal_attention_blocks_future_information(self) -> None:
        first = torch.tensor([[1, 2, 3, 4]])
        second = torch.tensor([[1, 2, 7, 6]])
        logits_a, _ = self.model(first)
        logits_b, _ = self.model(second)
        torch.testing.assert_close(logits_a[:, :2], logits_b[:, :2])

    def test_generation_length(self) -> None:
        result = self.model.generate(
            torch.tensor([[1, 2]]), max_new_tokens=3, top_k=2
        )
        self.assertEqual(result.shape, (1, 5))

    def test_tokenizer_round_trip(self) -> None:
        tokenizer = CharTokenizer.from_text("hello tiny")
        self.assertEqual(tokenizer.decode(tokenizer.encode("hello")), "hello")
        with self.assertRaises(ValueError):
            tokenizer.encode("?")


if __name__ == "__main__":
    unittest.main()
