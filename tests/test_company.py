"""Company-knowledge, masked-label, and inference smoke tests."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from chat_company import neural_answer
from tiny_llm import CharTokenizer, TinyConfig, TinyLLM
from tiny_llm.company import (
    CompanyFact, CompanyKnowledge, TrainingExample,
    grounded_answer, load_company, split_examples,
)
from train_company import encode_examples, sample_batch


class CompanyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = Path(__file__).resolve().parents[1] / "data/company.example.json"
        cls.knowledge = load_company(cls.path)

    def test_company_sample_valid_and_split(self) -> None:
        train, val = split_examples(self.knowledge)
        self.assertEqual(len(train), 10)
        self.assertEqual(len(val), 5)
        self.assertEqual({x.answer for x in train}, {x.answer for x in val})

    def test_grounded_known_question_and_unknown_refusal(self) -> None:
        answer, confidence = grounded_answer(self.knowledge, "What services do you provide?")
        self.assertIn("websites", answer)
        self.assertGreaterEqual(confidence, 0.62)
        answer, _ = grounded_answer(self.knowledge, "What is your refund policy?")
        self.assertIsNone(answer)

    def test_reject_duplicate_question(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            company = {
                "name": "Test",
                "facts": [
                    {"key": "one", "answer": "a", "questions": ["where are you?"]},
                    {"key": "two", "answer": "b", "questions": ["WHERE ARE YOU?"]},
                ],
            }
            path = Path(directory) / "company.json"
            path.write_text(json.dumps(company), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate question"):
                load_company(path)

    def test_masked_training_labels_and_padding(self) -> None:
        ex = TrainingExample("Where?", "Here.")
        tokenizer = CharTokenizer.from_text(ex.text + "Q: Hi?\nA: Yes.\n")
        short = TrainingExample("Hi?", "Yes.")
        encoded = encode_examples([ex, short], tokenizer, 64)
        inputs, targets = encoded[0]
        self.assertEqual(inputs.shape, targets.shape)
        self.assertTrue(torch.all(targets[:len(ex.prompt) - 1] == -100).item())
        self.assertEqual(int(targets[len(ex.prompt) - 1]), tokenizer.stoi[" "])
        x, y = sample_batch(encoded, 5, tokenizer.stoi[" "], "cpu")
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(torch.any(y == -100).item())

    def test_neural_inference_runs_with_random_weights(self) -> None:
        # Tests the inference path; NOT an answer-quality assertion.
        ex = TrainingExample("Where?", "Here.")
        tok = CharTokenizer.from_text(ex.text)
        model = TinyLLM(TinyConfig(tok.vocab_size, context_size=32,
                                    embed_dim=16, heads=4, layers=1, dropout=0))
        model.eval()
        reply = neural_answer(model, tok, "Where?", "cpu", 4, 0.0)
        self.assertIsInstance(reply, str)
        self.assertLessEqual(len(reply), 4)


if __name__ == "__main__":
    unittest.main()
