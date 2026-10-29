"""NCM research bridge data-export tests; no GPU required."""
import json
import tempfile
import unittest
from pathlib import Path

from experiments.ncm.prepare_dataset import prepare


class NCMPrepTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "company.json"
        self.output = self.root / "result"

    def write(self, facts):
        self.source.write_text(
            json.dumps({"name": "Example", "facts": facts}), encoding="utf-8"
        )

    def test_splits_and_manifest(self):
        self.write([
            {"key": "hours", "answer": "10 to 6", "questions": ["When open?", "Opening hours?", "Working times?"]},
            {"key": "location", "answer": "City", "questions": ["Where?", "Which city?"]},
            {"key": "delivery", "answer": "Ask store", "questions": ["Do you deliver?"]},
        ])
        result = prepare(self.source, self.output)
        self.assertEqual((result["train_examples"], result["validation_examples"]), (4, 2))
        train = [json.loads(row) for row in (self.output / "train.jsonl").read_text().splitlines()]
        val = [json.loads(row) for row in (self.output / "validation.jsonl").read_text().splitlines()]
        self.assertEqual(val[0]["question"], "Working times?")
        self.assertEqual(train[0]["prompt"], "Q: When open?\nA:")
        self.assertTrue(set(x["id"] for x in train).isdisjoint(x["id"] for x in val))
        self.assertFalse(result["model_training_performed"])

    def test_duplicate_questions_rejected(self):
        self.write([
            {"key": "a", "answer": "A", "questions": ["Hello there"]},
            {"key": "b", "answer": "B", "questions": ["  hello   THERE "]},
        ])
        with self.assertRaisesRegex(ValueError, "Duplicate question"):
            prepare(self.source, self.output)

    def test_invalid_multiline_answer_rejected(self):
        self.write([{"key": "a", "answer": "bad\nanswer", "questions": ["Ask"]}])
        with self.assertRaisesRegex(ValueError, "single-line"):
            prepare(self.source, self.output)


if __name__ == "__main__":
    unittest.main()
