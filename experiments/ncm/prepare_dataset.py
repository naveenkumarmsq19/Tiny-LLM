"""Export Cake Galaxy company FAQ pairs for an experimental future NCM training run.

This prepares data only. It neither implements nor trains Next-Concept Modeling.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def prepare(company_path: Path, output_dir: Path) -> dict:
    obj = json.loads(company_path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or not isinstance(obj.get("facts"), list):
        raise ValueError("Expected company object with facts list")
    if not isinstance(obj.get("name"), str) or not obj["name"].strip():
        raise ValueError("Company needs a name")

    train, validation = [], []
    seen = set()
    for index, fact in enumerate(obj["facts"]):
        if not isinstance(fact, dict):
            raise ValueError(f"Fact {index} must be an object")
        key, answer, questions = fact.get("key"), fact.get("answer"), fact.get("questions")
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"Fact {index} has no key")
        if not isinstance(answer, str) or not answer.strip() or "\n" in answer or "\r" in answer:
            raise ValueError(f"Fact {key!r} requires a single-line answer")
        if not isinstance(questions, list) or not questions:
            raise ValueError(f"Fact {key!r} has no questions")
        for j, question in enumerate(questions):
            if (not isinstance(question, str) or not question.strip()
                    or "\n" in question or "\r" in question):
                raise ValueError(f"Invalid question in fact {key!r}")
            normalized = " ".join(question.casefold().split())
            if normalized in seen:
                raise ValueError(f"Duplicate question across splits: {question!r}")
            seen.add(normalized)
            record = {
                "id": f"fact-{index:03d}-q{j:02d}",
                "fact_key": key,
                "question": question.strip(),
                "prompt": f"Q: {question.strip()}\nA:",
                "answer": answer.strip(),
            }
            # Same hold-out policy as tiny_llm.company.split_examples.
            split = validation if len(questions) >= 2 and j == len(questions)-1 else train
            split.append(record)

    if not train:
        raise ValueError("No training questions")
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, records in (("train.jsonl", train), ("validation.jsonl", validation)):
        with (output_dir / filename).open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    stats = {
        "company": obj["name"],
        "facts": len(obj["facts"]),
        "train_examples": len(train),
        "validation_examples": len(validation),
        "validation_design": "Held-out phrasing; fact/answer leakage across splits",
        "model_training_performed": False,
        "ncm_native_training_format": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company", type=Path, default=Path("data/cake_galaxy.json"))
    parser.add_argument("--out", type=Path, default=Path("experiments/ncm/generated"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.company, args.out), indent=2))


if __name__ == "__main__":
    main()
