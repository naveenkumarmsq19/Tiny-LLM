"""Existing lexical FAQ baseline only; NOT an NCM or neural model benchmark."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tiny_llm.company import grounded_answer, load_company  # noqa: E402


def evaluate(company: Path, validation: Path) -> dict:
    knowledge = load_company(company)
    total = matches = abstentions = 0
    for line in validation.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        answer, _ = grounded_answer(knowledge, record["question"])
        total += 1
        if answer is None:
            abstentions += 1
        elif answer.strip() == record["answer"].strip():
            matches += 1
    return {
        "method": "lexical_grounded_faq_not_ncm",
        "samples": total,
        "exact_matches": matches,
        "abstentions": abstentions,
        "exact_match_rate": matches / total if total else None,
        "limitation": "Matcher sees all source variants, including validation questions; not an independent generalization benchmark.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company", type=Path, default=Path("data/cake_galaxy.json"))
    parser.add_argument("--validation", type=Path, default=Path("experiments/ncm/generated/validation.jsonl"))
    args = parser.parse_args()
    print(json.dumps(evaluate(args.company, args.validation), indent=2))


if __name__ == "__main__":
    main()
