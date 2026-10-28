"""Validate company knowledge, build answer-supervised examples, and match known FAQs.

No external LLM or embeddings are involved in this module. The matching mode is
an explicitly separate, lexical FAQ lookup; it is not learned text generation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CompanyFact:
    key: str
    answer: str
    questions: tuple[str, ...]


@dataclass(frozen=True)
class CompanyKnowledge:
    name: str
    facts: tuple[CompanyFact, ...]


@dataclass(frozen=True)
class TrainingExample:
    question: str
    answer: str

    @property
    def prompt(self) -> str:
        return f"Q: {self.question}\nA:"

    @property
    def text(self) -> str:
        return f"{self.prompt} {self.answer}\n"


def load_company(path: str | Path) -> CompanyKnowledge:
    """Read the documented JSON schema and fail early on ambiguous data."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Company JSON must contain an object.")
    name = data.get("name")
    facts = data.get("facts")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Company JSON needs a non-empty 'name'.")
    if not isinstance(facts, list) or not facts:
        raise ValueError("Company JSON needs a non-empty 'facts' array.")

    parsed = []
    seen_keys = set()
    seen_questions = set()
    for index, fact in enumerate(facts):
        if not isinstance(fact, dict):
            raise ValueError(f"Fact {index} must be an object.")
        key, answer, questions = (
            fact.get("key"), fact.get("answer"), fact.get("questions")
        )
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"Fact {index} must have a non-empty 'key'.")
        if key in seen_keys:
            raise ValueError(f"Duplicate fact key: {key!r}.")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"Fact {key!r} must have a non-empty 'answer'.")
        if "\n" in answer or "\r" in answer:
            raise ValueError(f"Fact {key!r} must have a single-line answer.")
        if not isinstance(questions, list) or not questions or any(
            not isinstance(q, str) or not q.strip() or "\n" in q or "\r" in q
            for q in questions
        ):
            raise ValueError(f"Fact {key!r} needs non-empty, single-line questions.")
        for question in questions:
            normalized = " ".join(question.casefold().split())
            if normalized in seen_questions:
                raise ValueError(f"Duplicate question across facts: {question!r}.")
            seen_questions.add(normalized)
        parsed.append(
            CompanyFact(key=key.strip(), answer=answer.strip(),
                        questions=tuple(q.strip() for q in questions))
        )
        seen_keys.add(key)
    return CompanyKnowledge(name=name.strip(), facts=tuple(parsed))


def split_examples(
    knowledge: CompanyKnowledge,
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """Hold out the final question variant for each fact with >=2 variants.

    Answers overlap by design; validation checks unseen phrasings, not unseen
    company facts. This does not establish real-world accuracy.
    """
    train, validation = [], []
    for fact in knowledge.facts:
        for index, question in enumerate(fact.questions):
            example = TrainingExample(question=question, answer=fact.answer)
            if len(fact.questions) >= 2 and index == len(fact.questions) - 1:
                validation.append(example)
            else:
                train.append(example)
    return train, validation


# Drop common question fillers before lexical matching.
_STOPWORDS = {
    "a", "about", "an", "and", "are", "can", "company", "could", "do",
    "does", "for", "from", "give", "how", "i", "in", "is", "me", "of",
    "on", "please", "tell", "the", "their", "they", "to", "us", "we",
    "what", "when", "where", "which", "who", "with", "you", "your",
}


def keywords(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold())) - _STOPWORDS


def grounded_answer(
    knowledge: CompanyKnowledge,
    question: str,
    threshold: float = 0.62,
) -> tuple[str | None, float]:
    """Return an answer only for a strong lexical match with a supplied question.

    This deliberately declines unsupported questions instead of asking a tiny
    generative model to fabricate an answer.
    """
    query = keywords(question)
    if not query:
        return None, 0.0
    by_fact = []
    for fact in knowledge.facts:
        scores = []
        for candidate in fact.questions:
            words = keywords(candidate)
            overlap = len(query & words)
            if not words or not overlap:
                continue
            scores.append(0.75 * overlap / len(query) + 0.25 * overlap / len(words))
        if scores:
            by_fact.append((max(scores), fact.answer))
    if not by_fact:
        return None, 0.0
    by_fact.sort(key=lambda item: item[0], reverse=True)
    score, answer = by_fact[0]
    # If two different facts are equally plausible, ask the user to rephrase.
    if score < threshold or (
        len(by_fact) > 1 and abs(score - by_fact[1][0]) < 0.08
    ):
        return None, score
    return answer, score
