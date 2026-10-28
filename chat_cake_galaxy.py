"""Cake Galaxy customer-facts assistant: exact catalogue maths + conservative FAQ.

Grounded mode is deterministic and *not* the trained neural network.
Neural mode is available separately via chat_company.py --mode neural.
"""
from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from tiny_llm.company import grounded_answer, load_company


_PRICE_WORDS = {"price", "cost", "rate", "much", "rs", "rupees", "amount", "total", "₹", "charge", "charges"}
_EGGLESS = re.compile(r"\beggless\b", re.I)


def load_catalogue(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    products = data.get("products")
    if not isinstance(products, list) or not products:
        raise ValueError("Catalogue needs a products list.")
    for item in products:
        if (not isinstance(item, dict)
                or not isinstance(item.get("name"), str)
                or not isinstance(item.get("halfKg"), int)
                or item["halfKg"] <= 0):
            raise ValueError(f"Invalid product entry: {item!r}")
    return products


def get_weight(question: str) -> Decimal | None:
    q = question.casefold()
    if re.search(r"\b(half\s*(a\s*)?(kg|kilo|kilogram)|500\s*(g|gm|gram|grams))\b", q):
        return Decimal("0.5")
    if re.search(r"\b(one\s*(kg|kilo|kilogram))\b", q):
        return Decimal("1")
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*(kg|kgs|kilo|kilograms?|g|gm|grams?)\b", q)
    if not m:
        return None
    value = Decimal(m.group(1))
    return value / 1000 if m.group(2).startswith("g") else value


def calculate_price(half_kg: int, weight: Decimal, eggless: bool = False) -> Decimal:
    """Reference cake-base calculation only; no delivery, toppings, tax or checkout quote."""
    if not (Decimal("0.5") <= weight <= 20 and weight * 2 == int(weight * 2)):
        raise ValueError("Supported weights are 0.5 to 20 kg in 0.5 kg increments.")
    base = Decimal(half_kg) * weight * 2
    if weight >= 1:
        base *= Decimal("0.90")
    if eggless:
        base += weight * 2 * Decimal(50)
    return base.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def identify_product(question: str, products: list[dict]) -> dict | None:
    """Only exact name/unique name-without-'Cake' matches, never keyword guessing."""
    q = " ".join(re.findall(r"[a-z0-9]+", question.casefold()))
    found = []
    for item in products:
        name = " ".join(re.findall(r"[a-z0-9]+", item["name"].casefold()))
        variants = {name}
        if name.endswith(" cake"):
            variants.add(name[:-5])
        for alias in variants:
            if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", q):
                found.append((len(alias), item))
    if not found:
        return None
    found.sort(key=lambda x: x[0], reverse=True)
    if len(found) > 1 and found[0][0] == found[1][0] and found[0][1]["name"] != found[1][1]["name"]:
        return None
    return found[0][1]


def answer_cake_galaxy(question: str, company, products: list[dict]) -> str:
    if not question.strip():
        return "Please ask a question."
    q = question.casefold()
    if re.search(r"\b(custom|design|customized|personalized)\b", q):
        response, score = grounded_answer(company, question, threshold=0.62)
        if response is not None and "design" in response.casefold():
            return response
        return ("Design cake prices require a separate admin quotation after the "
                "reference image and requirements are reviewed. No automatic custom cake price is confirmed.")
    item = identify_product(question, products)
    if item:
        name, half = item["name"], item["halfKg"]
        wants_price = bool(_PRICE_WORDS & set(re.findall(r"\w+|₹", q))) or get_weight(question) is not None
        if not wants_price:
            return (f"{name} is on the approved reference catalogue. "
                    "Live stock and ordering availability must be checked on the website.")
        weight = get_weight(question)
        if weight is None:
            price1 = calculate_price(half, Decimal("1"))
            return (f"{name} approved reference price: 0.5 kg ₹{half:.2f}; "
                    f"1 kg cake base after 10% discount ₹{price1:.2f}. "
                    "Eggless, add-ons and delivery are extra. Live availability is unverified.")
        try:
            price = calculate_price(half, weight, eggless=bool(_EGGLESS.search(q)))
        except ValueError as exc:
            return str(exc)
        note = " including eggless surcharge" if _EGGLESS.search(q) else ""
        return (f"{name} {weight:g} kg reference cake price{note}: ₹{price:.2f}. "
                "Add-ons and delivery charges are excluded; verify the live checkout price.")
    response, _ = grounded_answer(company, question, threshold=0.66)
    return response or ("I don't have a verified answer in the Cake Galaxy company facts. "
                        "Please check the live website or contact the store.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cake Galaxy reference-fact chat (not neural AI).")
    parser.add_argument("--company", type=Path, default=Path("data/cake_galaxy.json"))
    parser.add_argument("--catalog", type=Path, default=Path("data/cake_galaxy_catalog.json"))
    parser.add_argument("--question", default="")
    args = parser.parse_args()
    company = load_company(args.company)
    products = load_catalogue(args.catalog)
    if args.question:
        print(answer_cake_galaxy(args.question, company, products))
        return
    print(f"Cake Galaxy FAQ assistant ready: {len(company.facts)} facts / {len(products)} prices. /quit to exit.")
    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            return
        if question.casefold() in {"/quit", "/exit", "exit"}:
            print("Bye!")
            return
        if question:
            print("AI:", answer_cake_galaxy(question, company, products))


if __name__ == "__main__":
    main()
