"""Cake Galaxy approval snapshot and pricing safety tests."""

import unittest
from decimal import Decimal
from pathlib import Path

from chat_cake_galaxy import (
    answer_cake_galaxy, calculate_price, get_weight,
    identify_product, load_catalogue,
)
from tiny_llm.company import load_company


ROOT = Path(__file__).resolve().parents[1]


class CakeGalaxyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.products = load_catalogue(ROOT / "data/cake_galaxy_catalog.json")
        cls.company = load_company(ROOT / "data/cake_galaxy.json")

    def test_all_32_reference_products(self):
        self.assertEqual(len(self.products), 32)
        names = {p["name"]: p["halfKg"] for p in self.products}
        self.assertEqual(names["Black Forest Cake"], 399)
        self.assertEqual(names["Almond Alphonso Cake"], 449)
        self.assertEqual(names["Red Velvet Cake"], 599)
        self.assertEqual(names["White Forest Cake"], 399)

    def test_weight_discount_boundary(self):
        self.assertEqual(calculate_price(449, Decimal("0.5")), Decimal("449.00"))
        self.assertEqual(calculate_price(449, Decimal("1")), Decimal("808.20"))
        self.assertEqual(calculate_price(449, Decimal("1.5")), Decimal("1212.30"))

    def test_eggless_surcharge_not_discounted(self):
        self.assertEqual(
            calculate_price(449, Decimal("1"), eggless=True), Decimal("908.20")
        )
        self.assertEqual(
            calculate_price(449, Decimal("0.5"), eggless=True), Decimal("499.00")
        )

    def test_unsupported_weight(self):
        with self.assertRaises(ValueError):
            calculate_price(449, Decimal("0.75"))
        with self.assertRaises(ValueError):
            calculate_price(449, Decimal("21"))

    def test_weight_extraction(self):
        self.assertEqual(get_weight("how much for half kg"), Decimal("0.5"))
        self.assertEqual(get_weight("1 kg red velvet"), Decimal("1"))
        self.assertEqual(get_weight("500 g cake"), Decimal("0.5"))

    def test_specific_name_not_generic_substring(self):
        p = identify_product(
            "price of German Black Forest Cake", self.products
        )
        self.assertEqual(p["name"], "German Black Forest Cake")
        p = identify_product("price of Black Forest Cake", self.products)
        self.assertEqual(p["name"], "Black Forest Cake")

    def test_approved_price_answer(self):
        a = answer_cake_galaxy(
            "How much is 1 kg eggless Almond Alphonso Cake?",
            self.company, self.products,
        )
        self.assertIn("₹908.20", a)
        self.assertIn("verify the live checkout", a)

    def test_custom_cake_no_invented_quote(self):
        a = answer_cake_galaxy(
            "What is the custom design cake price?",
            self.company, self.products,
        )
        self.assertIn("admin", a.casefold())
        self.assertNotIn("₹800", a)

    def test_unknown_question_refusal(self):
        a = answer_cake_galaxy(
            "What is the celebrity endorsement agreement?",
            self.company, self.products,
        )
        self.assertIn("don't have a verified answer", a)

    def test_approved_not_live_stock(self):
        a = answer_cake_galaxy(
            "Do you have Choco Oreo Cake?",
            self.company, self.products,
        )
        self.assertIn("Live stock", a)


if __name__ == "__main__":
    unittest.main()
