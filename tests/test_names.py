"""Base names and the one right/wrong rule for name-labelled regression rows (pipeline/names.py)."""
import importlib.util
import os
import unittest

import _paths  # noqa: F401
from pipeline.names import VARIANT_TOKENS, NameTruth, base_name, is_right

ORIGINAL = r"C:\repos\projects\card-scanner-data\scripts\score_real_labels.py"

NAMES = [
    "", "   ", "Pikachu", "pikachu", "PIKACHU", "Charizard ex", "Charizard EX", "Pikachu VMAX", "Pikachu V", "Mewtwo VSTAR",
    "Mewtwo & Mew-GX", "Mewtwo & Mew GX", "Flabébé", "Flabébé ex", "Pokémon Center", "Pokémon", "Mr. Mime", "Mr Mime",
    "Farfetch'd", "Farfetch’d", "Farfetch'd ex", "Nidoran♀", "Nidoran♂", "Nidoran ♀", "Professor's Research", "Professor’s Research",
    "Misty's Tears", "Brock’s Scouting", "Lillie's Determination", "Ethan's Adventure", "N's Zoroark ex", "Zoroark-GX",
    "Tapu Koko-GX", "Tapu Koko Star", "Gold Star Rayquaza", "Rayquaza ☆", "Rayquaza δ", "Rayquaza Delta Species", "δ Delta Species Dialga",
    "Vunion Charizard", "Venusaur & Snivy-GX", "Ho-Oh", "Ho-Oh ex", "Porygon-Z", "Type: Null", "Unown [A]", "Pikachu (Flying)",
    "Pikachu  with   spaces", "Pikachu\tTab", "Dark Raichu", "Rocket's Mewtwo ex", "Árbol", "Ex", "V", "ex v vmax", "Eevee's Bag",
    "Bill's Transfer", "Mime Jr.", "Kleavor VSTAR", "Arceus VSTAR & Dialga V",
]


def load_original():
    spec = importlib.util.spec_from_file_location("score_real_labels_original", ORIGINAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BaseName(unittest.TestCase):
    def test_pinned_examples(self):
        self.assertEqual(base_name("Charizard ex"), "charizard")
        self.assertEqual(base_name("Pikachu VMAX"), "pikachu")
        self.assertEqual(base_name("Mewtwo & Mew-GX"), "mewtwo mew")
        self.assertEqual(base_name("Professor’s Research"), "professor research")
        self.assertEqual(base_name("Farfetch’d"), "farfetch d")
        self.assertEqual(base_name("Rayquaza δ"), "rayquaza")
        self.assertEqual(base_name(""), "")

    def test_the_variant_tokens_are_the_study_set(self):
        self.assertEqual(set(VARIANT_TOKENS), {"ex", "v", "vmax", "vstar", "vunion", "gx", "star", "delta", "δ"})

    def test_there_are_enough_names_to_compare(self):
        self.assertGreaterEqual(len(NAMES), 40)

    @unittest.skipUnless(os.path.exists(ORIGINAL), f"the study script is absent ({ORIGINAL}); the port is not compared with it")
    def test_the_port_equals_the_study_original_on_every_name(self):
        original = load_original()
        for name in NAMES:
            with self.subTest(name=name):
                self.assertEqual(base_name(name), original.base_name(name))
                self.assertEqual(base_name(name), " ".join(t for t in original.normalise(name).split() if t not in original.VARIANT_TOKENS))


class RightRule(unittest.TestCase):
    def test_an_id_truth_is_exact_equality(self):
        self.assertTrue(is_right("c-1", "c-1", {}))
        self.assertFalse(is_right("c-1", "c-2", {"c-1": "Anything"}))

    def test_a_name_truth_is_right_through_an_accept_id(self):
        truth = NameTruth("pikachu", frozenset({"a-1", "b-2"}))
        self.assertTrue(is_right("b-2", truth, {}))
        self.assertFalse(is_right("c-3", truth, {}))

    def test_a_name_truth_is_right_through_a_new_card_with_the_same_base_name(self):
        truth = NameTruth("pikachu", frozenset())
        self.assertTrue(is_right("n-1", truth, {"n-1": "Pikachu ex"}))
        self.assertFalse(is_right("n-1", truth, {"n-1": "Raichu"}))
        self.assertFalse(is_right("n-2", truth, {"n-1": "Pikachu"}))  # not a new card, not accepted

    def test_a_name_truth_rejects_an_empty_name(self):
        with self.assertRaises(ValueError):
            NameTruth("", frozenset())


if __name__ == "__main__":
    unittest.main()
