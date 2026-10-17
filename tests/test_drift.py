"""Checks pins.json against the extension source it was copied from. Skipped in a data repo, where that source is absent."""
import json
import os
import re
import unittest

import _paths

LIB = os.path.join(os.path.dirname(_paths.ROOT), "extension-wasm", "lib")
with open(os.path.join(_paths.ROOT, "pins.json"), encoding="utf-8") as _f:
    PINS = json.load(_f)


def source(name):
    with open(os.path.join(LIB, name), encoding="utf-8") as f:
        return f.read()


@unittest.skipUnless(os.path.isdir(LIB), "extension source not next to this folder")
class Drift(unittest.TestCase):
    def test_asset_pins_match_setup_js(self):
        text = source("setup.js")
        for asset in PINS["assets"]:
            block = re.search(r'name: "%s",.*?\}' % re.escape(asset["name"]), text, re.S)
            self.assertIsNotNone(block, asset["name"])
            self.assertIn(asset["sha256"], block.group(0), asset["name"])
            self.assertIn(f'release: "{asset["release"]}"', block.group(0), asset["name"])

    def test_embedding_constants_match_embed_js(self):
        text = source("embed.js")
        self.assertIn(f"EMBEDDING_VERSION = {PINS['cropVersion']};", text)
        self.assertIn("ART_CROP_FRAC = 0.45;", text)
        self.assertIn("CLIP_SIZE = 224;", text)

    def test_reference_rule_matches_references_js(self):
        text = source("references.js")
        self.assertIn(f"REFERENCE_MIN = {PINS['reference']['min']};", text)
        self.assertIn(f"REFERENCE_MARGIN = {PINS['reference']['margin']};", text)
        self.assertIn(f"EMBEDDING_DIM = {PINS['dim']};", text)

    def test_high_tier_matches_fusion_js(self):
        text = source("fusion.js")
        self.assertIn(f"HIGH_TIER_GAP  = {PINS['highTier']['gap']};", text)
        self.assertIn(f"HIGH_TIER_SIM  = {PINS['highTier']['sim']};", text)

    def test_accept_rule_matches_fusion_js(self):
        from pipeline import gate

        text = source("fusion.js")
        self.assertEqual((gate.ACCEPT_MIN, gate.CLEAR_GAP, gate.HIGH_SIM), (PINS["accept"]["minSim"], PINS["accept"]["clearGap"], PINS["accept"]["highSim"]))
        self.assertIn(f"MIN_ACCEPT_SIM = {gate.ACCEPT_MIN};", text)
        self.assertIn(f"CLEAR_GAP      = {gate.CLEAR_GAP};", text)
        self.assertIn(f"HIGH_SIM       = {gate.HIGH_SIM};", text)


if __name__ == "__main__":
    unittest.main()
