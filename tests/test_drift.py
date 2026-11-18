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


def js_const(text, name):
    """Numeric value of `[export] const NAME = <number>;` in JS source, whatever the spacing. Fails if absent."""
    m = re.search(r"const\s+%s\s*=\s*(-?\d+(?:\.\d+)?)\s*;" % re.escape(name), text)
    assert m, f"{name} not found in source"
    return float(m.group(1)) if "." in m.group(1) else int(m.group(1))


def js_confirm_rows(text):
    """CONFIRM_ROWS as ((minHeight, minSim, minGap), ...) in source order. Fails if absent or empty."""
    block = re.search(r"const\s+CONFIRM_ROWS\s*=\s*Object\.freeze\(\[(.*?)\]\)\s*;", text, re.S)
    assert block, "CONFIRM_ROWS not found in source"
    rows = re.findall(
        r"minHeight:\s*(\d+)\s*,\s*minSim:\s*(\d+(?:\.\d+)?)\s*,\s*minGap:\s*(\d+(?:\.\d+)?)", block.group(1)
    )
    assert rows, "CONFIRM_ROWS has no rows"
    return tuple((int(h), float(a), float(g)) for h, a, g in rows)


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

    def test_fallback_accept_pins_match_fusion_js(self):
        text = source("fusion.js")
        self.assertEqual(js_const(text, "MIN_ACCEPT_SIM"), PINS["accept"]["minSim"])
        self.assertEqual(js_const(text, "CLEAR_GAP"), PINS["accept"]["clearGap"])
        self.assertEqual(js_const(text, "HIGH_SIM"), PINS["accept"]["highSim"])

    def test_confirm_rule_matches_fusion_js(self):
        from pipeline import gate

        text = source("fusion.js")
        self.assertEqual(js_const(text, "CONFIRM_MIN_HEIGHT"), gate.CONFIRM_MIN_HEIGHT)
        self.assertEqual(js_const(text, "TITLE_NAME_PENALTY"), gate.TITLE_NAME_PENALTY)
        self.assertEqual(js_confirm_rows(text), gate.CONFIRM_ROWS)

    def test_unknown_height_row_matches_fusion_js(self):
        from pipeline import gate

        text = source("fusion.js")
        rows = js_confirm_rows(text)
        m = re.search(r"const\s+UNKNOWN_HEIGHT_ROW\s*=\s*CONFIRM_ROWS\.find\(\(r\)\s*=>\s*r\.minHeight\s*===\s*(\d+)\)", text)
        assert m, "UNKNOWN_HEIGHT_ROW not found in source"
        js_row = next(r for r in rows if r[0] == int(m.group(1)))
        self.assertEqual(js_row, gate.UNKNOWN_HEIGHT_ROW)

    def test_digital_only_pattern_matches_search_js(self):
        from pipeline import new_sets

        m = re.search(r"const\s+DIGITAL_ONLY\s*=\s*/(.*?)/\s*;", source("search.js"))
        self.assertIsNotNone(m, "DIGITAL_ONLY not found in search.js")
        self.assertEqual(new_sets.DIGITAL_ONLY.pattern, m.group(1))
        self.assertEqual(new_sets.DIGITAL_ONLY.flags & re.IGNORECASE, 0)

    def test_const_parser_tolerates_format_and_fails_on_missing(self):
        self.assertEqual(js_const("export const CLEAR_GAP   =0.5 ;", "CLEAR_GAP"), 0.5)
        self.assertEqual(js_const("const CLEAR_GAP = 0.02;  // x", "CLEAR_GAP"), 0.02)
        with self.assertRaises(AssertionError):
            js_const("const OTHER = 1;", "CLEAR_GAP")
        with self.assertRaises(AssertionError):
            js_confirm_rows("const CONFIRM_ROWS = [];")


if __name__ == "__main__":
    unittest.main()
