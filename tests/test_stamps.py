"""Stamped TCGplayer products joined to one catalog card, and EX-era prices keyed by catalog id.

Fixtures under tests/fixtures/prices are cut from public TCGCSV, TCGdex and catalog meta.json files.
"""
import json
import os
import unittest

import _paths

from pipeline import stamps

FIXTURES = os.path.join(_paths.ROOT, "tests", "fixtures", "prices")


def fixture(name):
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as f:
        return json.load(f)


def catalog(extra_cards=(), extra_sets=()):
    return stamps.Catalog(fixture("catalog-meta.json") + list(extra_cards), fixture("tcgdex-sets.json") + list(extra_sets))


GROUPS = fixture("tcgcsv.json")


def run_groups(cat, report=None):
    report = {} if report is None else report
    rows, by = [], {}
    for group_id, files in GROUPS.items():
        rows += stamps.stamped_rows(cat, int(group_id), files["products"], files["prices"], report)
        by.update(stamps.by_catalog(cat, int(group_id), files["products"], files["prices"], report))
    return rows, by, report


class StampTags(unittest.TestCase):
    def setUp(self):
        self.names = catalog().set_names

    def check(self, cases, group_id=1539):
        for name, want in cases:
            self.assertEqual(stamps.stamp_of(name, group_id, self.names), want, name)

    def test_each_listed_tag_maps_to_its_stamp(self):
        self.check([
            ("Altaria - BW48 (Prerelease)", "prerelease"),
            ("Clefable (Prerelease)", "prerelease"),
            ("Machamp (XY Evolutions Prerelease)", "prerelease"),
            ("Crobat - BW51 (Team Plasma) (Prerelease)", "prerelease"),
            ("Altaria - BW48 (Prerelease) [Staff]", "staff"),
            ("Raikou - SM150 (Prerelease) (Staff)", "staff"),
            ("Finneon - SWSH240 (Prerelease Staff)", "staff"),
            ("Sunflora - SWSH269 (Prerelease) [STAFF]", "staff"),
            ("Champions Festival - XY27 (2014 Staff)", "staff"),
            ("Tropical Tidal Wave - HGSS18 (Worlds 10) [Staff]", "staff"),
            ("Judge - 78/95 (League Promo)", "league"),
            ("Volcanion - 25/114 (Pokemon League)", "league"),
            ("Tsareena - 20/149 (League Challenge) [1st Place]", "league"),
            ("Ninja Boy - 103/114 (League Cup)", "league"),
            ("Greninja - 40/122 (League) [2nd Place]", "league"),
            ("Misty's Determination - 104/122 (Regional Championships)", "championship"),
            ("Professor Turo's Scenario (Regional Championship)", "championship"),
            ("Parallel City - 145/162 (City Championships)", "championship"),
            ("Chaos Tower - 92/124 (National Championship Promo)", "championship"),
            ("Buddy-Buddy Poffin - 144/162 (North America International Championship)", "championship"),
            ("Champions Festival - BW95 (Worlds 13)", "championship"),
            ("Champions Festival - SM148 (World Championships 2018)", "championship"),
            ("Pikachu - 225 (World Championship 2025)", "championship"),
            ("Pikachu - 027 (Pokemon Center Exclusive)", "pokemoncenter"),
            ("Magneton - 159 (Pok\u00e9mon Center Exclusive)", "pokemoncenter"),
            ("Flutter Mane - 097 (Pokemon Center)", "pokemoncenter"),
            ("Rapidash [Pokemon Center NY]", "pokemoncenter"),
            ("Pikachu [W Stamped]", "wstamp"),
            ("Eevee ex - 075/131 (Prismatic Evolutions Stamp)", "setlogo"),
            ("Snorlax - 131/185 (Vivid Voltage Stamped)", "setlogo"),
            ("Bulbasaur - SM198 (Detective Pikachu Stamped)", "setlogo"),
            ("Lucario (Scarlet & Violet Stamp)", "setlogo"),
            ("Larvitar (Delta Species Stamp)", "setlogo"),
            ("Team Rocket's Articuno - 051/182 (Cosmos Holo) (Destined Rivals Stamp)", "setlogo"),
        ])

    def test_every_prize_pack_product_is_a_play_pokemon_stamp(self):
        self.check([
            ("Kricketune V - 006/163", "playpokemon"),
            ("Zekrom ex - 034/086 (Left Stamp)", "playpokemon"),
            ("Zekrom ex - 034/086 (Right Stamp)", "playpokemon"),
        ], group_id=stamps.PRIZE_PACK_GROUP)

    def test_trainer_names_and_unlisted_tags_are_not_stamps(self):
        self.check([
            ("Pokemon Center", None),
            ("Pok\u00e9mon Center Lady", None),
            ("Professor Oak", None),
            ("Championship Arena", None),
            ("Prerelease Pikachu", None),
            ("Flutter Mane - 097", None),
            ("Kricketune V - 006/163", None),
            ("Zekrom ex - 034/086 (Left Stamp)", None),
            ("Pikachu - 58/102 (E3 Stamped)", None),
            ("Primarina - 020/084 (What's Your Favorite Stamp)", None),
            ("Sprigatito - 012/193 (Pokemon Horizons Stamp)", None),
            ("Detective Pikachu - SM190 (Stamped)", None),
            ("Crobat - 56/149 (Prerelease Kit Exclusive)", None),
            ("Bulbasaur - SM198 (In-Store Event Promo)", None),
            ("Victory Cup - BW31 (Battle Road Autumn 2011) [1st Place]", None),
            ("Clefairy - 381/SM-P (Dream League)", None),
            ("Ace Trainer - 69/98 (International Challenge)", None),
        ])

    def test_the_tag_is_the_text_of_the_tags_after_the_number(self):
        for name, want in [
            ("Tsareena - 20/149 (League Challenge) [1st Place]", "League Challenge 1st Place"),
            ("Teal Mask Ogerpon ex - 025/167 (Prize Pack Series 6)", "Prize Pack Series 6"),
            ("Zekrom ex - 034/086 (Left Stamp)", "Left Stamp"),
            ("Bibarel - 121/172 (Cosmos Holo)", "Cosmos Holo"),
            ("Boss's Orders [Cyrus] - 132/172 (Cosmos Holo)", "Cosmos Holo"),
            ("Metal Energy (Special) - 80/90 (League Promo)", "League Promo"),
            ("Magneton - 159 (Pok\u00e9mon Center Exclusive)", "Pokemon Center Exclusive"),
            ("Raikou - SM150 (Prerelease)  (Staff)", "Prerelease Staff"),
            ("Pikachu [W Stamped]", "W Stamped"),
            ("Larvitar (Delta Species Stamp)", "Delta Species Stamp"),
            ("Kricketune V - 006/163", None),
            ("Boss's Orders [Cyrus] - 132/172", None),
        ]:
            self.assertEqual(stamps.stamp_tag(name), want, name)

    def test_base_name_drops_every_tag_then_the_number(self):
        self.assertEqual(stamps.base_name("Deoxys (Delta Species - Attack Forme)"), "Deoxys")
        self.assertEqual(stamps.base_name("Boss's Orders [Cyrus] - 132/172 (Regional Championships)"), "Boss's Orders")
        self.assertEqual(stamps.base_name("Altaria - BW48 (Prerelease) [Staff]"), "Altaria")
        self.assertEqual(stamps.base_name("Pikachu [W Stamped]"), "Pikachu")


class Join(unittest.TestCase):
    def setUp(self):
        self.cat = catalog()

    def test_a_promo_prefix_picks_the_promo_set(self):
        self.assertEqual(self.cat.match("Altaria", "BW48", 1407), ["bwp-BW48"])
        self.assertEqual(self.cat.match("Raikou", "SM150", 1861), ["smp-SM150"])
        self.assertEqual(self.cat.match("Finneon", "SWSH240", 2545), ["swshp-SWSH240"])

    def test_a_bare_number_is_an_svp_promo_only_in_the_sv_promo_group(self):
        self.assertEqual(self.cat.match("Pikachu", "027", stamps.SV_PROMO_GROUP), ["svp-027"])
        self.assertEqual(self.cat.match("Pikachu", "27", stamps.SV_PROMO_GROUP), ["svp-027"])
        self.assertEqual(self.cat.match("Pikachu", "027", 2374), [])

    def test_a_printed_total_picks_the_sets_and_the_name_picks_the_card(self):
        self.assertEqual(self.cat.match("Judge", "078/095", 1539), ["hgss2-78"])
        self.assertEqual(self.cat.match("Pikachu", "60/64", 1418), ["base2-60"])
        self.assertEqual(self.cat.match("Clefable", "01/64", 1418), ["base2-1"])
        self.assertEqual(self.cat.match("Larvitar", "73/113", 2374), ["ex11-73"])
        self.assertEqual(self.cat.match("Rapidash", "51/53", 1418), ["basep-51"])
        self.assertEqual(self.cat.match("Pikachu", "61/64", 1418), [])
        self.assertEqual(self.cat.match("Raichu", "60/64", 1418), [])
        self.assertEqual(self.cat.match("Pokemon Communication", "152a/181", 1539), [])

    def test_catalog_symbols_match_the_spelled_out_product_names(self):
        for name, number, want in [
            ("Nidoran F", "56/101", "ex15-56"),
            ("Charizard Star", "100/101", "ex15-100"),
            ("Delta Rainbow Energy", "88/101", "ex15-88"),
            ("Altaria ex", "90/101", "ex15-90"),
            ("Deoxys", "3/110", "ex13-3"),
        ]:
            self.assertEqual(self.cat.match(name, number, 2374), [want], name)


class StampedRows(unittest.TestCase):
    def test_rows_carry_the_catalog_id_stamp_tag_product_and_three_prices(self):
        rows, _, _ = run_groups(catalog())
        self.assertIn({"catalogId": "bwp-BW48", "stamp": "staff", "tag": "Prerelease Staff", "productId": 96566,
                       "subtype": "Holofoil", "market": 350.0, "mid": 229.99, "low": 229.99}, rows)
        self.assertIn({"catalogId": "bwp-BW48", "stamp": "prerelease", "tag": "Prerelease", "productId": 83520,
                       "subtype": "Holofoil", "market": 45.43, "mid": 109.99, "low": 54.0}, rows)
        self.assertIn({"catalogId": "swshp-SWSH240", "stamp": "staff", "tag": "Prerelease Staff", "productId": 680245,
                       "subtype": "Holofoil", "market": None, "mid": 99.0, "low": 99.0}, rows)
        self.assertIn({"catalogId": "sv08.5-075", "stamp": "setlogo", "tag": "Prismatic Evolutions Stamp", "productId": 618089,
                       "subtype": "Holofoil", "market": 8.83, "mid": 9.99, "low": 7.96}, rows)
        self.assertIn({"catalogId": "hgss2-78", "stamp": "league", "tag": "League Promo", "productId": 123183,
                       "subtype": "Reverse Holofoil", "market": 106.33, "mid": 208.0, "low": 179.99}, rows)
        self.assertIn({"catalogId": "basep-51", "stamp": "pokemoncenter", "tag": "Pokemon Center NY", "productId": 244347,
                       "subtype": "Normal", "market": 119.87, "mid": 207.0, "low": 199.99}, rows)
        self.assertIn({"catalogId": "swsh5-6", "stamp": "playpokemon", "tag": None, "productId": 475973,
                       "subtype": "Holofoil", "market": 4.51, "mid": 5.39, "low": 4.49}, rows)

    def test_products_sharing_a_card_stamp_and_subtype_differ_by_tag(self):
        rows, _, _ = run_groups(catalog())
        tsareena = sorted(r["tag"] for r in rows if r["catalogId"] == "sm1-20")
        self.assertEqual(tsareena, ["League Challenge 1st Place", "League Challenge 2nd Place"])
        zekrom = sorted(r["tag"] for r in rows if r["catalogId"] == "sv10.5b-034")
        self.assertEqual(zekrom, ["Left Stamp", "Right Stamp"])
        keys = [(r["catalogId"], r["stamp"], r["subtype"], r["tag"]) for r in rows]
        self.assertEqual(len(keys), len(set(keys)))

    def test_unstamped_products_and_trainer_cards_give_no_row(self):
        rows, _, _ = run_groups(catalog())
        products = {r["productId"] for r in rows}
        # Pokemon Center trainer, plain SVP promo, E3, "What's Your Favorite", in-store event, Battle Road.
        for product_id in (88213, 543946, 123633, 705526, 188320, 90353):
            self.assertNotIn(product_id, products, product_id)
        self.assertFalse([r for r in rows if r["catalogId"] == "basep-40"])
        self.assertTrue(all(r["stamp"] in stamps.STAMPS for r in rows))

    def test_the_report_counts_every_stamped_product(self):
        _, _, report = run_groups(catalog())
        staff = report["stamped"]["staff"]
        self.assertEqual(staff["products"], 6)
        self.assertEqual(staff["joined"], 5)
        self.assertEqual(staff["unmatched"], 1)
        self.assertEqual(report["stamped"]["playpokemon"]["joined"], 3)

    def test_an_ambiguous_match_is_dropped_and_counted(self):
        twin_set = {"id": "zz1", "name": "Synthetic Twin", "cardCount": {"official": 64}}
        twin = {"id": "zz1-60", "name": "Pikachu", "setCode": "zz1", "number": "60"}
        rows, _, report = run_groups(catalog([twin], [twin_set]))
        self.assertFalse([r for r in rows if r["productId"] == 126209])
        self.assertEqual(report["stamped"]["wstamp"]["ambiguous"], 1)
        self.assertEqual(report["stamped"]["wstamp"]["joined"], 0)

    def test_a_product_with_no_price_gives_no_row(self):
        unpriced = {"results": [{"productId": 1, "name": "Altaria - BW48 (Prerelease)", "groupId": 1407,
                                 "extendedData": [{"name": "Number", "value": "BW48"}]}]}
        prices = {"results": [{"productId": 1, "subTypeName": "Holofoil", "marketPrice": None, "midPrice": 0, "lowPrice": None}]}
        report = {}
        self.assertEqual(stamps.stamped_rows(catalog(), 1407, unpriced, prices, report), [])
        self.assertEqual(report["stamped"]["prerelease"]["unpriced"], 1)


class ByCatalog(unittest.TestCase):
    def test_ex_set_logo_cards_get_their_subtype_prices_by_catalog_id(self):
        _, by, _ = run_groups(catalog())
        self.assertEqual(by["ex15-1"], {"Holofoil": 72.38, "Reverse Holofoil": 173.65})
        self.assertEqual(by["ex15-56"], {"Normal": 0.84, "Reverse Holofoil": 14.59})
        self.assertEqual(by["ex15-88"], {"Normal": 1.14, "Reverse Holofoil": 23.99})
        self.assertEqual(by["ex15-100"], {"Holofoil": 39500.0})
        self.assertEqual(by["ex13-3"], {"Holofoil": 82.37, "Reverse Holofoil": 1498.99})
        self.assertNotIn("ex15-2", by)
        self.assertTrue(all(card.split("-")[0] in stamps.EX_STAMP_GROUPS.values() for card in by))

    def test_the_report_counts_joined_and_unmatched_cards_per_set(self):
        _, _, report = run_groups(catalog())
        ex15 = report["byCatalog"]["ex15"]
        self.assertEqual(ex15["cards"], 7)
        self.assertEqual(ex15["joined"], 5)
        self.assertEqual(ex15["unmatched"], 2)
        self.assertEqual(ex15["reverse"], 3)

    def test_only_the_seven_ex_groups_are_read(self):
        self.assertEqual(stamps.EX_STAMP_GROUPS, {1398: "ex10", 1429: "ex11", 1378: "ex12", 1379: "ex13",
                                                  1395: "ex14", 1411: "ex15", 1383: "ex16"})
        report = {}
        files = GROUPS["1539"]
        self.assertEqual(stamps.by_catalog(catalog(), 1539, files["products"], files["prices"], report), {})


class Load(unittest.TestCase):
    def test_the_catalog_loads_from_a_meta_file(self):
        cat = stamps.Catalog.load(os.path.join(FIXTURES, "catalog-meta.json"), fixture("tcgdex-sets.json"))
        self.assertEqual(cat.match("Altaria", "BW48", 1407), ["bwp-BW48"])

    def test_an_empty_catalog_or_set_list_is_refused(self):
        with self.assertRaisesRegex(ValueError, "catalog"):
            stamps.Catalog([], fixture("tcgdex-sets.json"))
        with self.assertRaisesRegex(ValueError, "sets"):
            stamps.Catalog(fixture("catalog-meta.json"), [])


if __name__ == "__main__":
    unittest.main()
