import contextlib
import io
import json
import unittest
import urllib.error
from unittest import mock

import _paths  # noqa: F401
import numpy as np

from pipeline import new_sets

KNOWN = {"base1-1", "base1-2", "sv9-1"}
SETS = [
    {"id": "base1", "name": "Base Set", "cardCount": {"total": 2}},
    {"id": "sv9", "name": "Journey Together", "cardCount": {"total": 3}},
    {"id": "sv10", "name": "Destined Rivals", "cardCount": {"total": 2}},
]
DETAIL = {
    "sv9": {"name": "Journey Together", "cards": [
        {"id": "sv9-1", "localId": "1", "name": "Pikachu", "image": "https://assets.tcgdex.net/en/sv/sv9/1"},
        {"id": "sv9-2", "localId": "2", "name": "Raichu", "image": "https://assets.tcgdex.net/en/sv/sv9/2"},
        {"id": "sv9-3", "localId": "3", "name": "No Picture"},
    ]},
    "sv10": {"name": "Destined Rivals", "cards": [
        {"id": "sv10-1", "localId": "1", "name": "Mew", "image": "https://assets.tcgdex.net/en/sv/sv10/1"},
        {"id": "sv10-2", "localId": 2, "name": "Eevee", "image": "https://assets.tcgdex.net/en/sv/sv10/2"},
    ]},
}
TAG = {"model": "0dba1302580da2a6", "crop": 3}
OTHER_TAG = {"model": "1111111111111111", "crop": 3}
PTCG = "https://raw.githubusercontent.com/PokemonTCG/pokemon-tcg-data/master/cards/en/"


def not_found(url):
    """What the real fetch raises for a 404 (urllib.request.urlopen)."""
    return urllib.error.HTTPError(url, 404, "Not Found", {}, None)


def asked_files(net):
    return [r[len(PTCG):-len(".json")] for r in net.requests if r.startswith(PTCG)]


def ptcg_card(sid, number, name, large=None):
    return {"id": f"{sid}-{number}", "name": name, "number": number,
            "images": {"large": large or f"https://images.pokemontcg.io/{sid}/{number}_hires.png"}}


def header_of(data):
    return json.loads(data[4 : 4 + int.from_bytes(data[:4], "little")])


class Net:
    def __init__(self, details=DETAIL, sets=SETS, bad_images=(), ptcg=None, ptcg_error=None, list_failures=0):
        """ptcg: {set id: card list} served as static files; any other file is a 404. ptcg_error: {set id: how many
        requests for its file fail with a 500 before it is served}. list_failures: how many requests for the set list
        fail with a 503 before it is served."""
        self.requests, self.details, self.sets, self.bad_images = [], details, sets, set(bad_images)
        self.ptcg, self.ptcg_error, self.list_failures = ptcg or {}, dict(ptcg_error or {}), list_failures

    def json(self, url):
        self.requests.append(url)
        if url.startswith(PTCG):
            sid = url[len(PTCG):-len(".json")]
            if self.ptcg_error.get(sid, 0) > 0:
                self.ptcg_error[sid] -= 1
                raise urllib.error.HTTPError(url, 500, "Server Error", {}, None)
            if sid not in self.ptcg:
                raise not_found(url)
            return self.ptcg[sid]
        if url.endswith("/sets"):
            if self.list_failures > 0:
                self.list_failures -= 1
                raise urllib.error.HTTPError(url, 503, "Service Unavailable", {}, None)
            return self.sets
        code = url.rsplit("/", 1)[1]
        if code not in self.details:
            raise OSError("down")
        return self.details[code]

    def image(self, url):
        self.requests.append(url)
        if url in self.bad_images:
            raise OSError("404")
        return url.encode()


def embedder(data, trusted=False):
    assert trusted
    v = np.zeros(512, dtype=np.float32)
    v[sum(data) % 512] = 1.0
    return v


def run(net=None, previous=None, known=KNOWN, cap=500, tag=TAG):
    net = net or Net()
    out = new_sets.build(known, previous, embedder, fetch=net.json, fetch_image=net.image, sleep=lambda s: None, cap=cap, tag=tag)
    return net, out


class FindNew(unittest.TestCase):
    def test_only_cards_missing_from_the_catalog_are_found_and_complete_sets_are_not_opened(self):
        net = Net()
        cards, sets = new_sets.find_new_cards(KNOWN, net.json, lambda s: None)
        self.assertEqual([c["id"] for c in cards], ["sv10-1", "sv10-2", "sv9-2"])  # newest set first; sv9-3 has no picture
        self.assertEqual(sets, ["sv10", "sv9"])
        self.assertNotIn(f"{new_sets.TCGDEX}/sets/base1", net.requests)  # base1: 2 known of 2
        self.assertEqual(cards[2], {"id": "sv9-2", "name": "Raichu", "set": "sv9", "setName": "Journey Together", "number": "2", "image": "https://assets.tcgdex.net/en/sv/sv9/2/high.webp"})
        self.assertEqual(cards[1]["number"], "2")

    def test_cap_and_a_failing_set(self):
        cards, _ = new_sets.find_new_cards(KNOWN, Net().json, lambda s: None, cap=2)
        self.assertEqual(len(cards), 2)
        net = Net(details={"sv10": DETAIL["sv10"]})  # sv9 detail fails
        cards, _ = new_sets.find_new_cards(KNOWN, net.json, lambda s: None)
        self.assertEqual([c["id"] for c in cards], ["sv10-1", "sv10-2"])

    def test_a_quiet_night_is_one_request(self):
        net = Net(sets=[SETS[0]])
        cards, _ = new_sets.find_new_cards(KNOWN, net.json, lambda s: None)
        self.assertEqual((cards, net.requests), ([], [f"{new_sets.TCGDEX}/sets"]))

    def test_pocket_sets_are_not_opened(self):
        sets = SETS + [{"id": "A1", "name": "Genetic Apex", "cardCount": {"total": 286}}, {"id": "B2", "name": "Fantastical Parade", "cardCount": {"total": 9}},
                       {"id": "P-A", "name": "Promo-A", "cardCount": {"total": 9}}]
        net = Net(sets=sets)
        _, found_sets = new_sets.find_new_cards(KNOWN, net.json, lambda s: None)
        self.assertEqual(found_sets, ["sv10", "sv9"])
        for code in ("A1", "B2", "P-A"):
            self.assertNotIn(f"{new_sets.TCGDEX}/sets/{code}", net.requests)

    def test_the_newest_set_is_taken_whole_before_an_older_one(self):
        # The TCGdex list is in release order, so the last entry is the newest.
        cards, sets = new_sets.find_new_cards(KNOWN, Net().json, lambda s: None, cap=2)
        self.assertEqual(([c["id"] for c in cards], sets), (["sv10-1", "sv10-2"], ["sv10"]))

    def test_a_cut_off_set_is_completed_on_the_next_run_without_repeats(self):
        def card(code, n):
            return {"id": f"{code}-{n}", "localId": str(n), "name": f"{code} card {n}", "image": f"https://assets.tcgdex.net/en/x/{code}/{n}"}

        details = {"old1": {"name": "Old", "cards": [card("old1", 1), card("old1", 2)]}, "new1": {"name": "New", "cards": [card("new1", 1), card("new1", 2), card("new1", 3)]}}
        sets = [{"id": "old1", "name": "Old", "cardCount": {"total": 2}}, {"id": "new1", "name": "New", "cardCount": {"total": 3}}]
        net = Net(details=details, sets=sets)
        _, first = run(net, known=set(), cap=2)
        self.assertEqual(first["added"], ["new1-1", "new1-2"])  # the newest set is cut off
        _, second = run(Net(details=details, sets=sets), previous=first["bytes"], known=set(), cap=2)
        self.assertEqual(second["added"], ["new1-3", "old1-1"])  # it is completed first, nothing repeats
        ids = [c["id"] for c in new_sets.read_extra(second["bytes"])[0]]
        self.assertEqual(ids, ["new1-1", "new1-2", "new1-3", "old1-1"])
        self.assertEqual(len(ids), len(set(ids)))

    def test_the_digital_only_pattern_is_the_extensions(self):
        self.assertEqual(new_sets.DIGITAL_ONLY.pattern, r"^(A\d|B\d|P-A)")


def fallback_net(ptcg, **kw):
    """Only sv9 is incomplete; it has one card without a TCGdex picture (sv9-3, "No Picture", number 3)."""
    return Net(sets=[SETS[1]], ptcg=ptcg, **kw)


def fallback(net, cap=500):
    cards, sets = new_sets.find_new_cards(KNOWN, net.json, lambda s: None, cap=cap)
    return {c["id"]: c for c in cards}, sets


class PictureFallback(unittest.TestCase):
    def test_a_number_and_name_match_gives_the_pokemontcg_picture(self):
        net = fallback_net({"sv9": [ptcg_card("sv9", "3", "No Picture")]})
        cards, sets = fallback(net)
        self.assertEqual(cards["sv9-3"], {"id": "sv9-3", "name": "No Picture", "set": "sv9", "setName": "Journey Together", "number": "3",
                                          "image": "https://images.pokemontcg.io/sv9/3_hires.png"})
        self.assertEqual(sets, ["sv9"])
        self.assertEqual(net.requests[-1], f"{PTCG}sv9.json")

    def test_numbers_and_names_are_normalised_before_they_are_compared(self):
        details = {"sv9": {"name": "Journey Together", "cards": [{"id": "sv9-3", "localId": "003", "name": "Pokémon  Flabébé-V"}]}}
        net = Net(details=details, sets=[SETS[1]], ptcg={"sv9": [ptcg_card("sv9", "3", "POKEMON FLABEBE V")]})
        self.assertEqual(fallback(net)[0]["sv9-3"]["image"], "https://images.pokemontcg.io/sv9/3_hires.png")
        details = {"sv9": {"name": "Journey Together", "cards": [{"id": "sv9-3", "localId": "SV003", "name": "X"}]}}
        net = Net(details=details, sets=[SETS[1]], ptcg={"sv9": [ptcg_card("sv9", "sv3", "x")]})
        self.assertIn("sv9-3", fallback(net)[0])

    def test_a_name_mismatch_or_a_duplicate_key_is_skipped(self):
        cards, _ = fallback(fallback_net({"sv9": [ptcg_card("sv9", "3", "Other Name")]}))
        self.assertNotIn("sv9-3", cards)
        dup = [ptcg_card("sv9", "3", "No Picture"), ptcg_card("sv9", "03", "No  Picture", large="https://images.pokemontcg.io/sv9/dup.png")]
        cards, _ = fallback(fallback_net({"sv9": dup}))
        self.assertNotIn("sv9-3", cards)
        self.assertEqual(sorted(cards), ["sv9-2"])  # the set's other cards are untouched

    def test_only_pokemontcg_io_pictures_of_sane_length_are_accepted(self):
        for bad in ("https://evil.example/sv9/3_hires.png", "http://images.pokemontcg.io/sv9/3.png", "https://images.pokemontcg.io/" + "a" * 280):
            cards, _ = fallback(fallback_net({"sv9": [ptcg_card("sv9", "3", "No Picture", large=bad)]}))
            self.assertNotIn("sv9-3", cards, bad)

    def test_set_id_spellings_are_tried_in_order_until_one_has_cards(self):
        cases = [("sm7.5", "sm75", ["sm7.5", "sm75"]), ("swsh12.5gg", "swsh12pt5gg", ["swsh12.5gg", "swsh125gg", "swsh12pt5gg"]), ("sv03", "sv3", ["sv03", "sv3"])]
        for code, spelled, tried in cases:
            details = {code: {"name": "S", "cards": [{"id": f"{code}-9", "localId": "9", "name": "Nine"}]}}
            sets = [{"id": code, "name": "S", "cardCount": {"total": 1}}]
            net = Net(details=details, sets=sets, ptcg={spelled: [ptcg_card(spelled, "9", "Nine")]})
            cards, _ = fallback(net)
            self.assertEqual(cards[f"{code}-9"]["image"], f"https://images.pokemontcg.io/{spelled}/9_hires.png", code)
            self.assertEqual(asked_files(net), tried, code)  # a 404 moves on to the next spelling without a retry

    def test_an_alias_is_tried_after_the_rule_based_spellings(self):
        details = {"cel25cc": {"name": "Classic", "cards": [{"id": "cel25cc-1", "localId": "1", "name": "Pikachu"}]}}
        sets = [{"id": "cel25cc", "name": "Classic", "cardCount": {"total": 1}}]
        net = Net(details=details, sets=sets, ptcg={"cel25c": [ptcg_card("cel25c", "1", "Pikachu")]})
        cards, _ = fallback(net)
        self.assertEqual(cards["cel25cc-1"]["image"], "https://images.pokemontcg.io/cel25c/1_hires.png")
        self.assertEqual(asked_files(net), ["cel25cc", "cel25c"])
        self.assertEqual(new_sets.SET_ALIASES["2011bw"], "mcd11")
        self.assertEqual(len(new_sets.SET_ALIASES), 17)

    def test_an_alias_whose_names_do_not_match_yields_no_cards(self):
        details = {"bog": {"name": "Best", "cards": [{"id": "bog-1", "localId": "1", "name": "Pikachu"}]}}
        sets = [{"id": "bog", "name": "Best", "cardCount": {"total": 1}}]
        net = Net(details=details, sets=sets, ptcg={"bp": [ptcg_card("bp", "1", "Raichu")]})
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            cards, _ = fallback(net)
        self.assertEqual(cards, {})
        self.assertIn("fallback bog: file bp, matched 0 of 1 cards", log.getvalue())

    def test_a_404_is_not_retried_and_does_not_wait(self):
        pauses = []
        net = Net(details={"sv03": {"name": "S", "cards": [{"id": "sv03-9", "localId": "9", "name": "Nine"}]}},
                  sets=[{"id": "sv03", "name": "S", "cardCount": {"total": 1}}], ptcg={"sv3": [ptcg_card("sv3", "9", "Nine")]})
        new_sets.find_new_cards(set(), net.json, pauses.append)
        self.assertEqual(asked_files(net), ["sv03", "sv3"])
        self.assertNotIn(new_sets.RETRY_DELAY, pauses)

    def test_another_failure_is_retried_once_after_a_pause(self):
        pauses = []
        net = fallback_net({"sv9": [ptcg_card("sv9", "3", "No Picture")]}, ptcg_error={"sv9": 1})
        cards, _ = new_sets.find_new_cards(KNOWN, net.json, pauses.append)
        self.assertIn("sv9-3", [c["id"] for c in cards])
        self.assertEqual(asked_files(net), ["sv9", "sv9"])
        self.assertEqual(pauses.count(new_sets.RETRY_DELAY), 1)
        self.assertEqual(new_sets.RETRY_DELAY, 5)

    def test_a_log_line_names_the_file_and_the_matches(self):
        def lines(ptcg):
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                fallback(fallback_net(ptcg))
            return [x for x in log.getvalue().splitlines() if x.startswith("fallback")]

        self.assertEqual(lines({"sv9": [ptcg_card("sv9", "3", "No Picture")]}), ["fallback sv9: file sv9, matched 1 of 1 cards"])
        self.assertEqual(lines({"sv9": [ptcg_card("sv9", "3", "Other")]}), ["fallback sv9: file sv9, matched 0 of 1 cards"])
        self.assertEqual(lines({}), ["fallback sv9: file none, matched 0 of 1 cards"])

    def test_every_request_is_paced(self):
        pauses = []
        net = fallback_net({"sv9": [ptcg_card("sv9", "3", "No Picture")]})
        new_sets.find_new_cards(KNOWN, net.json, pauses.append)
        self.assertEqual(len(pauses), len(net.requests) - 1)  # every request but the set list
        self.assertTrue(all(p == new_sets.REQUEST_DELAY for p in pauses))

    def test_an_error_leaves_the_sets_pictureless_cards_for_the_next_night(self):
        net = fallback_net({"sv9": [ptcg_card("sv9", "3", "No Picture")]}, ptcg_error={"sv9": 99})
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            cards, _ = fallback(net)
        self.assertEqual(sorted(cards), ["sv9-2"])
        self.assertIn("skip picture fallback for set sv9", log.getvalue())
        self.assertEqual(asked_files(net), ["sv9", "sv9"])  # one retry, then no further spellings

    def test_a_set_without_any_pictureless_card_asks_pokemontcg_io_nothing(self):
        net = Net(sets=[SETS[2]])
        new_sets.find_new_cards(KNOWN, net.json, lambda s: None)
        self.assertEqual([r for r in net.requests if r.startswith(PTCG)], [])

    def test_the_per_run_set_budget(self):
        count = new_sets.MAX_FALLBACK_SETS + 5
        sets = [{"id": f"x{n}", "name": "S", "cardCount": {"total": 1}} for n in range(count)]
        details = {f"x{n}": {"name": "S", "cards": [{"id": f"x{n}-1", "localId": "1", "name": "One"}]} for n in range(count)}
        ptcg = {f"x{n}": [ptcg_card(f"x{n}", "1", "One")] for n in range(count)}
        net = Net(details=details, sets=sets, ptcg=ptcg)
        cards, _ = new_sets.find_new_cards(set(), net.json, lambda s: None)
        self.assertEqual(new_sets.MAX_FALLBACK_SETS, 150)
        self.assertEqual(len(cards), new_sets.MAX_FALLBACK_SETS)
        self.assertEqual(len(asked_files(net)), new_sets.MAX_FALLBACK_SETS)

    def test_the_card_cap_still_bounds_the_result(self):
        net = fallback_net({"sv9": [ptcg_card("sv9", "3", "No Picture")]})
        cards, _ = fallback(net, cap=1)
        self.assertEqual(list(cards), ["sv9-2"])


def one_set(code, cards, name="S"):
    """(sets list, details) for a single incomplete set of the given raw TCGdex cards."""
    return [{"id": code, "name": name, "cardCount": {"total": 99}}], {code: {"name": name, "cards": cards}}


def raw_card(code, n, name=None, image=True, card_id=None):
    out = {"id": card_id or f"{code}-{n}", "localId": str(n), "name": name or f"{code} card {n}"}
    if image:
        out["image"] = f"https://assets.tcgdex.net/en/x/{code}/{n}"
    return out


def quiet_find(net, **kw):
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        cards, sets = new_sets.find_new_cards(kw.pop("known", set()), net.json, lambda s: None, **kw)
    return [c["id"] for c in cards], log.getvalue()


class BadShapes(unittest.TestCase):
    """One bad set or card must not sink the night: it is skipped with a log line."""

    def test_a_set_detail_that_is_not_a_dict_is_skipped(self):
        sets = [{"id": "ok1", "name": "S", "cardCount": {"total": 9}}, {"id": "bad1", "name": "S", "cardCount": {"total": 9}}]
        net = Net(details={"ok1": {"name": "S", "cards": [raw_card("ok1", 1)]}, "bad1": ["not", "a", "dict"]}, sets=sets)
        ids, log = quiet_find(net)
        self.assertEqual(ids, ["ok1-1"])
        self.assertIn("skip set bad1", log)

    def test_cards_that_are_not_a_list_skip_the_set(self):
        sets, details = one_set("bad1", "nope")
        ids, log = quiet_find(Net(details=details, sets=sets))
        self.assertEqual(ids, [])
        self.assertIn("skip set bad1", log)

    def test_a_card_entry_that_is_not_a_dict_is_skipped_and_the_rest_kept(self):
        sets, details = one_set("s1", ["text", None, 5, ["x"], raw_card("s1", 1)])
        ids, log = quiet_find(Net(details=details, sets=sets))
        self.assertEqual(ids, ["s1-1"])
        self.assertIn("skip card", log)

    def test_odd_set_list_entries_are_skipped(self):
        sets = ["text", None, {"id": "s1", "name": "S", "cardCount": [1]}]
        net = Net(details={"s1": {"name": "S", "cards": [raw_card("s1", 1)]}}, sets=sets)
        ids, _ = quiet_find(net)
        self.assertEqual(ids, ["s1-1"])

    def test_odd_pokemontcg_payloads_are_skipped_and_good_rows_still_match(self):
        sets, details = one_set("sv9", [raw_card("sv9", 3, "No Picture", image=False)])
        good = ptcg_card("sv9", "3", "No Picture")
        junk = ["text", None, 7, {"number": "3", "name": "No Picture", "images": [1]}, {"number": 3, "name": "No Picture", "images": {"large": "x"}},
                {"number": "3", "name": None, "images": {"large": 5}}, {"number": "4", "name": "Other", "images": "large"}]
        for payload in ([junk[0]], [junk[3]], junk, junk + [good]):
            net = Net(details=details, sets=sets, ptcg={"sv9": payload})
            ids, _ = quiet_find(net)
            self.assertEqual(ids, ["sv9-3"] if good in payload else [], payload)

    def test_an_unexpected_error_anywhere_in_a_sets_fallback_skips_only_that_set(self):
        sets = [{"id": "a1", "name": "S", "cardCount": {"total": 9}}, {"id": "b1", "name": "S", "cardCount": {"total": 9}}]
        details = {"a1": {"name": "S", "cards": [raw_card("a1", 1, image=False), raw_card("a1", 2)]},
                   "b1": {"name": "S", "cards": [raw_card("b1", 1, image=False), raw_card("b1", 2)]}}
        net = Net(details=details, sets=sets, ptcg={"a1": [ptcg_card("a1", "1", "a1 card 1")], "b1": [ptcg_card("b1", "1", "b1 card 1")]})
        real = new_sets._name_key
        calls = {"n": 0}

        def flaky(name):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return real(name)

        with mock.patch.object(new_sets, "_name_key", flaky):
            ids, log = quiet_find(net)
        self.assertIn("skip picture fallback for set b1", log)  # b1 is walked first (newest)
        self.assertEqual(ids, ["b1-2", "a1-2", "a1-1"])  # within a set, cards with a TCGdex picture come first


class IdsAndHosts(unittest.TestCase):
    def test_an_id_listed_twice_or_by_two_sets_is_taken_once(self):
        sets, details = one_set("s1", [raw_card("s1", 1), raw_card("s1", 1), raw_card("s1", 2)])
        self.assertEqual(quiet_find(Net(details=details, sets=sets))[0], ["s1-1", "s1-2"])
        sets = [{"id": "ab", "name": "S", "cardCount": {"total": 9}}, {"id": "ab-c", "name": "S", "cardCount": {"total": 9}}]
        details = {"ab": {"name": "S", "cards": [raw_card("ab", 1, card_id="ab-c-1")]}, "ab-c": {"name": "S", "cards": [raw_card("ab-c", 1)]}}
        self.assertEqual(quiet_find(Net(details=details, sets=sets))[0], ["ab-c-1"])

    def test_ids_in_the_previous_file_are_never_added_again(self):
        sets, details = one_set("s1", [raw_card("s1", 1), raw_card("s1", 1), raw_card("s1", 2)])
        _, first = run(Net(details=details, sets=sets), known=set(), cap=1)
        self.assertEqual(first["added"], ["s1-1"])
        _, second = run(Net(details=details, sets=sets), previous=first["bytes"], known=set())
        self.assertEqual(second["added"], ["s1-2"])
        ids = [c["id"] for c in new_sets.read_extra(second["bytes"])[0]]
        self.assertEqual(ids, ["s1-1", "s1-2"])

    def test_a_pictureless_card_already_known_is_not_looked_up_or_added(self):
        sets, details = one_set("sv9", [raw_card("sv9", 3, "No Picture", image=False)])
        net = Net(details=details, sets=sets, ptcg={"sv9": [ptcg_card("sv9", "3", "No Picture")]})
        self.assertEqual(quiet_find(net)[0], ["sv9-3"])
        self.assertEqual(quiet_find(net, known={"sv9-3"})[0], [])

    def test_an_id_over_64_characters_is_rejected(self):
        long_id, ok_id = "s1-" + "x" * 62, "s1-" + "x" * 61
        self.assertEqual((len(long_id), len(ok_id)), (65, 64))
        sets, details = one_set("s1", [raw_card("s1", 1, card_id=long_id), raw_card("s1", 2, card_id=ok_id)])
        self.assertEqual(quiet_find(Net(details=details, sets=sets))[0], [ok_id])

    def test_an_id_must_start_with_its_sets_id_and_a_dash(self):
        sets, details = one_set("s1", [raw_card("s1", 1, card_id="other-1"), raw_card("s1", 2, card_id="s1x-2"), raw_card("s1", 3)])
        self.assertEqual(quiet_find(Net(details=details, sets=sets))[0], ["s1-3"])

    def test_a_digital_only_id_is_rejected_even_inside_a_physical_set(self):
        sets, details = one_set("P", [raw_card("P", 1, card_id="P-A-1"), raw_card("P", 2)])
        self.assertEqual(quiet_find(Net(details=details, sets=sets))[0], ["P-2"])

    def test_a_tcgdex_picture_must_come_from_assets_tcgdex_net(self):
        bad = [raw_card("s1", 1), raw_card("s1", 2), raw_card("s1", 3)]
        bad[0]["image"] = "https://evil.example/en/x/s1/1"
        bad[1]["image"] = "http://assets.tcgdex.net/en/x/s1/2"
        sets, details = one_set("s1", bad)
        net = Net(details=details, sets=sets)
        self.assertEqual(quiet_find(net)[0], ["s1-3"])
        self.assertEqual([r for r in net.requests if r.startswith(PTCG)], [])  # a wrong host is a rejection, not a missing picture

    def test_the_pokemontcg_fallback_host_is_still_restricted(self):
        self.assertEqual(new_sets.POKEMONTCG_IMAGE_HOST, "https://images.pokemontcg.io/")
        self.assertEqual(new_sets.TCGDEX_IMAGE_HOST, "https://assets.tcgdex.net/")


class SetListFailure(unittest.TestCase):
    """The set list is the one request every night depends on: retry it once, then skip the night quietly."""

    def test_one_failure_then_success_proceeds_normally(self):
        pauses = []
        net = Net(list_failures=1)
        out = new_sets.build(KNOWN, None, embedder, fetch=net.json, fetch_image=net.image, sleep=pauses.append, tag=TAG)
        self.assertEqual(out["added"], ["sv10-1", "sv10-2", "sv9-2"])
        self.assertEqual(pauses[0], new_sets.RETRY_DELAY)
        self.assertEqual(len([r for r in net.requests if r.endswith("/sets")]), 2)

    def test_two_failures_keep_the_previous_file_byte_for_byte_and_add_nothing(self):
        _, first = run(cap=1)
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            net, out = run(Net(list_failures=2), previous=first["bytes"])
        self.assertEqual((out["bytes"], out["added"], out["sets"], out["total"]), (first["bytes"], [], [], 1))
        self.assertIn("new sets skipped tonight: ", log.getvalue())
        self.assertEqual(len([r for r in net.requests if r.endswith("/sets")]), 2)  # one retry, no more

    def test_two_failures_with_no_previous_file_make_no_file(self):
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            _, out = run(Net(list_failures=2))
        self.assertEqual((out["bytes"], out["added"], out["sets"], out["total"]), (None, [], [], 0))
        self.assertIn("new sets skipped tonight: ", log.getvalue())

    def test_a_set_list_that_is_not_a_list_is_skipped_the_same_way(self):
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            _, out = run(Net(sets={"oops": 1}))
        self.assertEqual((out["bytes"], out["added"]), (None, []))
        self.assertIn("new sets skipped tonight: ", log.getvalue())


def raw_extra(header, body=b""):
    """A catalogExtra-shaped file with whatever header the test wants (not necessarily valid)."""
    data = json.dumps(header).encode("utf-8")
    return len(data).to_bytes(4, "little") + data + body


class Starvation(unittest.TestCase):
    def test_a_newest_set_whose_pictures_all_fail_does_not_starve_an_older_one(self):
        # The reviewer's stub: cap 500, a newest set of 1,000 cards whose pictures 404, an older set with 10 good cards.
        sets = [{"id": "old1", "name": "Old", "cardCount": {"total": 10}}, {"id": "new1", "name": "New", "cardCount": {"total": 1000}}]
        details = {"old1": {"name": "Old", "cards": [raw_card("old1", n) for n in range(1, 11)]},
                   "new1": {"name": "New", "cards": [raw_card("new1", n) for n in range(1, 1001)]}}
        net = Net(details=details, sets=sets, bad_images={f"https://assets.tcgdex.net/en/x/new1/{n}/high.webp" for n in range(1, 1001)})
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            _, out = run(net, known=set(), cap=500)
        self.assertEqual(out["added"], [f"old1-{n}" for n in range(1, 11)])
        self.assertEqual(len([r for r in net.requests if "/new1/" in r and r.endswith("high.webp")]), new_sets.SET_FAILURES)
        self.assertIn("skip the rest of set new1", log.getvalue())

    def test_a_set_with_a_few_failing_cards_still_gets_its_good_ones(self):
        cards = [raw_card("s1", n) for n in range(1, 7)]
        sets, details = one_set("s1", cards)
        net = Net(details=details, sets=sets, bad_images={"https://assets.tcgdex.net/en/x/s1/2/high.webp", "https://assets.tcgdex.net/en/x/s1/4/high.webp"})
        _, out = run(net, known=set())
        self.assertEqual(out["added"], ["s1-1", "s1-3", "s1-5", "s1-6"])

    def test_one_set_gives_at_most_cap_candidates_and_tries_stay_bounded(self):
        sets, details = one_set("s1", [raw_card("s1", n) for n in range(1, 31)])
        net = Net(details=details, sets=sets, bad_images={f"https://assets.tcgdex.net/en/x/s1/{n}/high.webp" for n in range(1, 31)})
        with mock.patch.object(new_sets, "SET_FAILURES", 1000):  # only the candidate bounds apply
            run(net, known=set(), cap=4)
        self.assertEqual(len([r for r in net.requests if r.endswith("high.webp")]), 4)  # one set gives at most `cap` candidates
        cards, _ = new_sets.find_new_cards(set(), net.json, lambda s: None, cap=8, per_set_cap=4)
        self.assertEqual(len(cards), 4)

    def test_a_set_that_fails_too_often_is_skipped_for_the_night(self):
        sets, details = one_set("s1", [raw_card("s1", n) for n in range(1, 11)])
        net = Net(details=details, sets=sets, bad_images={f"https://assets.tcgdex.net/en/x/s1/{n}/high.webp" for n in range(1, 11)})
        log = io.StringIO()
        with mock.patch.object(new_sets, "SET_FAILURES", 3), contextlib.redirect_stdout(log):
            run(net, known=set())
        self.assertEqual(len([r for r in net.requests if r.endswith("high.webp")]), 3)
        self.assertEqual(log.getvalue().count("skip the rest of set s1"), 1)


class PreviousFile(unittest.TestCase):
    def test_an_unreadable_previous_file_is_rebuilt_from_nothing(self):
        good_card = {"id": "x-1", "name": "N", "set": "x", "setName": "S", "number": "1", "image": "https://assets.tcgdex.net/a"}
        header = {"v": 1, "dim": 512, "entries": 1, "cardIds": ["x-1"], "cards": [good_card]}
        broken = [
            b"garbage",
            raw_extra([1, 2, 3]),  # a header that is not a dict
            raw_extra(None),
            raw_extra({**header, "cardIds": [7], "cards": [{**good_card, "id": 7}]}, bytes(512)),  # card ids that are not strings
            raw_extra({**header, "cardIds": [["x"]], "cards": [{**good_card, "id": ["x"]}]}, bytes(512)),
            raw_extra({**header, "cards": ["text"]}, bytes(512)),
        ]
        for previous in broken:
            for tag in (TAG, None):
                log = io.StringIO()
                with contextlib.redirect_stdout(log):
                    _, out = run(previous=previous, tag=tag)
                self.assertIn("previous catalogExtra is unreadable: rebuilding", log.getvalue(), previous)
                self.assertEqual(out["added"], ["sv10-1", "sv10-2", "sv9-2"], previous)
                new_sets.read_extra(out["bytes"])  # and the new file is valid

    def test_a_readable_previous_file_does_not_log_unreadable(self):
        _, first = run(cap=1)
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            run(previous=first["bytes"], cap=1)
        self.assertNotIn("unreadable", log.getvalue())

    def test_the_total_rows_limit_is_enforced(self):
        _, first = run(cap=2)  # two rows
        with mock.patch.object(new_sets, "MAX_TOTAL", 3):
            net, near = run(previous=first["bytes"])
            self.assertEqual((near["added"], near["total"]), (["sv9-2"], 3))  # only one more fits
            net, full = run(previous=near["bytes"])
            self.assertEqual((full["added"], full["bytes"], full["total"]), ([], near["bytes"], 3))
            self.assertEqual(net.requests, [])  # a full file does not even ask for the set list


def prev_card(n, **over):
    card = {"id": f"s1-{n}", "name": f"Card {n}", "set": "s1", "setName": "Set", "number": str(n), "image": f"https://assets.tcgdex.net/en/x/s1/{n}/high.webp"}
    card.update(over)
    return card


def previous_file(cards):
    rows = np.zeros((len(cards), 512), dtype=np.int8)
    for i in range(len(cards)):
        rows[i, i] = 127
    return new_sets.write_extra(cards, rows, tag=TAG)


class PreviousRows(unittest.TestCase):
    """Rows carried over from the previous night are checked like fresh ones; the bad ones are dropped with one log line."""

    def tidy(self, bad, known=KNOWN):
        previous = previous_file([prev_card(1), bad])
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            _, out = run(Net(sets=[]), previous=previous, known=known)
        self.assertEqual([c["id"] for c in new_sets.read_extra(out["bytes"])[0]], ["s1-1"], bad)
        self.assertEqual((out["added"], out["total"], out["dropped"]), ([], 1, 1))
        self.assertIn("dropped 1 previous catalogExtra rows: ", log.getvalue())
        self.assertIn(bad["id"], log.getvalue())

    def test_a_picture_from_another_host(self):
        self.tidy(prev_card(2, image="https://evil.example/a.webp"))
        self.tidy(prev_card(2, image="http://assets.tcgdex.net/a.webp"))

    def test_both_known_picture_hosts_stay_allowed(self):
        previous = previous_file([prev_card(1), prev_card(2, image="https://images.pokemontcg.io/s1/2_hires.png")])
        _, out = run(Net(sets=[]), previous=previous)
        self.assertEqual((out["bytes"], out["dropped"]), (previous, 0))  # nothing dropped: the file is kept byte for byte

    def test_an_id_that_is_too_long_or_has_the_wrong_shape(self):
        self.tidy(prev_card(2, id="s1-" + "x" * 62))
        self.tidy(prev_card(2, id="other-2"))
        self.tidy(prev_card(2, id="A1-2", set="A1"))

    def test_text_over_the_limit_or_with_control_characters(self):
        self.tidy(prev_card(2, name="x" * 201))
        self.tidy(prev_card(2, name="Pika​chu"))
        self.tidy(prev_card(2, setName="Se\x07t"))

    def test_a_duplicate_id(self):
        previous = previous_file([prev_card(1), prev_card(1)])
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            _, out = run(Net(sets=[]), previous=previous)
        self.assertEqual([c["id"] for c in new_sets.read_extra(out["bytes"])[0]], ["s1-1"])
        self.assertIn("dropped 1 previous", log.getvalue())

    def test_an_id_that_is_now_in_the_catalog(self):
        self.tidy(prev_card(2), known=KNOWN | {"s1-2"})

    def test_the_log_line_names_the_count_and_at_most_five_ids(self):
        previous = previous_file([prev_card(n, image="https://evil.example/x") for n in range(1, 8)])
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            _, out = run(Net(sets=[]), previous=previous)
        line = [x for x in log.getvalue().splitlines() if x.startswith("dropped")][0]
        self.assertIn("dropped 7 previous catalogExtra rows", line)
        self.assertEqual(sum(f"s1-{n}" in line for n in range(1, 8)), 5)
        self.assertIsNone(out["bytes"])  # nothing is left


class Downloads(unittest.TestCase):
    def test_a_redirect_to_another_host_is_refused_and_a_same_host_one_followed(self):
        import urllib.request

        handler = new_sets._SameHostRedirects()
        req = urllib.request.Request("https://assets.tcgdex.net/a")
        with self.assertRaises(urllib.error.HTTPError):
            handler.redirect_request(req, None, 302, "Found", {}, "https://evil.example/b")
        with self.assertRaises(urllib.error.HTTPError):
            handler.redirect_request(req, None, 302, "Found", {}, "http://assets.tcgdex.net/b")  # not https
        with self.assertRaises(urllib.error.HTTPError):
            handler.redirect_request(req, None, 302, "Found", {}, "//evil.example/b")
        again = handler.redirect_request(req, None, 302, "Found", {}, "/other/b")
        self.assertEqual(again.full_url, "https://assets.tcgdex.net/other/b")

    def test_the_downloader_uses_that_handler(self):
        self.assertTrue(any(isinstance(h, new_sets._SameHostRedirects) for h in new_sets._OPENER.handlers))


class TextAndDuplicates(unittest.TestCase):
    def test_control_and_format_characters_refuse_the_card_and_are_counted(self):
        cards = [raw_card("s1", 1, "Pika​chu"), raw_card("s1", 2, "Ra\x07ichu"), raw_card("s1", 3, "Flabébé"), raw_card("s1", 4, "Nidoran♀"), raw_card("s1", 5, "δ Species"),
                 raw_card("s1", 6, "Left‮Right")]
        sets, details = one_set("s1", cards)
        ids, log = quiet_find(Net(details=details, sets=sets))
        self.assertEqual(ids, ["s1-3", "s1-4", "s1-5"])
        self.assertIn("refused 3 cards with control or format characters", log)

    def test_a_duplicate_number_and_name_on_the_tcgdex_side_gets_no_fallback_picture(self):
        cards = [raw_card("sv9", 3, "Twin", image=False), raw_card("sv9", "03", "Twin", image=False, card_id="sv9-03"), raw_card("sv9", 4, "Solo", image=False)]
        sets, details = one_set("sv9", cards)
        ptcg = {"sv9": [ptcg_card("sv9", "3", "Twin"), ptcg_card("sv9", "4", "Solo")]}
        ids, _ = quiet_find(Net(details=details, sets=sets, ptcg=ptcg))
        self.assertEqual(ids, ["sv9-4"])  # the two "Twin" cards are ambiguous, "Solo" is not


class SetUrls(unittest.TestCase):
    def test_the_set_id_is_quoted_in_the_detail_url(self):
        codes = ["a/b", "a?b", "a#b", "a/../b", "a b"]
        listing = [{"id": c, "name": "S", "cardCount": {"total": 9}} for c in codes]
        seen = []

        def fetch(url):
            if url.endswith("/sets"):
                return listing
            seen.append(url)
            return {"name": "S", "cards": []}

        new_sets.find_new_cards(set(), fetch, lambda s: None)
        self.assertEqual(sorted(seen), sorted(f"{new_sets.TCGDEX}/sets/{q}" for q in ("a%2Fb", "a%3Fb", "a%23b", "a%2F..%2Fb", "a%20b")))

    def test_a_dot_or_dot_dot_set_id_is_not_requested(self):
        seen = []

        def fetch(url):
            if url.endswith("/sets"):
                return [{"id": ".", "name": "S", "cardCount": {"total": 9}}, {"id": "..", "name": "S", "cardCount": {"total": 9}}]
            seen.append(url)
            return {"name": "S", "cards": []}

        new_sets.find_new_cards(set(), fetch, lambda s: None)
        self.assertEqual(seen, [])


class Budgets(unittest.TestCase):
    def test_the_embed_loop_has_a_wall_clock_budget_and_keeps_what_it_added(self):
        sets, details = one_set("s1", [raw_card("s1", n) for n in range(1, 6)])
        ticks = iter(range(0, 100000, 1000))  # every look at the clock is 1,000 s later
        log = io.StringIO()
        self.assertEqual(new_sets.EMBED_SECONDS, 1800)
        net = Net(details=details, sets=sets)
        with contextlib.redirect_stdout(log):
            out = new_sets.build(set(), None, embedder, fetch=net.json, fetch_image=net.image, sleep=lambda s: None, tag=TAG, clock=lambda: next(ticks))
        self.assertEqual(out["added"], ["s1-1"])  # started at 0 s, 1,000 s is within budget, 2,000 s is not
        self.assertEqual(log.getvalue().count("embed time budget"), 1)
        new_sets.read_extra(out["bytes"])

    def test_the_fallback_has_a_wall_clock_budget(self):
        count = 5
        sets = [{"id": f"x{n}", "name": "S", "cardCount": {"total": 1}} for n in range(count)]
        details = {f"x{n}": {"name": "S", "cards": [raw_card(f"x{n}", 1, "One", image=False)]} for n in range(count)}
        ptcg = {f"x{n}": [ptcg_card(f"x{n}", "1", "One")] for n in range(count)}
        net = Net(details=details, sets=sets, ptcg=ptcg)
        ticks = iter(range(0, 100000, 400))  # every look at the clock is 400 s later
        self.assertEqual(new_sets.FALLBACK_SECONDS, 600)
        ids, log = quiet_find(net, clock=lambda: next(ticks))
        self.assertEqual(len(ids), 2)  # started at 0 s, 400 s is within budget, 800 s is not
        self.assertEqual(len(asked_files(net)), 2)
        self.assertEqual(log.count("fallback time budget"), 1)

    def test_a_card_whose_picture_fails_does_not_use_a_slot(self):
        _, out = run(Net(bad_images={"https://assets.tcgdex.net/en/sv/sv10/1/high.webp"}), cap=1)
        # sv10-1 failed and did not use the slot. One set gives at most `cap` (1) candidates, so the next set's card took it.
        self.assertEqual(out["added"], ["sv9-2"])

    def test_tries_are_bounded_at_twice_the_cap(self):
        everything = {f"https://assets.tcgdex.net/en/sv/{s}/{n}/high.webp" for s in ("sv9", "sv10") for n in (1, 2)}
        net = Net(bad_images=everything)
        _, out = run(net, cap=1)
        self.assertEqual(out["added"], [])
        self.assertEqual(len([r for r in net.requests if r.endswith("high.webp")]), 2)
        net = Net(bad_images=everything)
        run(net, cap=2)
        self.assertEqual(len([r for r in net.requests if r.endswith("high.webp")]), 3)  # only three candidates exist


class Build(unittest.TestCase):
    def test_writes_rows_in_the_extension_layout(self):
        _, out = run()
        self.assertEqual(out["added"], ["sv10-1", "sv10-2", "sv9-2"])
        self.assertEqual((out["sets"], out["total"]), (["sv10", "sv9"], 3))
        cards, rows = new_sets.read_extra(out["bytes"])
        self.assertEqual([c["id"] for c in cards], out["added"])
        self.assertEqual(rows.shape, (3, 512))
        self.assertEqual(int(rows[0].max()), 127)
        head = header_of(out["bytes"])
        self.assertEqual(head["cardIds"], out["added"])
        self.assertEqual(head["entries"], 3)

    def test_the_header_carries_the_model_and_crop(self):
        _, out = run()
        head = header_of(out["bytes"])
        self.assertEqual((head["model"], head["crop"]), (TAG["model"], TAG["crop"]))
        self.assertEqual(new_sets.extra_tag(out["bytes"]), (TAG["model"], TAG["crop"]))
        _, plain = run(tag=None)
        self.assertNotIn("model", header_of(plain["bytes"]))
        self.assertEqual(new_sets.extra_tag(plain["bytes"]), (None, None))

    def test_the_tagged_fixture_the_extension_test_reads_stays_byte_identical(self):
        """tests/shared-data-pack.test.js feeds this file to the extension's parseCatalogExtra."""
        import os

        cards = [
            {"id": "sv9-2", "name": "Raichu", "set": "sv9", "setName": "Journey Together", "number": "2", "image": "https://assets.tcgdex.net/en/sv/sv9/2/high.webp"},
            {"id": "sv9-3", "name": "No Picture", "set": "sv9", "setName": "Journey Together", "number": "3", "image": "https://images.pokemontcg.io/sv9/3_hires.png"},
        ]
        vecs = np.random.RandomState(11).randint(-127, 128, size=(2, 512)).astype(np.int8)
        data = new_sets.write_extra(cards, vecs, tag=TAG)
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "catalog-extra-tagged-v1.bin")
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(data)
        with open(path, "rb") as f:
            self.assertEqual(f.read(), data)

    def test_a_previous_file_of_another_model_or_untagged_is_rebuilt_from_scratch(self):
        _, old = run(cap=1, tag=OTHER_TAG)
        _, untagged = run(cap=1, tag=None)
        for previous in (old["bytes"], untagged["bytes"]):
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                _, out = run(previous=previous)
            self.assertIn("catalogExtra was made with another model: rebuilding", log.getvalue())
            self.assertEqual(out["added"], ["sv10-1", "sv10-2", "sv9-2"])  # the earlier file's card is embedded again, not copied
            self.assertEqual(new_sets.extra_tag(out["bytes"]), (TAG["model"], TAG["crop"]))
            self.assertEqual(out["total"], 3)

    def test_a_previous_file_of_the_same_model_is_extended(self):
        _, first = run(cap=1)
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            _, second = run(previous=first["bytes"], cap=1)
        self.assertNotIn("another model", log.getvalue())
        self.assertEqual([c["id"] for c in new_sets.read_extra(second["bytes"])[0]], ["sv10-1", "sv10-2"])

    def test_is_cumulative_and_does_not_repeat_earlier_cards(self):
        _, first = run(cap=1)
        self.assertEqual(first["added"], ["sv10-1"])
        _, second = run(previous=first["bytes"], cap=1)
        self.assertEqual(second["added"], ["sv10-2"])
        cards, rows = new_sets.read_extra(second["bytes"])
        self.assertEqual([c["id"] for c in cards], ["sv10-1", "sv10-2"])
        self.assertTrue(np.array_equal(rows[0], new_sets.read_extra(first["bytes"])[1][0]))  # old rows are copied, not re-embedded
        self.assertEqual(new_sets.extra_tag(second["bytes"]), (TAG["model"], TAG["crop"]))

    def test_a_picture_that_fails_is_skipped_and_left_for_tomorrow(self):
        _, out = run(Net(bad_images={"https://assets.tcgdex.net/en/sv/sv10/1/high.webp"}))
        self.assertEqual(out["added"], ["sv10-2", "sv9-2"])
        _, again = run(previous=out["bytes"])
        self.assertEqual(again["added"], ["sv10-1"])

    def test_nothing_new_keeps_the_previous_file_byte_for_byte(self):
        _, first = run()
        _, again = run(previous=first["bytes"])
        self.assertEqual((again["added"], again["bytes"]), ([], first["bytes"]))
        _, none = run(Net(sets=[SETS[0]]))
        self.assertIsNone(none["bytes"])
        net, quiet = run(Net(sets=[SETS[0]]), previous=first["bytes"])
        self.assertEqual((quiet["bytes"], net.requests), (first["bytes"], [f"{new_sets.TCGDEX}/sets"]))  # a quiet night keeps the tagged file as is

    def test_unusable_embeddings_are_dropped(self):
        def bad(data, trusted=False):
            return np.full(512, np.nan, dtype=np.float32)

        net = Net()
        out = new_sets.build(KNOWN, None, bad, fetch=net.json, fetch_image=net.image, sleep=lambda s: None)
        self.assertEqual(out["added"], [])

    def test_read_extra_rejects_damage(self):
        _, out = run()
        for broken in (out["bytes"][:-1], b"\x01", out["bytes"] + b"x"):
            with self.assertRaises(ValueError):
                new_sets.read_extra(broken)

    def test_requests_are_paced(self):
        pauses = []
        net = Net()
        new_sets.build(KNOWN, None, embedder, fetch=net.json, fetch_image=net.image, sleep=pauses.append)
        self.assertTrue(pauses and all(p == new_sets.REQUEST_DELAY for p in pauses))
        self.assertGreaterEqual(len(pauses), 2 + 3)  # two sets opened, three pictures


if __name__ == "__main__":
    unittest.main()
