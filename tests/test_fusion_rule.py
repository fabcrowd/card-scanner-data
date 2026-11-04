import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from fusion_rule import classify  # noqa: E402

IDS = ["a", "b", "c"]


def test_empty():
    r = classify([], [])
    assert r.label == "abstain" and r.reason == "no_hits" and r.id is None


def test_low_sim_abstains_just_below_accept():
    assert classify([0.7199, 0.5], IDS).reason == "low_sim"


def test_at_accept_threshold_with_clear_gap():
    r = classify([0.72, 0.60], IDS)
    assert r.accepted and r.reason == "clear_gap" and r.id == "a" and not r.high


def test_tie_gap_just_below_clear_gap_abstains_without_names():
    r = classify([0.80, 0.7801], IDS)
    assert r.reason == "ambiguous" and not r.accepted
    assert classify([0.80, 0.77], IDS).reason == "clear_gap"


def test_high_sim_overrides_tie():
    r = classify([0.86, 0.855], IDS)
    assert r.reason == "high_sim" and r.accepted and not r.high


def test_name_match_needs_sim_075():
    n = ["X", "X", "Y"]
    assert classify([0.76, 0.755], IDS, n).reason == "name_match"
    # below 0.75 name_match is skipped; the same-name pair then wins by name_vote (count 2)
    assert classify([0.74, 0.739], IDS, n).reason == "name_vote"


def test_name_vote_beats_lookalike():
    r = classify([0.799, 0.798, 0.791], IDS, ["E", "S", "S"])
    assert r.reason == "name_vote" and r.id == "b"


def test_name_vote_tie_between_names_abstains():
    assert classify([0.799, 0.798], IDS, ["A", "B"]).reason == "ambiguous"


def test_missing_names_disable_name_paths():
    assert classify([0.76, 0.755], IDS, [None, None, None]).reason == "ambiguous"


def test_high_tier_boundaries():
    assert classify([0.82, 0.79], IDS).high
    assert not classify([0.82, 0.796], IDS).high      # lead 0.024
    assert not classify([0.819, 0.7], IDS).high
    assert classify([0.819, 0.7], IDS).confidence == 0.79


def test_height_caps():
    base = ([0.9, 0.7], IDS)
    assert classify(*base, height=0).label == "sure"
    assert classify(*base, height=300).confidence == 0.9
    r = classify(*base, height=250)
    assert r.label == "sure" and r.confidence == 0.79 and r.high
    r = classify(*base, height=199)
    assert r.label == "guess" and r.confidence == 0.49 and r.accepted
