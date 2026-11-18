"""Unit tests for the Amendment 1 counting and decision in scripts/newsets_eval.py. Tiny arrays, no network, no model."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import newsets_eval as N  # noqa: E402


def sgc(s, g, c):
    return {"S": s, "G": g, "C": c}


def test_sgc_counting_on_a_hand_made_example():
    base_ok = [1, 1, 0, 0, 1, 0]
    arm_ok = [0, 1, 1, 0, 0, 0]
    base_sw = [0, 0, 1, 0, 0, 0]
    arm_sw = [1, 0, 0, 1, 1, 0]
    # boxes 0 and 4 were right and are now wrong (S=2); box 2 was wrong and is now right (G=1);
    # sure-wrong: 3 under the arm minus 1 at baseline (C=2)
    assert N.count_sgc(base_ok, arm_ok, base_sw, arm_sw) == {"S": 2, "G": 1, "C": 2}


def test_sgc_can_be_negative_for_c():
    assert N.count_sgc([1], [1], [1], [0])["C"] == -1


def test_all_conditions_met_chooses_e():
    d = N.decide_real(sgc(1, 1, 0), sgc(2, 3, 2), 1000)
    assert d["decision"] == "E" and d["all_pass"] and not d["t_arm_warning"]


def test_condition_one_flips_the_decision():
    d = N.decide_real(sgc(0, 0, 0), sgc(0, 0, 3), 1000)  # C(E)=3 > C(T)+2=2
    assert not d["conditions"]["c1"]["pass"] and d["decision"] == "T"
    assert N.decide_real(sgc(0, 0, 0), sgc(0, 0, 2), 1000)["decision"] == "E"  # equal to the limit still passes


def test_condition_two_flips_the_decision():
    d = N.decide_real(sgc(1, 1, 0), sgc(5, 0, 0), 1000)  # S-G = 5 <= 0+5 passes
    assert d["conditions"]["c2"]["pass"] and d["decision"] == "E"
    d = N.decide_real(sgc(1, 1, 0), sgc(5, 0, 0), 1000)
    d = N.decide_real(sgc(0, 0, 0), sgc(5, 0, 0), 1000)
    assert d["conditions"]["c2"]["pass"]
    d = N.decide_real(sgc(0, 1, 0), sgc(5, 0, 0), 1000)  # 5 > -1 + 5 = 4
    assert not d["conditions"]["c2"]["pass"] and d["decision"] == "T"


def test_condition_three_flips_the_decision():
    d = N.decide_real(sgc(0, 0, 0), sgc(5, 5, 0), 1000)  # limit 5.0 passes
    assert d["conditions"]["c3"]["pass"] and d["decision"] == "E"
    d = N.decide_real(sgc(0, 0, 0), sgc(6, 6, 0), 1000)  # 6 > 5.0
    assert not d["conditions"]["c3"]["pass"] and d["decision"] == "T"


def test_t_arm_warning_blocks_any_recommendation():
    d = N.decide_real(sgc(0, 0, 3), sgc(0, 0, 3), 1000)  # C(T) > 2, E conditions otherwise pass
    assert d["t_arm_warning"] and d["decision"] == "owner_decision"
    d = N.decide_real(sgc(6, 0, 0), sgc(0, 0, 0), 1000)  # S(T)=6 > 0.5% of 1000
    assert d["t_arm_warning"] and d["decision"] == "owner_decision"
    d = N.decide_real(sgc(5, 0, 2), sgc(0, 0, 0), 1000)  # exactly at the limits: no warning
    assert not d["t_arm_warning"]
