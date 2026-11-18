"""Unit tests for scripts/japanese_study.py (preregistered Japanese-card study). No network, no model, tiny arrays."""
import json
import os
import sys
from fractions import Fraction

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import japanese_study as S  # noqa: E402

HEADER = "box_id\tname\tconfidence\ttag\tnote\n"


def write_labels(label_dir, rows_a, rows_b, agreed):
    for letter, rows in (("A", rows_a), ("B", rows_b)):
        d = os.path.join(label_dir, letter)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "labels.tsv"), "w", encoding="utf-8") as f:
            f.write(HEADER + "".join("\t".join(str(x) for x in r) + "\n" for r in rows))
    with open(os.path.join(label_dir, "agreed.json"), "w", encoding="utf-8") as f:
        json.dump(agreed, f)


def agreed_row(box_id, name="Pikachu", h=250.0):
    return {"box_id": box_id, "name": name, "split": "final", "site": "ebay", "listing": f"ebay:{box_id}", "score_h": h,
            "review": False}


# ---- class assignment -------------------------------------------------------------------------------------------

def test_j_needs_both_foreign_card_and_both_japan(tmp_path):
    rows_a = [
        (1, "Pikachu", "HIGH", "foreign_card", "Japanese Pikachu"),   # J
        (2, "Pikachu", "HIGH", "foreign_card", "Japanese Pikachu"),   # only B says japan -> other foreign
        (3, "Pikachu", "HIGH", "foreign_card", "Korean card"),        # neither says japan -> other foreign
        (4, "", "HIGH", "card_back", "japan pokemon card back"),      # card back with Japan note -> left out
        (5, "Pikachu", "HIGH", "english_card", "in sleeve"),          # E
        (6, "Pikachu", "HIGH", "foreign_card", "JAPAN"),              # J (case-insensitive)
    ]
    rows_b = [
        (1, "Pikachu", "HIGH", "foreign_card", "japan"),
        (2, "Pikachu", "HIGH", "foreign_card", "foreign"),
        (3, "Pikachu", "HIGH", "foreign_card", "chinese"),
        (4, "", "HIGH", "card_back", "japan card back"),
        (5, "Pikachu", "HIGH", "english_card", ""),
        (6, "Pikachu", "HIGH", "foreign_card", "Japanese"),
    ]
    write_labels(str(tmp_path), rows_a, rows_b, [agreed_row(1), agreed_row(5), agreed_row(6)])
    cls = {b: v["cls"] for b, v in S.classes_for_set(str(tmp_path)).items()}
    assert cls[1] == "J"
    assert cls[2] == "other_foreign"
    assert cls[3] == "other_foreign"
    assert cls[4] == "left_out"
    assert cls[5] == "E"
    assert cls[6] == "J"


def test_agreed_box_with_one_foreign_tag_is_other_foreign_not_e(tmp_path):
    rows_a = [(1, "Pikachu", "HIGH", "foreign_card", "korean"), (2, "Pikachu", "HIGH", "english_card", "")]
    rows_b = [(1, "Pikachu", "HIGH", "english_card", ""), (2, "Pikachu", "HIGH", "english_card", "")]
    write_labels(str(tmp_path), rows_a, rows_b, [agreed_row(1), agreed_row(2)])
    cls = {b: v["cls"] for b, v in S.classes_for_set(str(tmp_path)).items()}
    assert cls[1] == "other_foreign"
    assert cls[2] == "E"


def test_english_needs_agreed_and_unagreed_is_left_out(tmp_path):
    rows_a = [(1, "Pikachu", "HIGH", "english_card", ""), (2, "", "HIGH", "unreadable", "")]
    rows_b = [(1, "Raichu", "HIGH", "english_card", ""), (2, "", "HIGH", "unreadable", "")]
    write_labels(str(tmp_path), rows_a, rows_b, [])
    cls = {b: v["cls"] for b, v in S.classes_for_set(str(tmp_path)).items()}
    assert cls == {1: "left_out", 2: "left_out"}


def test_height_bands():
    assert [S.band_of(x) for x in (10, 149.9, 150, 199.9, 200, 299.9, 300, 800)] == [
        "<150", "<150", "150-199", "150-199", "200-299", "200-299", "300+", "300+"]


# ---- quantisation -----------------------------------------------------------------------------------------------

def test_quantise_round_trip_keeps_cosine():
    rng = np.random.default_rng(0)
    v = rng.standard_normal((200, 512)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    q, scale = S.quantize_rows(v)
    assert q.dtype == np.int8 and scale.dtype == np.float32 and q.shape == v.shape and scale.shape == (200,)
    assert np.abs(q).max() <= 127
    back = S.dequantize_rows(q, scale)
    assert back.dtype == np.float32
    assert np.allclose(np.linalg.norm(back, axis=1), 1.0, atol=1e-5)
    cos = np.einsum("ij,ij->i", v, back)
    assert (1 - cos).max() < 0.001


def test_quantise_scale_is_max_abs_over_127():
    v = np.zeros((1, 512), np.float32)
    v[0, 3], v[0, 9] = -0.5, 0.25
    q, scale = S.quantize_rows(v)
    assert scale[0] == pytest.approx(0.5 / 127)
    assert q[0, 3] == -127


# ---- Wilson -----------------------------------------------------------------------------------------------------

def test_wilson_known_values():
    lo, hi = S.wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    lo, hi = S.wilson(0, 20)
    assert lo == pytest.approx(0.0, abs=1e-9) and hi == pytest.approx(3.8416 / 23.8416, abs=1e-3)
    assert S.wilson(0, 0) == (0.0, 1.0)


# ---- fit picker -------------------------------------------------------------------------------------------------

def one_data(d, h, cls):
    return {"d": np.asarray(d, float), "h": np.asarray(h, float), "cls": np.asarray(cls)}


def cell_for(cells, rows, h_min, m):
    (c,) = [x for x in cells if x["rows"] == rows and x["h_min"] == h_min and abs(x["m"] - m) < 1e-9]
    return c


def test_grid_has_21_m_values_and_four_choices_each():
    data = {r: one_data([0.0], [250], ["E"]) for r in ("T", "E")}
    cells = S.evaluate_grid(data)
    assert S.M_GRID[0] == 0.0 and S.M_GRID[-1] == 0.1 and len(S.M_GRID) == 21
    assert len(cells) == 2 * 2 * 21


def test_constraint_overall_blocks_and_larger_m_wins_ties():
    # 100 E at 250 px (one band with >= 50), 3 of them have d = 0.06; 10 J all d = 0.08.
    d = [0.06] * 3 + [-0.1] * 97 + [0.08] * 10
    h = [250] * 110
    cls = ["E"] * 100 + ["J"] * 10
    cells = S.evaluate_grid({"T": one_data(d, h, cls), "E": one_data(d, h, cls)})
    assert not cell_for(cells, "T", 150, 0.06)["ok"]        # 3% overall > 2%
    assert cell_for(cells, "T", 150, 0.065)["ok"]
    best = S.pick(cells)
    assert (best["rows"], best["h_min"], best["m"]) == ("T", 150, 0.08)
    assert best["j_flag"] == 10 and best["j_n"] == 10


def test_constraint_per_band_applies_at_50_and_above():
    # band 200-299: 100 E, none flagged. band 150-199: 60 E, 3 flagged (5% > 3%); overall 3/160 = 1.9% <= 2%.
    d = [-0.1] * 100 + [0.05] * 3 + [-0.1] * 57 + [0.09] * 4
    h = [250] * 100 + [170] * 60 + [250] * 4
    cls = ["E"] * 160 + ["J"] * 4
    cells = S.evaluate_grid({"T": one_data(d, h, cls), "E": one_data(d, h, cls)})
    assert not cell_for(cells, "T", 150, 0.05)["ok"]       # band 150-199 fails
    assert cell_for(cells, "T", 200, 0.05)["ok"]           # that band is below H_MIN, not counted
    assert cell_for(cells, "T", 150, 0.055)["ok"]


def test_band_with_under_50_english_is_ignored():
    # 40 E in band 150-199 with 5 flagged (12.5%); 300 E elsewhere unflagged; overall 5/340 = 1.5%.
    d = [0.05] * 5 + [-0.1] * 35 + [-0.1] * 300 + [0.09] * 4
    h = [170] * 40 + [250] * 300 + [250] * 4
    cls = ["E"] * 340 + ["J"] * 4
    cells = S.evaluate_grid({"T": one_data(d, h, cls), "E": one_data(d, h, cls)})
    c = cell_for(cells, "T", 150, 0.05)
    assert c["ok"]
    assert c["bands"]["150-199"]["e_n"] == 40


def cell(rows, h_min, m, j_flag, j_n, ok=True):
    return {"rows": rows, "h_min": h_min, "m": m, "j_flag": j_flag, "j_n": j_n, "ok": ok}


def test_pick_highest_recall_then_h150_then_t_then_larger_m():
    assert S.pick([cell("T", 150, 0.02, 5, 10), cell("E", 150, 0.02, 6, 10)])["rows"] == "E"
    # equal recall as fractions (3/6 == 1/2): h_min 150 first
    best = S.pick([cell("T", 200, 0.05, 1, 2), cell("E", 150, 0.01, 3, 6)])
    assert (best["rows"], best["h_min"]) == ("E", 150)
    # then T rows
    best = S.pick([cell("E", 150, 0.05, 5, 10), cell("T", 150, 0.01, 5, 10)])
    assert best["rows"] == "T"
    # then larger m
    best = S.pick([cell("T", 150, 0.01, 5, 10), cell("T", 150, 0.04, 5, 10), cell("T", 150, 0.03, 5, 10)])
    assert best["m"] == 0.04
    # cells that fail the constraint never win
    assert S.pick([cell("T", 150, 0.01, 9, 10, ok=False), cell("T", 150, 0.02, 4, 10)])["m"] == 0.02


def test_pick_recall_floor_is_40_percent():
    assert S.pick([cell("T", 150, 0.01, 39, 100)]) is None
    assert S.pick([cell("T", 150, 0.01, 40, 100)]) is not None
    assert S.pick([]) is None
    assert S.RECALL_FLOOR == Fraction(2, 5)


# ---- fit / final commands: files and refusals --------------------------------------------------------------------

def write_fit_inputs(out_dir, with_pick=True):
    n_e, n_j = 100, 10
    boxes, d = [], []
    for i in range(n_e):
        boxes.append({"row": i, "set": "obscure", "box_id": i, "cls": "E", "h": 250.0, "band": "200-299", "site": "ebay",
                      "listing": f"l{i}", "name_a": "x", "name_b": "x"})
        d.append(-0.1)
    for j in range(n_j):
        boxes.append({"row": n_e + j, "set": "fresh-20261111", "box_id": 1000 + j, "cls": "J" if with_pick else "left_out",
                      "h": 250.0, "band": "200-299", "site": "ebay", "listing": f"j{j}", "name_a": "y", "name_b": "y"})
        d.append(0.08)
    # a box from the final-test set must not enter the fit
    for k in range(10):
        boxes.append({"row": n_e + n_j + k, "set": "check-20261113", "box_id": k, "cls": "E", "h": 250.0, "band": "200-299",
                      "site": "ebay", "listing": "f", "name_a": "z", "name_b": "z"})
        d.append(0.5)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "classes.json"), "w", encoding="utf-8") as f:
        json.dump({"boxes": boxes}, f)
    for rows in ("T", "E"):
        np.savez(os.path.join(out_dir, f"features_{rows}.npz"), d=np.asarray(d))


def test_fit_writes_fit_and_freeze_and_uses_fit_pool_only(tmp_path):
    write_fit_inputs(str(tmp_path))
    S.cmd_fit(str(tmp_path))
    freeze = json.load(open(tmp_path / "freeze.json", encoding="utf-8"))
    assert freeze["ship"] is True
    assert (freeze["rows"], freeze["h_min"], freeze["m"]) == ("T", 150, 0.08)
    assert freeze["fit"]["j_flag"] == 10 and freeze["fit"]["e_flag"] == 0 and freeze["fit"]["e_n"] == 100
    assert "timestamp" in freeze
    assert len(json.load(open(tmp_path / "fit.json", encoding="utf-8"))["cells"]) == 84


def test_fit_refuses_when_freeze_exists(tmp_path):
    write_fit_inputs(str(tmp_path))
    (tmp_path / "freeze.json").write_text("{}", encoding="utf-8")
    with pytest.raises(S.StudyRefusal):
        S.cmd_fit(str(tmp_path))
    assert (tmp_path / "freeze.json").read_text(encoding="utf-8") == "{}"
    assert not (tmp_path / "fit.json").exists()


def test_fit_with_no_qualifying_choice_writes_ship_false(tmp_path):
    write_fit_inputs(str(tmp_path), with_pick=False)  # no J boxes: recall is 0/0, below the floor
    S.cmd_fit(str(tmp_path))
    freeze = json.load(open(tmp_path / "freeze.json", encoding="utf-8"))
    assert freeze["ship"] is False


def test_final_refuses_without_freeze(tmp_path):
    with pytest.raises(S.StudyRefusal):
        S.cmd_final(str(tmp_path))


def test_final_refuses_when_freeze_has_no_pick(tmp_path):
    (tmp_path / "freeze.json").write_text(json.dumps({"ship": False}), encoding="utf-8")
    with pytest.raises(S.StudyRefusal):
        S.cmd_final(str(tmp_path))


def test_final_refuses_a_second_run(tmp_path):
    (tmp_path / "freeze.json").write_text(json.dumps({"ship": True, "rows": "T", "h_min": 150, "m": 0.05}), encoding="utf-8")
    (tmp_path / "final.json").write_text("{}", encoding="utf-8")
    with pytest.raises(S.StudyRefusal):
        S.cmd_final(str(tmp_path))
    assert (tmp_path / "final.json").read_text(encoding="utf-8") == "{}"


def test_gates():
    g = S.gates(e_flag=2, e_n=200, j_flag=30, j_n=40)
    assert g["G1"]["pass"] and g["G2"]["pass"]
    g = S.gates(e_flag=6, e_n=200, j_flag=30, j_n=40)       # 3.0% passes the point test, Wilson upper > 5%
    assert not g["G1"]["pass"]
    g = S.gates(e_flag=0, e_n=200, j_flag=23, j_n=40)       # 57.5% recall
    assert g["G1"]["pass"] and not g["G2"]["pass"]
    assert not S.gates(e_flag=0, e_n=0, j_flag=0, j_n=0)["G2"]["pass"]


def test_latin_script_name_check():
    assert S.is_latin_name("Pikachu ex")
    assert S.is_latin_name("Farfetch'd")
    assert not S.is_latin_name("ピカチュウ")
    assert not S.is_latin_name("")
