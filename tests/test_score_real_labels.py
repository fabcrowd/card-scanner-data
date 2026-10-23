import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import score_real_labels as s  # noqa: E402


def test_normalise_possessive_punct_case():
    assert s.normalise("Professor Rowan's  Letter") == "professor rowan letter"
    assert s.normalise("Mime Jr.") == "mime jr"
    assert s.normalise("Ho-Oh") == "ho oh"


def test_base_vs_strict_suffixes():
    assert s.match("Mewtwo V", "Mewtwo", True)
    assert not s.match("Mewtwo V", "Mewtwo", False)
    assert s.match("Eevee ex", "Eevee", True)
    assert s.match("Beldum δ", "Beldum", True)
    assert s.match("Professor Rowan", "Professor Rowan", False)
    assert not s.match("Mew", "Mewtwo", True)


def test_parse_label_counts():
    labels = s.parse_labels()
    assert len(labels) == 301
    kinds = [l["kind"] for l in labels.values()]
    assert (kinds.count("card"), kinds.count("noncard"), kinds.count("pile")) == (294, 4, 3)
    conf = [l["conf"] for l in labels.values() if l["kind"] == "card"]
    assert (conf.count(2), conf.count(1), conf.count(0)) == (68, 203, 23)
    assert labels[(1, 1)]["name"] == "Onix" and labels[(1, 1)]["tilde"]


def _box(photo, names, accepted=False, h=80, cos=0.9):
    return {"photo": photo, "height": h, "accepted": accepted,
            "top5": [{"id": f"x{i}", "name": n, "cos": cos - i * 0.01} for i, n in enumerate(names)]}


def test_hand_checked_scoring():
    boxes = s.number_boxes([_box("a.jpg", ["Eevee ex", "Mew"], accepted=True),
                            _box("a.jpg", ["Pikachu", "Mewtwo V"]),
                            _box("a.jpg", ["Muk", "Zapdos"], accepted=True)])
    labels = {(1, 1): {"kind": "card", "name": "Eevee", "conf": 2},
              (1, 2): {"kind": "card", "name": "Mewtwo", "conf": 1},
              (1, 3): {"kind": "pile", "name": "-", "conf": 2}}
    r = s.score_model(boxes, labels)
    ar = r["ALL_READABLE"]
    assert ar["overall"]["n"] == 2
    assert ar["overall"]["top1_base"] == 0.5
    assert ar["overall"]["top5_base"] == 1.0
    assert ar["overall"]["top1_strict"] == 0.0
    assert ar["fusion_accepted"]["accepted"] == 1 and ar["fusion_accepted"]["precision"] == 1.0
    assert r["CLEAR"]["overall"]["n"] == 1
    assert r["false_accepts_noncard_pile"] == {"boxes": 1, "accepted": 1}
