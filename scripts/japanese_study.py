"""Analysis for the preregistered Japanese-card study (data/reports/japanese-20261117/PREREGISTRATION.md, Amendment 1).

Sub-commands: classes, sanity, features --rows T|E, fit, final, synth --rows T|E [--n N], report.
`fit` and `final` each write their result once and refuse to run again. Nothing here tunes against the sealed set.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from datetime import datetime, timezone
from fractions import Fraction

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from pipeline.catalog import Catalog  # noqa: E402
from score_real_labels import base_name  # noqa: E402

REPORTS = os.path.join(ROOT, "data", "reports")
OUT = os.path.join(REPORTS, "japanese-20261117")
JA_DIR = os.path.join(ROOT, "data", "ja-catalog")
FEAT_DIR = r"C:\repos\projects\auto card lister\.tmp\detect-qa\diag-20261113"
MODEL_E = os.path.join(ROOT, "train", "out", "fastE_all", "epoch16-ema-int8.onnx")
MODEL_E_SHA_PREFIX = "0dba1302"

SETS = ["obscure", "fresh-20261111", "check-20261112", "check-20261112c", "check-20261113"]
FIT_POOL = ["obscure", "fresh-20261111", "check-20261112", "check-20261112c"]
FINAL_SET = "check-20261113"
BANDS = ["<150", "150-199", "200-299", "300+"]
FIT_BANDS = BANDS[1:]
H_MINS = (150, 200)
ROWS_CHOICES = ("T", "E")  # order matters only for output; the tie-break prefers T
M_GRID = [round(i * 0.005, 3) for i in range(21)]
M_EPS = 1e-9
MAX_FALSE_OVERALL = Fraction(2, 100)
MAX_FALSE_BAND = Fraction(3, 100)
BAND_MIN_ENGLISH = 50
RECALL_FLOOR = Fraction(2, 5)
G1_MAX_RATE = Fraction(3, 100)
G1_MAX_WILSON_UPPER = 0.05
G2_MIN_RECALL = Fraction(3, 5)
MIN_NAMED_BOXES = 20
SANITY_TOL = 0.002
FOREIGN_TAG = "foreign_card"


class StudyRefusal(Exception):
    """A rule of the study forbids this step right now."""


# ---- small helpers ----------------------------------------------------------------------------------------------

def band_of(h):
    if h < 150:
        return "<150"
    if h < 200:
        return "150-199"
    if h < 300:
        return "200-299"
    return "300+"


def wilson(k, n, z=1.96):
    """95% Wilson score interval (lo, hi) for k of n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(max(0.0, c - h)), float(min(1.0, c + h))


def is_latin_name(s):
    letters = [ch for ch in (s or "") if ch.isalpha()]
    return bool(letters) and all(unicodedata.name(ch, "").startswith("LATIN") for ch in letters)


def norm_card_name(name):
    """heldout_report's name normalisation (score_real_labels.base_name) after dropping '(Mirror Holo)'-style suffixes."""
    return base_name(re.sub(r"\([^)]*\)", " ", name or ""))


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def lower_priority():
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL
    except Exception:  # not Windows: priority is left alone
        pass


def run_real_photos():
    import run_real_photos as R
    return R


def english_catalog():
    """Catalog without the digital-only ids, and the ids of its rows."""
    R = run_real_photos()
    cat = Catalog.load(os.path.join(ROOT, "assets", "catalog.bin.gz"), os.path.join(ROOT, "assets", "id_map.json"))
    keep = np.array([not R.DIGITAL_RE.match(c) for c in cat.ids])
    return Catalog(cat.vectors[keep], [c for c, k in zip(cat.ids, keep) if k]), int((~keep).sum())


# ---- quantisation (the shipped Japanese file stores int8 rows with one float32 scale per row) ---------------------

def quantize_rows(v):
    v = np.asarray(v, dtype=np.float32)
    scale = (np.abs(v).max(axis=1) / 127).astype(np.float32)
    scale[scale == 0] = 1.0
    q = np.rint(v / scale[:, None]).astype(np.int8)
    return q, scale


def dequantize_rows(q, scale):
    x = q.astype(np.float32) * np.asarray(scale, dtype=np.float32)[:, None]
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def load_ja_rows(rows):
    """(ids int64, unit vectors float32) of the Japanese rows after the int8 round trip."""
    z = np.load(os.path.join(JA_DIR, f"emb_{rows}.npz"))
    q, scale = quantize_rows(z["vectors"])
    return np.asarray(z["ids"], dtype=np.int64), dequantize_rows(q, scale)


def load_products():
    out = {}
    with open(os.path.join(JA_DIR, "products.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                out[int(p["productId"])] = p
    return out


# ---- classes ----------------------------------------------------------------------------------------------------

def read_labels(path):
    """{box_id: (name, tag, note)} from one labeller's tsv (header box_id name confidence tag note)."""
    out = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = (line.split("\t") + [""] * 5)[:5]
            out[int(parts[0])] = (parts[1].strip(), parts[3].strip(), parts[4].strip())
    return out


def assign_class(a, b, in_agreed):
    """J / other_foreign / E / left_out from two labellers' (name, tag, note) rows (None when a labeller has none)."""
    a, b = a or ("", "", ""), b or ("", "", "")
    foreign_a, foreign_b = a[1] == FOREIGN_TAG, b[1] == FOREIGN_TAG
    says_japan = lambda r: "japan" in (r[1] + " " + r[2]).lower()  # noqa: E731
    if foreign_a and foreign_b and says_japan(a) and says_japan(b):
        return "J"
    if foreign_a or foreign_b:
        return "other_foreign"
    if in_agreed:
        return "E"
    return "left_out"


def classes_for_set(label_dir):
    la = read_labels(os.path.join(label_dir, "A", "labels.tsv"))
    lb = read_labels(os.path.join(label_dir, "B", "labels.tsv"))
    agreed = {int(r["box_id"]): r for r in read_json(os.path.join(label_dir, "agreed.json"))}
    out = {}
    for box in sorted(set(la) | set(lb) | set(agreed)):
        ag = agreed.get(box)
        out[box] = {"cls": assign_class(la.get(box), lb.get(box), ag is not None),
                    "name_a": (la.get(box) or ("",))[0], "name_b": (lb.get(box) or ("",))[0],
                    "score_h": ag["score_h"] if ag else None, "site": ag["site"] if ag else None,
                    "listing": ag["listing"] if ag else None}
    return out


def read_feature_rows():
    with open(os.path.join(FEAT_DIR, "features.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def cmd_classes(out_dir=OUT):
    feats = read_feature_rows()
    labels = {s: classes_for_set(os.path.join(REPORTS, s, "labels")) for s in SETS}
    boxes, no_label, diffs = [], 0, []
    for i, r in enumerate(feats):
        c = labels.get(r["set"], {}).get(r.get("box_id"))  # a set without a labels folder (photos30) has no label rows
        if c is None:
            no_label += 1
            c = {"cls": "left_out", "name_a": "", "name_b": "", "score_h": None}
        if c["score_h"] is not None:
            diffs.append(abs(c["score_h"] - r["h"]))
        h = c["score_h"] if c["score_h"] is not None else r["h"]
        boxes.append({"row": i, "set": r["set"], "box_id": r.get("box_id"), "cls": c["cls"], "h": h, "band": band_of(h),
                      "site": r.get("site"), "listing": r.get("listing"), "name_a": c["name_a"], "name_b": c["name_b"]})
    counts = {s: {k: 0 for k in ("J", "other_foreign", "E", "left_out")} for s in SETS}
    for b in boxes:
        counts.setdefault(b["set"], {k: 0 for k in ("J", "other_foreign", "E", "left_out")})
        counts[b["set"]][b["cls"]] += 1
    band_counts = {k: {bd: 0 for bd in BANDS} for k in ("J", "E")}
    for b in boxes:
        if b["cls"] in band_counts:
            band_counts[b["cls"]][b["band"]] += 1
    os.makedirs(out_dir, exist_ok=True)
    write_json(os.path.join(out_dir, "classes.json"), {"created": now_iso(), "counts": counts, "band_counts": band_counts,
                                                       "boxes": boxes})
    print(f"boxes: {len(boxes)}; boxes with no label row (left out): {no_label}")
    print(f"{'set':18s} {'J':>5s} {'other_foreign':>14s} {'E':>6s} {'left_out':>9s}")
    for s in counts:
        c = counts[s]
        print(f"{s:18s} {c['J']:5d} {c['other_foreign']:14d} {c['E']:6d} {c['left_out']:9d}")
    print("by height band (all five sets): " + " | ".join(f"{bd}: J {band_counts['J'][bd]}, E {band_counts['E'][bd]}" for bd in BANDS))
    for name, sets in (("fit pool", FIT_POOL), ("final test", [FINAL_SET])):
        bc = {k: {bd: sum(1 for b in boxes if b["set"] in sets and b["cls"] == k and b["band"] == bd) for bd in BANDS}
              for k in ("J", "E")}
        print(f"by height band ({name}): " + " | ".join(f"{bd}: J {bc['J'][bd]}, E {bc['E'][bd]}" for bd in BANDS))
    print(f"largest |score_h - h| where both exist: {max(diffs):.4f} over {len(diffs)} boxes")


# ---- sanity and features ----------------------------------------------------------------------------------------

def load_emb(n_expected=None):
    emb = np.load(os.path.join(FEAT_DIR, "emb.npy")).astype(np.float32)
    if n_expected is not None and len(emb) != n_expected:
        raise ValueError(f"emb.npy has {len(emb)} rows but features.jsonl has {n_expected}")
    return emb


def cmd_sanity():
    lower_priority()
    feats = read_feature_rows()
    emb = load_emb(len(feats))
    cat, n_dig = english_catalog()
    _, sims = cat.top_k_batch(emb, k=1)
    best = sims[:, 0].astype(np.float64)
    stored = np.array([r["cos1"] for r in feats], dtype=np.float64)
    diff = np.abs(best - stored)
    print(f"boxes {len(feats)}; English rows used {len(cat.ids)} ({n_dig} digital-only rows left out)")
    print(f"largest |best English cosine - stored cos1|: {diff.max():.6f}")
    print(f"rows differing by more than {SANITY_TOL}: {int((diff > SANITY_TOL).sum())}")


def cmd_features(rows):
    lower_priority()
    feats = read_feature_rows()
    emb = load_emb(len(feats))
    en, _ = english_catalog()
    ja_ids, ja_vec = load_ja_rows(rows)
    ja = Catalog(ja_vec, [int(i) for i in ja_ids])
    en_idx, en_cos = en.top_k_batch(emb, k=5)
    ja_idx, ja_cos = ja.top_k_batch(emb, k=5)
    best_en = en_cos[:, 0].astype(np.float64)
    best_ja = ja_cos[:, 0].astype(np.float64)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"features_{rows}.npz")
    np.savez(path, best_en=best_en, best_ja=best_ja, d=best_ja - best_en, ja_ids=ja_ids[ja_idx], ja_cos=ja_cos,
             en_ids=np.array(en.ids)[en_idx], en_cos=en_cos)
    print(f"wrote {path}: {len(feats)} boxes, {len(ja_ids)} Japanese rows, {len(en.ids)} English rows")


# ---- fit --------------------------------------------------------------------------------------------------------

def evaluate_grid(data):
    """data[rows] = {d, h, cls} (arrays over the boxes to use). Returns one cell per (rows, H_MIN, m)."""
    cells = []
    for rows in ROWS_CHOICES:
        d, h, cls = data[rows]["d"], data[rows]["h"], data[rows]["cls"]
        is_j, is_e = cls == "J", cls == "E"
        for h_min in H_MINS:
            sel = h >= h_min
            bands = {bd: np.array([band_of(x) == bd for x in h]) for bd in FIT_BANDS}
            for m in M_GRID:
                flag = d >= m - M_EPS
                j, e = sel & is_j, sel & is_e
                cell = {"rows": rows, "h_min": h_min, "m": m, "j_n": int(j.sum()), "j_flag": int((j & flag).sum()),
                        "e_n": int(e.sum()), "e_flag": int((e & flag).sum()), "bands": {}}
                ok = cell["e_flag"] * 100 <= 2 * cell["e_n"]
                for bd in FIT_BANDS:
                    be, bj = e & bands[bd], j & bands[bd]
                    b = {"e_n": int(be.sum()), "e_flag": int((be & flag).sum()), "j_n": int(bj.sum()),
                         "j_flag": int((bj & flag).sum())}
                    cell["bands"][bd] = b
                    if b["e_n"] >= BAND_MIN_ENGLISH and b["e_flag"] * 100 > 3 * b["e_n"]:
                        ok = False
                cell["ok"] = bool(ok)
                cell["recall"] = cell["j_flag"] / cell["j_n"] if cell["j_n"] else None
                cell["false_flag_rate"] = cell["e_flag"] / cell["e_n"] if cell["e_n"] else None
                cells.append(cell)
    return cells


def pick(cells):
    """Highest recall among cells meeting the constraint; then H_MIN 150, T rows, larger m. None if the best is under 40%."""
    eligible = [c for c in cells if c["ok"] and c["j_n"] > 0 and Fraction(c["j_flag"], c["j_n"]) >= RECALL_FLOOR]
    if not eligible:
        return None
    return max(eligible, key=lambda c: (Fraction(c["j_flag"], c["j_n"]), c["h_min"] == 150, c["rows"] == "T", c["m"]))


def load_classes(out_dir):
    return read_json(os.path.join(out_dir, "classes.json"))["boxes"]


def load_features(out_dir, rows):
    path = os.path.join(out_dir, f"features_{rows}.npz")
    if not os.path.exists(path):
        raise StudyRefusal(f"{path} is missing: run `features --rows {rows}` first")
    return np.load(path)


def cmd_fit(out_dir=OUT):
    if os.path.exists(os.path.join(out_dir, "freeze.json")):
        raise StudyRefusal("freeze.json already exists: the fit is done once")
    boxes = [b for b in load_classes(out_dir) if b["set"] in FIT_POOL]
    idx = np.array([b["row"] for b in boxes], dtype=int)
    h = np.array([b["h"] for b in boxes], dtype=float)
    cls = np.array([b["cls"] for b in boxes])
    data = {rows: {"d": np.asarray(load_features(out_dir, rows)["d"], dtype=float)[idx], "h": h, "cls": cls}
            for rows in ROWS_CHOICES}
    cells = evaluate_grid(data)
    best = pick(cells)
    write_json(os.path.join(out_dir, "fit.json"), {
        "created": now_iso(), "fit_pool": FIT_POOL, "m_grid": M_GRID, "recall_floor": float(RECALL_FLOOR),
        "boxes": {"J": int((cls == "J").sum()), "E": int((cls == "E").sum())}, "cells": cells, "pick": best})
    if best is None:
        freeze = {"ship": False, "reason": "no choice met the constraint with recall of at least 40%", "timestamp": now_iso()}
    else:
        freeze = {"ship": True, "rows": best["rows"], "h_min": best["h_min"], "m": best["m"],
                  "fit": {k: best[k] for k in ("recall", "j_flag", "j_n", "e_flag", "e_n", "false_flag_rate", "bands")},
                  "timestamp": now_iso()}
    write_json(os.path.join(out_dir, "freeze.json"), freeze)
    print("pick:", json.dumps(freeze))
    if best is None:
        print("NOTHING QUALIFIES: the rule is not shipped and the final test is not run.")


# ---- final test -------------------------------------------------------------------------------------------------

def gates(e_flag, e_n, j_flag, j_n):
    """G1 (English safety) and G2 (usefulness), exactly as preregistered."""
    lo, hi = wilson(e_flag, e_n)
    g1_rate = e_flag / e_n if e_n else None
    g1 = bool(e_n and e_flag * 100 <= 3 * e_n and hi <= G1_MAX_WILSON_UPPER)
    g2_recall = j_flag / j_n if j_n else None
    g2 = bool(j_n and j_flag * 5 >= 3 * j_n)
    jlo, jhi = wilson(j_flag, j_n)
    return {"G1": {"pass": g1, "false_flags": e_flag, "english_boxes": e_n, "rate": g1_rate, "wilson95": [lo, hi],
                   "need": "rate <= 3.0% and Wilson upper <= 5.0%"},
            "G2": {"pass": g2, "flagged": j_flag, "japanese_boxes": j_n, "recall": g2_recall, "wilson95": [jlo, jhi],
                   "need": "recall >= 60%"}}


def rate_rows(boxes, flagged, key, order):
    """Per-group recall (J) and false-flag rate (E), each with Wilson 95%."""
    out = {}
    for g in order:
        row = {}
        for cls, name in (("J", "recall"), ("E", "false_flags")):
            sel = [i for i, b in enumerate(boxes) if b["cls"] == cls and b[key] == g]
            k = int(sum(flagged[i] for i in sel))
            row[name] = {"flagged": k, "n": len(sel), "rate": k / len(sel) if sel else None, "wilson95": list(wilson(k, len(sel)))}
        out[g] = row
    return out


def classify_today(feat, boxes, flagged_mask):
    """How many English boxes today's rule (fusion_rule.classify) would confirm, and how many of them the new rule flags."""
    try:
        from fusion_rule import classify
        nm = run_real_photos().load_names()
    except Exception as err:  # names or the port are not on disk: say so, do not approximate
        return {"available": False, "reason": f"{type(err).__name__}: {err}"}
    confirmed = flagged_confirmed = accepted = 0
    for i, b in enumerate(boxes):
        if b["cls"] != "E":
            continue
        ids = [str(x) for x in feat["en_ids"][b["row"]]]
        read = classify([float(x) for x in feat["en_cos"][b["row"]]], ids, [nm.get(x) for x in ids], int(round(b["h"])))
        accepted += bool(read.accepted)
        if read.label == "sure":
            confirmed += 1
            flagged_confirmed += bool(flagged_mask[i])
    return {"available": True, "english_boxes": sum(b["cls"] == "E" for b in boxes), "accepted": accepted,
            "confirmed": confirmed, "confirmed_and_flagged": flagged_confirmed,
            "note": "scripts/fusion_rule.py is a partial port of fusion.js (no reference rows, adapter terms, title/OCR, set caps); "
                    "'confirmed' means label == 'sure'; inputs are the five best English rows (digital-only left out), "
                    "their names from card-catalog meta, and the box height."}


def cmd_final(out_dir=OUT):
    freeze_path = os.path.join(out_dir, "freeze.json")
    if not os.path.exists(freeze_path):
        raise StudyRefusal("freeze.json does not exist: run `fit` first")
    freeze = read_json(freeze_path)
    if not freeze.get("ship") or "rows" not in freeze:
        raise StudyRefusal("freeze.json holds no pick: the rule is not shipped and the final test is not run")
    if os.path.exists(os.path.join(out_dir, "final.json")):
        raise StudyRefusal("final.json already exists: the final test is scored once")
    rows, h_min, m = freeze["rows"], freeze["h_min"], freeze["m"]
    feat = load_features(out_dir, rows)
    boxes = [b for b in load_classes(out_dir) if b["set"] == FINAL_SET]
    d = np.array([feat["d"][b["row"]] for b in boxes], dtype=float)
    flagged = d >= m - M_EPS
    in_scope = np.array([b["h"] >= h_min for b in boxes])
    is_j = np.array([b["cls"] == "J" for b in boxes]) & in_scope
    is_e = np.array([b["cls"] == "E" for b in boxes]) & in_scope
    is_o = np.array([b["cls"] == "other_foreign" for b in boxes])
    result = gates(int((is_e & flagged).sum()), int(is_e.sum()), int((is_j & flagged).sum()), int(is_j.sum()))
    scoped = [b for b, s in zip(boxes, in_scope) if s]
    scoped_flag = flagged[in_scope]
    products = load_products()
    pname = lambda pid: (products.get(int(pid)) or {}).get("name", "")  # noqa: E731

    candidates, named, top1, top5 = [], 0, 0, 0
    for i in np.flatnonzero(is_j & flagged):
        b = boxes[i]
        ids = [int(x) for x in feat["ja_ids"][b["row"]]]
        cos = [float(x) for x in feat["ja_cos"][b["row"]]]
        candidates.append({"set": b["set"], "box_id": b["box_id"], "listing": b["listing"], "h": b["h"], "d": float(d[i]),
                           "name_a": b["name_a"], "name_b": b["name_b"],
                           "candidates": [{"product_id": p, "name": pname(p), "cos": c} for p, c in zip(ids, cos)]})
        na, nb = norm_card_name(b["name_a"]), norm_card_name(b["name_b"])
        if is_latin_name(b["name_a"]) and is_latin_name(b["name_b"]) and na and na == nb:
            named += 1
            names = [norm_card_name(pname(p)) for p in ids]
            top1 += names[0] == na
            top5 += na in names
    write_json(os.path.join(out_dir, "final_candidates.json"), {"set": FINAL_SET, "rows": rows, "m": m, "h_min": h_min,
                                                                "flagged_j_boxes": candidates})
    other_flag = int((is_o & in_scope & flagged).sum())
    final = {
        "created": now_iso(), "set": FINAL_SET, "frozen": {"rows": rows, "h_min": h_min, "m": m}, "gates": result,
        "ship": bool(result["G1"]["pass"] and result["G2"]["pass"]),
        "by_band": rate_rows(scoped, scoped_flag, "band", FIT_BANDS),
        "by_site": rate_rows(scoped, scoped_flag, "site", sorted({b["site"] for b in scoped})),
        "other_foreign": {"at_or_above_h_min": int((is_o & in_scope).sum()), "flagged_at_or_above_h_min": other_flag,
                          "all_heights": int(is_o.sum()), "flagged_all_heights": int((is_o & flagged).sum())},
        "flagged_j_name_check": {"flagged_j_boxes": len(candidates), "same_latin_name_both_labellers": named,
                                 "first_candidate": top1, "in_first_five": top5,
                                 "needs_visual_check": named < MIN_NAMED_BOXES,
                                 "name_normalisation": "base_name after dropping parenthesised text"},
        "today_rule": classify_today(feat, scoped, scoped_flag),
    }
    write_json(os.path.join(out_dir, "final.json"), final)
    print(json.dumps({"gates": result, "ship": final["ship"]}, indent=1))


# ---- synthetic reads --------------------------------------------------------------------------------------------

def sha256_prefix(path, n=8):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def cmd_synth(rows, n=1000):
    lower_priority()
    from PIL import Image
    import eval_synth_small as E
    import heldout_eval as HE
    import large_crop_check as LC
    from pipeline.embed import ClipEmbedder
    if sha256_prefix(MODEL_E) != MODEL_E_SHA_PREFIX:
        raise StudyRefusal(f"{MODEL_E} does not start with sha256 {MODEL_E_SHA_PREFIX}")
    products = load_products()
    ja_ids, ja_vec = load_ja_rows(rows)
    ja = Catalog(ja_vec, [int(i) for i in ja_ids])
    have = sorted(int(p) for p in ja_ids if os.path.exists(os.path.join(JA_DIR, "images", f"{int(p)}.jpg")))
    sample = random.Random(HE.SEED).sample(have, min(n, len(have)))
    path_of = lambda p: os.path.join(JA_DIR, "images", f"{p}.jpg")  # noqa: E731
    fulls = []
    for p in sample:
        with Image.open(path_of(p)) as im:
            fulls.append(im.convert("RGB"))
    rng = random.Random(HE.SEED)
    inputs = {f"h{h}": [E.degrade(img, h, rng) for img in fulls] for h in HE.HEIGHTS}
    for h in HE.LARGE:
        inputs[f"large_{h or 'full'}"] = [LC.png_bytes_of(path_of(p)) if h is None else LC.degrade(img, h)
                                          for p, img in zip(sample, fulls)]
    en, _ = english_catalog()
    nm = run_real_photos().load_names()
    en_names = [norm_card_name(nm.get(c)) for c in en.ids]
    emb = ClipEmbedder(MODEL_E)
    row_of = {int(p): i for i, p in enumerate(ja_ids)}
    truth_norm = [norm_card_name(products[p]["name"]) for p in sample]
    cells = {}
    for cell, datas in inputs.items():
        qs = np.stack([emb(x, trusted=True) for x in datas])
        j_idx, j_cos = ja.top_k_batch(qs, k=5)
        e_idx, e_cos = en.top_k_batch(qs, k=1)
        j_ids = [[int(ja_ids[k]) for k in r] for r in j_idx]
        j_names = [[norm_card_name(products[p]["name"]) for p in r] for r in j_ids]
        t1 = sum(r[0] == p for r, p in zip(j_ids, sample))
        t5 = sum(p in r for r, p in zip(j_ids, sample))
        n1 = sum(r[0] == t for r, t in zip(j_names, truth_norm))
        n5 = sum(t in r for r, t in zip(j_names, truth_norm))
        twin = sum(bool(t) and en_names[e_idx[i, 0]] == t for i, t in enumerate(truth_norm))
        dd = j_cos[:, 0].astype(np.float64) - e_cos[:, 0].astype(np.float64)
        N = len(sample)
        cells[cell] = {"n": N, "top1_id": t1, "top5_id": t5, "top1_name": n1, "top5_name": n5, "twin_coverage": twin,
                       "wilson95": {k: list(wilson(v, N)) for k, v in
                                    (("top1_id", t1), ("top5_id", t5), ("top1_name", n1), ("top5_name", n5), ("twin_coverage", twin))},
                       "mean_d": float(dd.mean()), "true_row_present": all(p in row_of for p in sample)}
        print(cell, json.dumps({k: v for k, v in cells[cell].items() if k != "wilson95"}), flush=True)
    write_json(os.path.join(OUT, f"synth_{rows}.json"),
               {"created": now_iso(), "rows": rows, "seed": HE.SEED, "n": len(sample), "model": MODEL_E,
                "name_normalisation": "base_name after dropping parenthesised text", "cells": cells})


# ---- report -----------------------------------------------------------------------------------------------------

def pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def pct_ci(d):
    return f"{d['flagged']}/{d['n']} = {pct(d['rate'])} [{100 * d['wilson95'][0]:.1f}, {100 * d['wilson95'][1]:.1f}]"


def cmd_report(out_dir=OUT):
    def get(name):
        p = os.path.join(out_dir, name)
        return read_json(p) if os.path.exists(p) else None

    classes, fit, freeze, final = get("classes.json"), get("fit.json"), get("freeze.json"), get("final.json")
    synth = {r: get(f"synth_{r}.json") for r in ROWS_CHOICES}
    L = ["# Japanese cards: telling them from English ones (study of 2026-11-17)", ""]
    if final:
        g = final["gates"]
        L += [f"The automatic rule {'passes' if final['ship'] else 'does not pass'} its ship gates. "
              f"G1 (English safety) {'passed' if g['G1']['pass'] else 'failed'}: {g['G1']['false_flags']} of {g['G1']['english_boxes']} "
              f"English boxes at or above the frozen height were flagged ({pct(g['G1']['rate'])}, 95% Wilson upper end "
              f"{100 * g['G1']['wilson95'][1]:.1f}%; limits 3.0% and 5.0%). "
              f"G2 (usefulness) {'passed' if g['G2']['pass'] else 'failed'}: {g['G2']['flagged']} of {g['G2']['japanese_boxes']} "
              f"Japanese boxes were flagged ({pct(g['G2']['recall'])}; limit 60%)."]
        if not final["ship"]:
            L += ["", "Because a gate failed, the automatic rule does not ship. The Japanese catalog can still ship for manual search and pricing."]
    elif freeze is not None and not freeze.get("ship"):
        L += ["No choice met the fit constraint with recall of at least 40%, so the rule is not shipped and the final test was not run."]
    elif freeze is not None:
        L += ["The rule is frozen but the final test has not been run yet, so no gate has a result."]
    else:
        L += ["The fit has not been run yet, so there is no verdict."]
    L += ["", "The rules of the study are in `PREREGISTRATION.md` (with Amendment 1). This report is written by `scripts/japanese_study.py report`.", ""]
    L += ["## Frozen rule", ""]
    if freeze and freeze.get("ship"):
        L += [f"Rows: {freeze['rows']}. `H_MIN`: {freeze['h_min']} px. Margin `m`: {freeze['m']:.3f}. A box is treated as Japanese when its "
              f"height is at least `H_MIN` and `d >= m`, where `d` is the best Japanese cosine minus the best English cosine. "
              f"Frozen at {freeze['timestamp']}.", ""]
    elif freeze:
        L += [f"Nothing was frozen: {freeze.get('reason', '')}.", ""]
    else:
        L += ["Not frozen yet.", ""]
    if classes:
        L += ["## Boxes by class", "", "| set | J | other foreign | E | left out |", "|---|---|---|---|---|"]
        for s, c in classes["counts"].items():
            L.append(f"| {s} | {c['J']} | {c['other_foreign']} | {c['E']} | {c['left_out']} |")
        L.append("")
    if fit and freeze and freeze.get("ship"):
        f = freeze["fit"]
        L += ["## Fit pool at the pick", "",
              f"Japanese recall {f['j_flag']}/{f['j_n']} = {pct(f['recall'])}; English false flags {f['e_flag']}/{f['e_n']} = {pct(f['false_flag_rate'])}.", "",
              "| band | English boxes | English flagged | Japanese boxes | Japanese flagged |", "|---|---|---|---|---|"]
        for bd, b in f["bands"].items():
            L.append(f"| {bd} | {b['e_n']} | {b['e_flag']} | {b['j_n']} | {b['j_flag']} |")
        L += ["", f"The full grid ({len(fit['cells'])} choices) is in `fit.json`.", ""]
    if final:
        L += ["## Final test (check-20261113, scored once)", "", "| band | Japanese flagged (recall) | English flagged |", "|---|---|---|"]
        for bd, r in final["by_band"].items():
            L.append(f"| {bd} | {pct_ci(r['recall'])} | {pct_ci(r['false_flags'])} |")
        L += ["", "| site | Japanese flagged (recall) | English flagged |", "|---|---|---|"]
        for s, r in final["by_site"].items():
            L.append(f"| {s} | {pct_ci(r['recall'])} | {pct_ci(r['false_flags'])} |")
        o, nmchk, tr = final["other_foreign"], final["flagged_j_name_check"], final["today_rule"]
        L += ["", f"Other foreign boxes flagged: {o['flagged_at_or_above_h_min']} of {o['at_or_above_h_min']} at or above the frozen height "
              f"({o['flagged_all_heights']} of {o['all_heights']} at any height).", ""]
        if nmchk["needs_visual_check"]:
            L += [f"Of {nmchk['flagged_j_boxes']} flagged Japanese boxes, only {nmchk['same_latin_name_both_labellers']} have the same Latin-script "
                  f"name from both labellers, fewer than {MIN_NAMED_BOXES}. The name figure is therefore replaced by a blind visual check of "
                  "`final_candidates.json`, reported separately with its count.", ""]
        else:
            L += [f"Of {nmchk['same_latin_name_both_labellers']} flagged Japanese boxes with the same Latin-script name from both labellers, "
                  f"the first Japanese candidate has that name in {nmchk['first_candidate']} and one of the first five in {nmchk['in_first_five']}.", ""]
        if tr["available"]:
            L += [f"Today's rule (`scripts/fusion_rule.py`, a partial port) would confirm {tr['confirmed']} of {tr['english_boxes']} English boxes; "
                  f"the new rule flags {tr['confirmed_and_flagged']} of those.", ""]
        else:
            L += [f"Today's rule could not be run from what is on disk ({tr['reason']}).", ""]
    for r in ROWS_CHOICES:
        s = synth[r]
        if not s:
            continue
        L += [f"## Synthetic reads, {r} rows (n={s['n']}, seed {s['seed']})", "",
              "| cell | top-1 id | top-5 id | top-1 name | top-5 name | twin coverage | mean d |", "|---|---|---|---|---|---|---|"]
        for cell, c in s["cells"].items():
            ci = lambda k: f"{100 * c[k] / c['n']:.1f}% [{100 * c['wilson95'][k][0]:.1f}, {100 * c['wilson95'][k][1]:.1f}]"  # noqa: E731
            L.append(f"| {cell} | {ci('top1_id')} | {ci('top5_id')} | {ci('top1_name')} | {ci('top5_name')} | {ci('twin_coverage')} | {c['mean_d']:.3f} |")
        L.append("")
    L += ["## Limits", "",
          "The boxes come from five blind-labelled harvests, so the Japanese counts are small and the intervals are wide. "
          "The embeddings are from the Python twin of the extension's crop; the built extension must still agree with the Python flag on at least 95% of matched fit-pool boxes. "
          "Some labellers wrote Japanese names in Japanese script while the catalog names are English, which limits the name figure. "
          "Synthetic reads start from clean catalog pictures and do not stand in for photos. "
          "Names are compared with the heldout report's normalisation after dropping parenthesised text such as '(Mirror Holo)'. "
          "Today's rule is a partial offline port of the extension's fusion code.", ""]
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print(f"wrote {os.path.join(out_dir, 'README.md')}")


# ---- entry point ------------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("classes", "sanity", "fit", "final", "report"):
        sub.add_parser(name)
    for name in ("features", "synth"):
        p = sub.add_parser(name)
        p.add_argument("--rows", choices=ROWS_CHOICES, required=True)
        if name == "synth":
            p.add_argument("--n", type=int, default=1000)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "classes":
            cmd_classes()
        elif a.cmd == "sanity":
            cmd_sanity()
        elif a.cmd == "features":
            cmd_features(a.rows)
        elif a.cmd == "fit":
            cmd_fit()
        elif a.cmd == "final":
            cmd_final()
        elif a.cmd == "synth":
            cmd_synth(a.rows, a.n)
        elif a.cmd == "report":
            cmd_report()
    except StudyRefusal as err:
        print(f"REFUSED: {err}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
