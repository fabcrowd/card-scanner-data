"""New-set evaluation (preregistered in data/reports/newsets-20261117/PREREGISTRATION.md). CPU only. Tunes nothing.

Sub-commands, each rerunnable: fetch, rows, reads, score, report.
  fetch   download the live catalog-extra.bin (hash checked) and the new cards' pictures
  rows    T rows (live file, v2.0 model) and E rows (model E on clean art, nightly quantisation); row checks
  reads   synthetic reads of new + control cards exactly as scripts/heldout_eval.py makes them (seed 777), embedded with model E
  score   search with T rows and with E rows; top-1/top-5/name tables, paired E-T difference; writes results.json
  report  writes README.md
"""
import ctypes
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "train"))
import eval_synth_small as E  # noqa: E402
import large_crop_check as LC  # noqa: E402
from gpu_data import norm_name  # noqa: E402
from pipeline.catalog import Catalog  # noqa: E402
from pipeline.embed import ClipEmbedder  # noqa: E402
from pipeline.new_sets import read_extra  # noqa: E402
from run_real_photos import load_names  # noqa: E402
from score_real_labels import base_name  # noqa: E402

OUT = os.path.join(ROOT, "data", "reports", "newsets-20261117")
WORK = os.path.join(OUT, "work")
ART = os.path.join(WORK, "art")
EXTRA_PATH = os.path.join(WORK, "catalog-extra.bin")
EXTRA_URL = "https://github.com/fabcrowd/card-scanner-data/releases/download/pack-20261117-33/catalog-extra.bin"
EXTRA_SHA = "4ab3ccc5fcdc1eb348ec8ad1ec19a454541dd77a431575a4c2d7aff25d1d7901"
EXTRA_SIZE = 233385
V2_MODEL = os.path.join(ROOT, "assets", "clip_visual_int8.onnx")
E_MODEL = os.path.join(ROOT, "train", "out", "fastE_all", "epoch16-ema-int8.onnx")
V2_SHA, E_SHA = "77538aad", "0dba1302"
DIGITAL = re.compile(r"^(A\d|B\d|P-A)")
NEW_SETS = {"me06", "30th"}
CONTROL_PREFIX = "me05-"
HEIGHTS = [90, 120, 150, 200]
LARGE = [None, 400, 300]
SEED = 777
SCALE = 127
THREADS = 8
DELAY = 0.15
UA = "card-scanner-eval/1"
CELLS = [f"h{h}" for h in HEIGHTS] + ["large_full", "large_400", "large_300"]
PRIMARY, GUARD, MARGIN = "h200", "large_full", 2.0


def below_normal():
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def check_models():
    for path, want in ((V2_MODEL, V2_SHA), (E_MODEL, E_SHA)):
        got = sha(path)
        if not got.startswith(want):
            sys.exit(f"STOP: {path} sha256 {got[:8]} != {want}")


def embedder(path):
    import onnxruntime as ort

    emb = ClipEmbedder(path)
    so = ort.SessionOptions()
    so.intra_op_num_threads = THREADS
    so.inter_op_num_threads = 1
    emb._session = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    return emb


def new_cards():
    cards, rows = read_extra(open(EXTRA_PATH, "rb").read())
    keep = [i for i, c in enumerate(cards) if not DIGITAL.match(c["id"])]
    cards = [cards[i] for i in keep]
    assert {c["set"] for c in cards} == NEW_SETS, {c["set"] for c in cards}
    return cards, rows[keep]


def art_path(card):
    return os.path.join(ART, card["id"] + ".webp")


def get(url, retries_left=0):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


# ---------------------------------------------------------------- fetch
def cmd_fetch():
    os.makedirs(ART, exist_ok=True)
    if not os.path.exists(EXTRA_PATH):
        data = get(EXTRA_URL)
        if len(data) != EXTRA_SIZE or hashlib.sha256(data).hexdigest() != EXTRA_SHA:
            sys.exit(f"STOP: catalog-extra.bin size/hash mismatch ({len(data)} bytes)")
        open(EXTRA_PATH, "wb").write(data)
    if sha(EXTRA_PATH) != EXTRA_SHA:
        sys.exit("STOP: catalog-extra.bin hash mismatch")
    print("catalog-extra.bin ok", EXTRA_SHA)
    cards, _ = new_cards()
    by_set = {s: sum(c["set"] == s for c in cards) for s in sorted(NEW_SETS)}
    print("new cards", len(cards), by_set)
    refused, got, skipped = 0, 0, 0
    for c in cards:
        p = art_path(c)
        if os.path.exists(p):
            skipped += 1
            continue
        time.sleep(DELAY)
        try:
            data = get(c["image"])
        except urllib.error.HTTPError as err:
            if err.code in (403, 429):
                refused += 1
                if refused >= 3:
                    sys.exit(f"STOP: host refused ({err.code}) {refused} times in a row at {c['id']}")
            print("fail", c["id"], err)
            continue
        except Exception as err:  # noqa: BLE001
            print("fail", c["id"], err)
            continue
        refused = 0
        open(p, "wb").write(data)
        got += 1
    missing = [c["id"] for c in cards if not os.path.exists(art_path(c))]
    print(f"pictures: downloaded {got}, already present {skipped}, missing {len(missing)} {missing[:10]}")


# ---------------------------------------------------------------- rows
def unit(int8_rows):
    v = np.asarray(int8_rows, dtype=np.float32) / SCALE
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def cmd_rows():
    check_models()
    cards, live = new_cards()
    e_emb, v_emb = embedder(E_MODEL), embedder(V2_MODEL)
    t_unit = unit(live)
    e_int8, v2_unit = [], []
    for c in cards:
        data = open(art_path(c), "rb").read()
        ve = e_emb(data, trusted=True)
        e_int8.append(np.clip(np.rint(ve * SCALE), -SCALE, SCALE).astype(np.int8))
        v2_unit.append(v_emb(data, trusted=True))
    e_int8 = np.stack(e_int8)
    e_unit = unit(e_int8)
    cos_v2_live = np.einsum("ij,ij->i", np.stack(v2_unit), t_unit)
    cos_t_e = np.einsum("ij,ij->i", t_unit, e_unit)
    np.savez(os.path.join(WORK, "rows.npz"), ids=np.array([c["id"] for c in cards]), T=t_unit, E=e_unit, e_int8=e_int8)
    check = {"n": len(cards),
             "v2_vs_live": {"mean": float(cos_v2_live.mean()), "min": float(cos_v2_live.min()),
                            "min_id": cards[int(cos_v2_live.argmin())]["id"]},
             "T_vs_E": {"mean": float(cos_t_e.mean()), "min": float(cos_t_e.min())}}
    json.dump(check, open(os.path.join(WORK, "row_check.json"), "w"), indent=2)
    print(json.dumps(check))


# ---------------------------------------------------------------- reads
def groups():
    """(cat, names, new cards, control rows) with a fixed order: new cards first, then control cards."""
    cat = Catalog.load(os.path.join(ROOT, "assets", "catalog.bin.gz"), os.path.join(ROOT, "assets", "id_map.json"))
    names = load_names()
    cards, _ = new_cards()
    control = [c for c in cat.ids if c.startswith(CONTROL_PREFIX)]
    return cat, names, cards, control


def cmd_reads():
    check_models()
    cat, names, cards, control = groups()
    paths = [art_path(c) for c in cards] + [E.image_path(c) for c in control]
    assert all(paths), "a control picture is missing"
    rng = random.Random(SEED)
    fulls = []
    for p in paths:
        with Image.open(p) as im:
            fulls.append(im.convert("RGB"))
    inputs = {f"h{h}": [E.degrade(img, h, rng) for img in fulls] for h in HEIGHTS}
    for h in LARGE:
        inputs["large_" + ("full" if h is None else str(h))] = [
            LC.png_bytes_of(p) if h is None else LC.degrade(img, h) for p, img in zip(paths, fulls)]
    emb = embedder(E_MODEL)
    q = {cell: np.stack([emb(d, trusted=True) for d in inputs[cell]]) for cell in CELLS}
    np.savez(os.path.join(WORK, "queries.npz"), n_new=len(cards), n_control=len(control), **q)
    print("reads done", {k: v.shape for k, v in q.items()}, "new", len(cards), "control", len(control))


# ---------------------------------------------------------------- score
def wilson(k, n):
    if n == 0:
        return {"k": 0, "n": 0, "pct": None, "lo": None, "hi": None}
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"k": int(k), "n": int(n), "pct": 100 * p, "lo": 100 * (c - h), "hi": 100 * (c + h)}


def paired(a, b, n_boot=2000):
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    m = d[idx].mean(1)
    return {"diff": 100 * float(d.mean()), "lo": 100 * float(np.percentile(m, 2.5)), "hi": 100 * float(np.percentile(m, 97.5))}


def cmd_score():
    cat, names, cards, control = groups()
    rows = np.load(os.path.join(WORK, "rows.npz"))
    qs = np.load(os.path.join(WORK, "queries.npz"))
    n_new, n_ctl = int(qs["n_new"]), int(qs["n_control"])
    assert n_new == len(cards) and n_ctl == len(control)
    keep = [i for i, c in enumerate(cat.ids) if not DIGITAL.match(c)]
    base_ids = [cat.ids[i] for i in keep]
    base_vec = cat.vectors[keep]
    overlap = set(base_ids) & {c["id"] for c in cards}
    assert not overlap, overlap
    new_ids = [c["id"] for c in cards]
    new_set = set(new_ids)
    assert list(rows["ids"]) == new_ids
    name_of = {**names, **{c["id"]: c["name"] for c in cards}}
    truth_ids = new_ids + control
    truth_base = [base_name(name_of.get(i) or "") for i in truth_ids]
    shipped_norm = {norm_name(names.get(i)) for i in base_ids}
    novel = np.array([norm_name(c["name"]) not in shipped_norm for c in cards])
    res = {"counts": {"new": n_new, "control": n_ctl, "new_name_novel": int(novel.sum()), "new_name_seen": int((~novel).sum()),
                      "search_rows_shipped_nondigital": len(base_ids), "new_by_set": {s: sum(c["set"] == s for c in cards) for s in sorted(NEW_SETS)}},
           "cells": {}, "paired": {}, "decision": {}}
    flags = {}
    for rowset in ("T", "E"):
        cat2 = Catalog(np.vstack([base_vec, rows[rowset]]), base_ids + new_ids)
        search_names = [base_name(name_of.get(i) or "") for i in cat2.ids]
        res["cells"][rowset] = {}
        for cell in CELLS:
            idx, _ = cat2.top_k_batch(qs[cell], k=5)
            t1 = np.array([cat2.ids[idx[i, 0]] == truth_ids[i] for i in range(len(truth_ids))])
            t5 = np.array([truth_ids[i] in [cat2.ids[j] for j in idx[i]] for i in range(len(truth_ids))])
            nm = np.array([search_names[idx[i, 0]] == truth_base[i] for i in range(len(truth_ids))])
            on_new = np.array([cat2.ids[idx[i, 0]] in new_set for i in range(len(truth_ids))])
            flags[(rowset, cell)] = t1
            sl = {"new": slice(0, n_new), "control": slice(n_new, None)}
            out = {}
            for g, s in sl.items():
                out[g] = {m: wilson(int(a[s].sum()), len(a[s])) for m, a in (("top1_id", t1), ("top5_id", t5), ("top1_name", nm))}
                out[g]["first_hit_is_new_row"] = {"k": int(on_new[s].sum()), "n": len(on_new[s])}
            nv, sn = slice(0, n_new), slice(0, n_new)
            out["new_name_novel"] = {m: wilson(int(a[:n_new][novel].sum()), int(novel.sum())) for m, a in (("top1_id", t1), ("top5_id", t5), ("top1_name", nm))}
            out["new_name_seen"] = {m: wilson(int(a[:n_new][~novel].sum()), int((~novel).sum())) for m, a in (("top1_id", t1), ("top5_id", t5), ("top1_name", nm))}
            res["cells"][rowset][cell] = out
    for g, s in (("new", slice(0, n_new)), ("control", slice(n_new, None))):
        res["paired"][g] = {cell: paired(flags[("E", cell)][s], flags[("T", cell)][s]) for cell in CELLS}
    p = res["cells"]
    d_primary = p["E"][PRIMARY]["new"]["top1_id"]["pct"] - p["T"][PRIMARY]["new"]["top1_id"]["pct"]
    d_guard = p["E"][GUARD]["new"]["top1_id"]["pct"] - p["T"][GUARD]["new"]["top1_id"]["pct"]
    choose_e = d_primary > MARGIN and d_guard >= 0
    res["decision"] = {"primary_cell": PRIMARY, "guard_cell": GUARD, "margin_points": MARGIN,
                       "E_minus_T_primary_points": d_primary, "E_minus_T_large_full_points": d_guard,
                       "chosen": "E" if choose_e else "T"}
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
    print(json.dumps({"counts": res["counts"], "decision": res["decision"]}))


# ---------------------------------------------------------------- report
# ---------------------------------------------------------------- realboxes (Amendment 1)
REAL_C_SLACK, REAL_SG_SLACK, REAL_S_MAX_FRACTION = 2, 5, 0.005
ARMS = ("baseline", "T", "E")


def count_sgc(base_ok, arm_ok, base_sure_wrong, arm_sure_wrong):
    """S: right at baseline, not with the arm. G: wrong at baseline, right with the arm.
    C: confident (sure) wrong boxes under the arm minus the same count at baseline."""
    b, a = np.asarray(base_ok, bool), np.asarray(arm_ok, bool)
    return {"S": int((b & ~a).sum()), "G": int((~b & a).sum()),
            "C": int(np.asarray(arm_sure_wrong, bool).sum()) - int(np.asarray(base_sure_wrong, bool).sum())}


def decide_real(t, e, n_right_base):
    """Amendment 1 decision from the S/G/C of the T and E arms. decision: 'E', 'T', or 'owner_decision' when the T arm itself
    has C(T) > 2 or S(T) above 0.5% of the boxes right at baseline (then neither kind of row is recommended)."""
    c1 = {"lhs": e["C"], "limit": t["C"] + REAL_C_SLACK}
    c1["pass"] = c1["lhs"] <= c1["limit"]
    c2 = {"lhs": e["S"] - e["G"], "limit": t["S"] - t["G"] + REAL_SG_SLACK}
    c2["pass"] = c2["lhs"] <= c2["limit"]
    c3 = {"lhs": e["S"], "limit": REAL_S_MAX_FRACTION * n_right_base}
    c3["pass"] = c3["lhs"] <= c3["limit"] + 1e-9
    warn = bool(t["C"] > 2 or t["S"] > REAL_S_MAX_FRACTION * n_right_base + 1e-9)
    all_pass = c1["pass"] and c2["pass"] and c3["pass"]
    decision = "owner_decision" if warn else ("E" if all_pass else "T")
    return {"conditions": {"c1": c1, "c2": c2, "c3": c3}, "all_pass": bool(all_pass), "t_arm_warning": warn, "decision": decision}


def cmd_realboxes():
    import japanese_study as J
    from fusion_rule import classify

    cls = J.read_json(os.path.join(J.OUT, "classes.json"))["boxes"]
    agreed = {s: {int(r["box_id"]): r["name"] for r in J.read_json(os.path.join(J.REPORTS, s, "labels", "agreed.json"))} for s in J.SETS}
    boxes = [b for b in cls if b["cls"] == "E" and b["set"] in agreed and b["box_id"] in agreed[b["set"]]]
    skipped = sum(b["cls"] == "E" for b in cls) - len(boxes)
    emb = J.load_emb(len(J.read_feature_rows()))[[b["row"] for b in boxes]]
    cat = Catalog.load(os.path.join(ROOT, "assets", "catalog.bin.gz"), os.path.join(ROOT, "assets", "id_map.json"))
    keep = [i for i, c in enumerate(cat.ids) if not DIGITAL.match(c)]
    base_ids, base_vec = [cat.ids[i] for i in keep], cat.vectors[keep]
    cards, _ = new_cards()
    rows = np.load(os.path.join(WORK, "rows.npz"))
    new_ids = [c["id"] for c in cards]
    assert list(rows["ids"]) == new_ids
    name_of = {**load_names(), **{c["id"]: c["name"] for c in cards}}
    new_set = set(new_ids)
    truth = [base_name(agreed[b["set"]][b["box_id"]]) for b in boxes]
    heights = [int(round(b["h"])) for b in boxes]
    n = len(boxes)
    per = {}
    for arm in ARMS:
        extra = [] if arm == "baseline" else [rows[arm]]
        arm_cat = Catalog(np.vstack([base_vec] + extra), base_ids + ([] if arm == "baseline" else new_ids))
        idx, cos = arm_cat.top_k_batch(emb, k=5)
        recs = []
        for i in range(n):
            ids = [arm_cat.ids[j] for j in idx[i]]
            names = [name_of.get(x) for x in ids]
            sims = [float(x) for x in cos[i]]
            read = classify(sims, ids, names, heights[i])
            chosen_ok = base_name(name_of.get(read.id) or "") == truth[i]
            recs.append({"ids": ids, "cos": sims, "ok": base_name(names[0] or "") == truth[i], "new": ids[0] in new_set,
                         "sure_wrong": bool(read.label == "sure" and not chosen_ok)})
        per[arm] = recs
    flag = lambda arm, k: np.array([r[k] for r in per[arm]], bool)  # noqa: E731
    sets = np.array([b["set"] for b in boxes])
    bands = np.array([J.band_of(h) for h in [b["h"] for b in boxes]])
    base_ok, base_sw = flag("baseline", "ok"), flag("baseline", "sure_wrong")
    n_right = int(base_ok.sum())

    def arm_stats(arm, mask):
        return count_sgc(base_ok[mask], flag(arm, "ok")[mask], base_sw[mask], flag(arm, "sure_wrong")[mask])

    allm = np.ones(n, bool)
    arms = {a: arm_stats(a, allm) for a in ("T", "E")}
    dec = decide_real(arms["T"], arms["E"], n_right)
    by_band = {bd: {"boxes": int((bands == bd).sum()), "right_at_baseline": int((base_ok & (bands == bd)).sum()),
                    **{a: arm_stats(a, bands == bd) for a in ("T", "E")}} for bd in J.BANDS}
    by_set = {s: {"boxes": int((sets == s).sum()), "right_at_baseline": int((base_ok & (sets == s)).sum()),
                  **{a: arm_stats(a, sets == s) for a in ("T", "E")}} for s in J.SETS}
    first_hits = {}
    for arm in ("T", "E"):
        isnew, ok = flag(arm, "new"), flag(arm, "ok")
        cos1 = np.array([r["cos"][0] for r in per[arm]])
        stat = lambda v: {"n": int(len(v)), "mean": float(v.mean()) if len(v) else None,  # noqa: E731
                          "p90": float(np.percentile(v, 90)) if len(v) else None}
        first_hits[arm] = {"new_row_first_hits": int(isnew.sum()), "new_row_first_hits_with_agreed_name": int((isnew & ok).sum()),
                           "cos_new_row_first_hits": stat(cos1[isnew]), "cos_shipped_row_first_hits": stat(cos1[~isnew])}
    res = {"counts": {"boxes": n, "right_at_baseline": n_right, "skipped_class_E_without_agreed_name": int(skipped),
                      "by_set": {s: int((sets == s).sum()) for s in J.SETS}, "by_band": {bd: int((bands == bd).sum()) for bd in J.BANDS},
                      "shipped_nondigital_rows": len(base_ids), "new_rows_per_arm": len(new_ids)},
           "baseline": {"first_hit_name_accuracy": n_right / n, "right": n_right, "sure_wrong": int(base_sw.sum())},
           "arms": arms, "decision": dec, "by_band": by_band, "by_set": by_set, "first_hits": first_hits,
           "boxes": [{"set": b["set"], "box_id": b["box_id"], "h": b["h"], "agreed": truth[i],
                      **{a: {"ids": per[a][i]["ids"], "cos": [round(x, 4) for x in per[a][i]["cos"]], "ok": per[a][i]["ok"],
                             "sure_wrong": per[a][i]["sure_wrong"]} for a in ARMS}} for i, b in enumerate(boxes)]}
    json.dump(res, open(os.path.join(OUT, "realboxes.json"), "w"))
    show = {k: res[k] for k in ("counts", "baseline", "arms", "decision", "first_hits")}
    print(json.dumps(show, indent=1))


def real_section(rb):
    cnt, base, arms, dec, fh = rb["counts"], rb["baseline"], rb["arms"], rb["decision"], rb["first_hits"]
    L = ["## Real-box check (Amendment 1)", "",
         f"The check uses {cnt['boxes']} English boxes from real photos with an agreed card name, in sets "
         + ", ".join(f"`{s}` ({k})" for s, k in cnt["by_set"].items()) + ". By height band: "
         + ", ".join(f"{bd} px {k}" for bd, k in cnt["by_band"].items()) + ". "
         f"Searching the {cnt['shipped_nondigital_rows']} shipped non-digital rows alone, the first hit has the agreed name in {base['right']} boxes "
         f"({100 * base['first_hit_name_accuracy']:.1f} percent); {base['sure_wrong']} boxes are labelled sure with a wrong name at baseline. "
         f"Each arm adds the {cnt['new_rows_per_arm']} new-card rows (T rows or E rows).", "",
         "| arm | S (stolen) | G (gained) | C (added confident errors) |", "|---|---|---|---|"]
    for a in ("T", "E"):
        L.append(f"| {a} | {arms[a]['S']} | {arms[a]['G']} | {arms[a]['C']} |")
    cd = dec["conditions"]
    L += ["", "| condition | value | limit | met |", "|---|---|---|---|",
          f"| C(E) <= C(T) + 2 | {cd['c1']['lhs']} | {cd['c1']['limit']} | {'yes' if cd['c1']['pass'] else 'no'} |",
          f"| S(E) - G(E) <= S(T) - G(T) + 5 | {cd['c2']['lhs']} | {cd['c2']['limit']} | {'yes' if cd['c2']['pass'] else 'no'} |",
          f"| S(E) <= 0.5% of boxes right at baseline | {cd['c3']['lhs']} | {cd['c3']['limit']:.2f} | {'yes' if cd['c3']['pass'] else 'no'} |",
          "",
          f"Decision: {dec['decision']}. T arm warning (C(T) > 2 or S(T) above 0.5 percent of right-at-baseline): {'fired' if dec['t_arm_warning'] else 'did not fire'}.",
          "", "Counts by height band and by set (reported only, not gates):", "",
          "| group | boxes | right at baseline | S(T) | G(T) | C(T) | S(E) | G(E) | C(E) |", "|---|---|---|---|---|---|---|---|---|"]
    for title, d in (("band", rb["by_band"]), ("set", rb["by_set"])):
        for k, v in d.items():
            L.append(f"| {title} {k} | {v['boxes']} | {v['right_at_baseline']} | {v['T']['S']} | {v['T']['G']} | {v['T']['C']} | {v['E']['S']} | {v['E']['G']} | {v['E']['C']} |")
    L += ["", "First hits that are new-set rows (reported only):", "",
          "| arm | boxes with a new-set first hit | of those, agreed name | mean cosine, new-row hits | 90th percentile | mean cosine, shipped-row hits | 90th percentile |",
          "|---|---|---|---|---|---|---|"]
    for a in ("T", "E"):
        f = fh[a]
        g = lambda s, k: "-" if f[s][k] is None else f"{f[s][k]:.4f}"  # noqa: E731
        L.append(f"| {a} | {f['new_row_first_hits']} | {f['new_row_first_hits_with_agreed_name']} | {g('cos_new_row_first_hits', 'mean')} | "
                 f"{g('cos_new_row_first_hits', 'p90')} | {g('cos_shipped_row_first_hits', 'mean')} | {g('cos_shipped_row_first_hits', 'p90')} |")
    L.append("")
    return L


def fmt(w):
    return "-" if w["pct"] is None else f"{w['pct']:.1f} [{w['lo']:.1f},{w['hi']:.1f}]"


def table(res, group, metric, rowsets=("T", "E")):
    head = "| rows | " + " | ".join(CELLS) + " |\n|---|" + "---|" * len(CELLS)
    lines = [head]
    for r in rowsets:
        lines.append(f"| {r} | " + " | ".join(fmt(res["cells"][r][c][group][metric]) for c in CELLS) + " |")
    return "\n".join(lines)


def cmd_report():
    res = json.load(open(os.path.join(OUT, "results.json"), encoding="utf-8"))
    chk = json.load(open(os.path.join(WORK, "row_check.json"), encoding="utf-8"))
    d, cnt, c = res["decision"], res["counts"], res["cells"]
    t_p, e_p = c["T"][PRIMARY]["new"]["top1_id"], c["E"][PRIMARY]["new"]["top1_id"]
    chosen = d["chosen"]
    why = ("E rows beat T rows by more than 2.0 points at 200 px and are not lower at large full"
           if chosen == "E" else
           "E rows did not both beat T rows by more than 2.0 points at 200 px and stay at least level at large full")
    rb_path = os.path.join(OUT, "realboxes.json")
    rb = json.load(open(rb_path, encoding="utf-8")) if os.path.exists(rb_path) else None
    ctl_t, ctl_e = c["T"][PRIMARY]["control"]["first_hit_is_new_row"], c["E"][PRIMARY]["control"]["first_hit_is_new_row"]
    ctl_pair = res["paired"]["control"][PRIMARY]
    out = ["# New-set rows study, 2026-11-17", ""]
    if rb is None:
        out += [f"Final choice: not decided yet. The synthetic rule alone picks {chosen} rows, but Amendment 1 of `PREREGISTRATION.md` also requires the real-box check "
                "(`realboxes` step), which has not been run.", ""]
    else:
        dec, ar, cond = rb["decision"], rb["arms"], rb["decision"]["conditions"]
        final_text = {"E": "E rows (made by model E)", "T": "T rows (made by the v2.0 model, the model that made the shipped catalog)",
                      "owner_decision": "neither kind of row, until the owner decides (the T arm itself broke a limit of the amendment)"}[dec["decision"]]
        out += [f"Final choice under the amended rule: {final_text}.", "",
                f"The synthetic rule alone picked {chosen} rows: at 200 px on the new cards E rows beat T rows by {d['E_minus_T_primary_points']:+.1f} points "
                f"(T {fmt(t_p)} percent, E {fmt(e_p)} percent, n={cnt['new']}). "
                f"The real-box check of Amendment 1 uses {rb['counts']['boxes']} labelled English boxes from real photos. "
                f"Against searching the shipped catalog alone, T rows steal {ar['T']['S']} boxes and gain {ar['T']['G']}, and add {ar['T']['C']} confident errors; "
                f"E rows steal {ar['E']['S']}, gain {ar['E']['G']}, and add {ar['E']['C']} confident errors "
                f"(S is boxes right at baseline and wrong with the arm's rows, G is the reverse, C is confident wrong answers under the arm minus the same count at baseline). "
                f"The three conditions: C(E) <= C(T) + 2 is {'met' if cond['c1']['pass'] else 'not met'} ({ar['E']['C']} against {cond['c1']['limit']}); "
                f"S(E) - G(E) <= S(T) - G(T) + 5 is {'met' if cond['c2']['pass'] else 'not met'} ({cond['c2']['lhs']} against {cond['c2']['limit']}); "
                f"S(E) at most 0.5 percent of the {rb['counts']['right_at_baseline']} boxes right at baseline is {'met' if cond['c3']['pass'] else 'not met'} "
                f"({ar['E']['S']} against {cond['c3']['limit']:.2f}). "
                + (f"The T arm warning (C(T) > 2 or S(T) above 0.5 percent) {'fired' if dec['t_arm_warning'] else 'did not fire'}. ") +
                "Details are in the real-box section below and in `realboxes.json`.", ""]
    out += ["## Synthetic result (preregistered rule, before the amendment)", "",
            f"The synthetic rule picks {chosen} rows ({why}). In the primary cell (top-1 by exact card id at 200 px, new cards, n={cnt['new']}) "
            f"T rows score {fmt(t_p)} percent and E rows score {fmt(e_p)} percent; E minus T is {d['E_minus_T_primary_points']:+.1f} points. "
            f"At large full the difference is {d['E_minus_T_large_full_points']:+.1f} points, but large full is an uninformative guard: "
            "its query is the row's own picture, so for E rows (same model, same picture) it cannot fail.",
            "",
            f"The control cards show a cost. With E rows, {ctl_e['k']} of {ctl_e['n']} control reads at 200 px land on a new-set row as their first hit; with T rows {ctl_t['k']} do. "
            f"The paired E minus T difference for control cards at 200 px is {ctl_pair['diff']:+.1f} points with an interval of [{ctl_pair['lo']:+.1f},{ctl_pair['hi']:+.1f}], which reaches zero, "
            "so this is not a proven loss; it is why Amendment 1 adds a real-photo condition.",
            "",
           "T rows are the rows in the live `catalog-extra.bin` (made by the v2.0 model, `77538aad`). E rows are made by model E (`0dba1302`) from the same clean picture, "
           "rounded to int8 the way the nightly job does it. The rule is in `PREREGISTRATION.md` and was not changed. Queries are synthetic reads embedded with model E. "
           "All cells are percent with a 95 percent Wilson interval in brackets.",
           "",
           f"New cards are the {cnt['new']} physical cards of sets `me06` ({cnt['new_by_set']['me06']}) and `30th` ({cnt['new_by_set']['30th']}). "
           f"Control cards are the {cnt['control']} cards of `me05` in the shipped catalog, which model E was trained on. "
           f"The search set is the {cnt['search_rows_shipped_nondigital']} shipped non-digital rows plus the new cards' rows. "
           f"A new card is name-novel when no shipped non-digital card has the same name after `norm_name` ({cnt['new_name_novel']} cards); the other {cnt['new_name_seen']} are name-seen.",
           "",
           "Cell names: h90 to h200 are reads of the card picture shrunk to that height in pixels; large_full is the clean picture, large_400 and large_300 are shrunk to that height.",
           ""]
    for title, g, m in (("New cards, top-1 by exact id", "new", "top1_id"), ("New cards, top-5 by exact id", "new", "top5_id"),
                        ("New cards, top-1 by name (variant words such as ex and v ignored, as `base_name`)", "new", "top1_name"),
                        ("New cards, name-novel only, top-1 by exact id", "new_name_novel", "top1_id"),
                        ("New cards, name-seen only, top-1 by exact id", "new_name_seen", "top1_id"),
                        ("Control cards (me05), top-1 by exact id", "control", "top1_id"),
                        ("Control cards (me05), top-5 by exact id", "control", "top5_id"),
                        ("Control cards (me05), top-1 by name", "control", "top1_name")):
        out += [f"## {title}", "", table(res, g, m), ""]
    out += ["## Paired difference, E rows minus T rows, top-1 by exact id",
            "",
            "Points, with a 95 percent paired bootstrap interval over cards (2,000 resamples, seed 0).",
            "",
            "| group | " + " | ".join(CELLS) + " |", "|---|" + "---|" * len(CELLS)]
    for g in ("new", "control"):
        out.append(f"| {g} | " + " | ".join(f"{res['paired'][g][x]['diff']:+.1f} [{res['paired'][g][x]['lo']:+.1f},{res['paired'][g][x]['hi']:+.1f}]" for x in CELLS) + " |")
    v, te = chk["v2_vs_live"], chk["T_vs_E"]
    out += ["",
            "## Row checks",
            "",
            f"Re-embedding the {chk['n']} clean pictures with the v2.0 model gives vectors with mean cosine {v['mean']:.4f} and minimum cosine {v['min']:.4f} "
            f"(card {v['min_id']}) to the live file's rows, so the live rows are v2.0-model rows up to int8 rounding. "
            f"The mean cosine between each card's T row and E row is {te['mean']:.4f} (minimum {te['min']:.4f}).",
            ""]
    if rb is not None:
        out += real_section(rb)
    out += ["## Limits",
            "",
            "The synthetic reads are a clean catalog picture shrunk and JPEG-compressed by the existing generators; they are a floor check, not a claim about field accuracy. "
            "The real-box check uses labelled English boxes from earlier harvests; it scores names, not card ids, so a new-set row with the same name as an older card counts as right. "
            "It was not built to measure how well new-set cards are read from photos, and no labelled real photos of the two new sets are known to exist in the non-sealed data. "
            "`scripts/fusion_rule.py` is a partial port of the extension's rule, so the confident-error counts are an offline approximation. "
            "The control cards were in model E's training data and the new cards were not, so the two groups also differ in how the model was fitted; the gap decides nothing. "
            "The new cards' pictures come from TCGdex, the same kind of source as the shipped pictures, but any difference in picture style between sets is part of what is measured. "
            "No threshold in `lib/fusion.js` was changed."]
    open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("README written", os.path.join(OUT, "README.md"))


if __name__ == "__main__":
    below_normal()
    os.makedirs(WORK, exist_ok=True)
    {"fetch": cmd_fetch, "rows": cmd_rows, "reads": cmd_reads, "score": cmd_score, "realboxes": cmd_realboxes, "report": cmd_report}[sys.argv[1]]()
