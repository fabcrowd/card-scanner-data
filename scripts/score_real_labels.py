"""Score real-photo model results against the blind reviewer labels (labels-claude.txt).

Box id = (photo number by sorted listing of the user-saved folder, 1-based order of the box within that photo in
results.json). Name match: BASE (variant suffix tokens stripped both sides) is primary, STRICT is normalise-only.
"""
import glob
import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = os.path.join(ROOT, "data", "fb-eval-pack", "user-saved", "labels-claude.txt")
REPORTS = os.path.join(ROOT, "data", "reports")
OUT = os.path.join(REPORTS, "real-labelled", "scores.json")
VARIANT_TOKENS = {"ex", "v", "vmax", "vstar", "vunion", "gx", "star", "delta", "δ"}
HEIGHT_BUCKETS = [("<100", 0, 100), ("100-149", 100, 150), (">=150", 150, 10**9)]
HALF_A = {1, 2, 6, 7}
HALF_B = {3, 4, 5}
PRECISION_FLOOR = 0.80
LINE_RE = re.compile(r"^p(\d+)-#(\d+)\s*\|\s*(\w+)\s*\|\s*(.*?)\s*\|\s*(\d)\s*$")


def normalise(name):
    s = name.lower().replace("’", "'")
    s = re.sub(r"'s\b", "", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def base_name(name):
    return " ".join(t for t in normalise(name).split() if t not in VARIANT_TOKENS)


def parse_labels(path=LABELS):
    """Return {(photo, box): {kind, name, conf}}; '~' prefix on names is stripped (recognised from art only)."""
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            m = LINE_RE.match(line.strip())
            if not m:
                raise ValueError(f"bad label line: {line!r}")
            p, b, kind, name, conf = m.groups()
            out[(int(p), int(b))] = {"kind": kind, "name": name.lstrip("~").strip(), "conf": int(conf),
                                     "tilde": name.startswith("~")}
    return out


def number_boxes(boxes):
    """Map results.json boxes to (photo number, order) with photo number = sorted listing of photo names."""
    photos = sorted({b["photo"] for b in boxes})
    num = {p: i + 1 for i, p in enumerate(photos)}
    seen = Counter()
    out = {}
    for b in boxes:
        seen[b["photo"]] += 1
        out[(num[b["photo"]], seen[b["photo"]])] = b
    return out


def match(pred, truth, base):
    f = base_name if base else normalise
    return f(pred) == f(truth)


def box_eval(box, label):
    names = [t["name"] for t in box["top5"]]
    return {"top1_base": bool(names) and match(names[0], label["name"], True),
            "top5_base": any(match(n, label["name"], True) for n in names),
            "top1_strict": bool(names) and match(names[0], label["name"], False)}


def bucket(h):
    return next(lab for lab, lo, hi in HEIGHT_BUCKETS if lo <= h < hi)


def subsets(labels):
    return {"ALL_READABLE": lambda l: l["kind"] == "card" and l["name"] != "?",
            "CLEAR": lambda l: l["kind"] == "card" and l["conf"] == 2}


def acc_table(rows):
    n = len(rows)
    d = max(n, 1)
    return {"n": n, "top1_base": sum(r["e"]["top1_base"] for r in rows) / d,
            "top5_base": sum(r["e"]["top5_base"] for r in rows) / d,
            "top1_strict": sum(r["e"]["top1_strict"] for r in rows) / d}


def accepted_stats(rows):
    acc = [r for r in rows if r["box"]["accepted"]]
    return {"n": len(rows), "accepted": len(acc), "coverage": len(acc) / max(len(rows), 1),
            "precision": (sum(r["e"]["top1_base"] for r in acc) / len(acc)) if acc else None}


def cos1(r):
    return r["box"]["top5"][0]["cos"] if r["box"]["top5"] else 0.0


def fit_threshold(rows):
    """Lowest top-1 cosine threshold with max coverage subject to precision >= floor (None if unattainable)."""
    best = None
    for t in sorted({cos1(r) for r in rows}):
        sel = [r for r in rows if cos1(r) >= t]
        prec = sum(r["e"]["top1_base"] for r in sel) / len(sel)
        if prec >= PRECISION_FLOOR and (best is None or len(sel) > best[1]):
            best = (t, len(sel))
    return best[0] if best else None


def apply_threshold(rows, t):
    sel = [r for r in rows if cos1(r) >= t]
    return {"threshold": t, "n": len(rows), "coverage": len(sel) / max(len(rows), 1),
            "precision": (sum(r["e"]["top1_base"] for r in sel) / len(sel)) if sel else None}


def refit(rows):
    out = {}
    for tag, train_set in (("fit_A_apply_B", HALF_A), ("fit_B_apply_A", HALF_B)):
        train = [r for r in rows if r["photo"] in train_set]
        test = [r for r in rows if r["photo"] not in train_set]
        t = fit_threshold(train)
        out[tag] = {"train_n": len(train), "unattainable": True} if t is None else \
            {"train": apply_threshold(train, t), "test": apply_threshold(test, t)}
    return out


def hubs(rows):
    wrong = [r["box"]["top5"][0] for r in rows if not r["e"]["top1_base"] and r["box"]["top5"]]
    cnt = Counter((t["id"], t["name"]) for t in wrong)
    n = len(wrong)
    common = cnt.most_common(5)
    return {"n_wrong": n, "top1_share": common[0][1] / n if n else None,
            "top5_share": sum(c for _, c in common) / n if n else None,
            "top5": [{"id": i, "name": nm, "count": c} for (i, nm), c in common]}


def load_model_results(path):
    with open(path, encoding="utf-8") as fh:
        return number_boxes(json.load(fh)["boxes"])


def model_dirs():
    names = ["real-baseline-shipped"] + sorted(os.path.basename(p) for p in glob.glob(os.path.join(REPORTS, "real-student-*")))
    return [n for n in names if os.path.exists(os.path.join(REPORTS, n, "results.json"))]


def score_model(boxes, labels):
    assert set(boxes) == set(labels), "results do not map 1:1 to labels"
    res = {}
    for sname, pred in subsets(labels).items():
        rows = [{"photo": k[0], "key": k, "box": boxes[k], "label": labels[k], "e": box_eval(boxes[k], labels[k])}
                for k in sorted(labels) if pred(labels[k])]
        res[sname] = {"overall": acc_table(rows),
                      "buckets": {b: acc_table([r for r in rows if bucket(r["box"]["height"]) == b])
                                  for b, _, _ in HEIGHT_BUCKETS},
                      "fusion_accepted": accepted_stats(rows)}
        if sname == "ALL_READABLE":
            res[sname]["refit"] = refit(rows)
            res[sname]["wrong_hubs"] = hubs(rows)
    res["false_accepts_noncard_pile"] = {
        "boxes": sum(l["kind"] in ("noncard", "pile") for l in labels.values()),
        "accepted": sum(boxes[k]["accepted"] for k, l in labels.items() if l["kind"] in ("noncard", "pile"))}
    return res


def correct_map(boxes, labels):
    return {k: box_eval(boxes[k], labels[k])["top1_base"] for k in labels
            if labels[k]["kind"] == "card" and labels[k]["name"] != "?"}


def disagreements(a, b, labels, limit=15):
    """Readable boxes where right/wrong differs; ranked by top-1 cosine gap between the two models."""
    ca, cb = correct_map(a, labels), correct_map(b, labels)
    diff = [k for k in ca if ca[k] != cb[k]]
    diff.sort(key=lambda k: -abs(cos1({"box": a[k]}) - cos1({"box": b[k]})))
    return {"total_disagreements": len(diff), "ids": [f"p{k[0]}-#{k[1]}" for k in diff[:limit]]}


def pct(x):
    return "  -  " if x is None else f"{100 * x:5.1f}"


def print_tables(results):
    for sname in ("ALL_READABLE", "CLEAR"):
        print(f"\n== {sname}  (top1B/top5B/top1S %, n)")
        print(f"{'model':28s} {'all':>20s} " + " ".join(f"{b:>20s}" for b, _, _ in HEIGHT_BUCKETS))
        for m, r in results["models"].items():
            def cell(t):
                return f"{pct(t['top1_base'])}/{pct(t['top5_base'])}/{pct(t['top1_strict'])} n{t['n']:<3d}"
            s = r[sname]
            print(f"{m:28s} {cell(s['overall'])} " + " ".join(cell(s['buckets'][b]) for b, _, _ in HEIGHT_BUCKETS))
    print("\n== Shipped fusion accepted set (ALL_READABLE): accepted/n, coverage, precision")
    for m, r in results["models"].items():
        a = r["ALL_READABLE"]["fusion_accepted"]
        print(f"{m:28s} {a['accepted']:3d}/{a['n']:<3d} cov {pct(a['coverage'])} prec {pct(a['precision'])}")
    print(f"\n== Threshold refit (prec>={PRECISION_FLOOR}); train t -> test cov/prec")
    for m, r in results["models"].items():
        for tag, v in r["ALL_READABLE"]["refit"].items():
            if v.get("unattainable"):
                print(f"{m:28s} {tag}: unattainable on train")
            else:
                print(f"{m:28s} {tag}: t={v['train']['threshold']:.4f} train cov {pct(v['train']['coverage'])} "
                      f"prec {pct(v['train']['precision'])} | test cov {pct(v['test']['coverage'])} "
                      f"prec {pct(v['test']['precision'])}")
    print("\n== Wrong top-1 hubs (share of wrong)")
    for m, r in results["models"].items():
        h = r["ALL_READABLE"]["wrong_hubs"]
        print(f"{m:28s} wrong {h['n_wrong']} top1 {pct(h['top1_share'])} top5 {pct(h['top5_share'])} "
              + ", ".join(f"{x['name']}x{x['count']}" for x in h["top5"]))
    print("\n== False accepts on noncard/pile")
    for m, r in results["models"].items():
        f = r["false_accepts_noncard_pile"]
        print(f"{m:28s} {f['accepted']}/{f['boxes']}")
    print("\n== Disagreement shipped vs B-raw:", results["disagreement_shipped_vs_B_raw"])


def main():
    labels = parse_labels()
    models = {m: load_model_results(os.path.join(REPORTS, m, "results.json")) for m in model_dirs()}
    results = {"label_counts": {"n": len(labels),
                                "kind": dict(Counter(l["kind"] for l in labels.values())),
                                "card_conf": dict(Counter(l["conf"] for l in labels.values() if l["kind"] == "card"))},
               "models": {m: score_model(b, labels) for m, b in models.items()}}
    if "real-baseline-shipped" in models and "real-student-B-raw" in models:
        results["disagreement_shipped_vs_B_raw"] = disagreements(models["real-baseline-shipped"],
                                                                 models["real-student-B-raw"], labels)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    print(results["label_counts"])
    print_tables(results)
    print("saved", OUT)


if __name__ == "__main__":
    sys.exit(main())
