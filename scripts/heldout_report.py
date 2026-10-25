"""Assemble the fastEho tables from train/out/ho/synth_*.json and train/out/ho/xs/boxes.json. Prints markdown."""
import json
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "train"))
from gpu_data import norm_name  # noqa: E402
from run_real_photos import load_names  # noqa: E402
from score_real_labels import base_name  # noqa: E402

HO = os.path.join(ROOT, "train", "out", "ho")
LABELS = ["shipped", "F", "E-ho", "F-ho"]
SYN_H = ["90", "120", "150", "200"]
LARGE = ["full", "400", "300"]
RNG = np.random.default_rng(0)


def wilson(k, n):
    if n == 0:
        return "-"
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{100 * p:.1f} [{100 * (c - h):.1f},{100 * (c + h):.1f}]"


def paired(a, b, n_boot=4000):
    """Mean difference a-b (pts) and 95% paired bootstrap CI over cards. a, b bool arrays."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    idx = RNG.integers(0, len(d), (n_boot, len(d)))
    m = d[idx].mean(1)
    return 100 * d.mean(), 100 * np.percentile(m, 2.5), 100 * np.percentile(m, 97.5)


def fmt_p(t):
    return f"{t[0]:+.1f} [{t[1]:+.1f},{t[2]:+.1f}]"


def flags(rs, truth, novel_mask=None):
    """Per-card boolean arrays for a list of reads."""
    t1 = np.array([r["ids"][0] == t for r, t in zip(rs, truth)])
    t5 = np.array([t in r["ids"] for r, t in zip(rs, truth)])
    acc = np.array([r["accepted"] for r in rs])
    right = np.array([r["accepted"] and r["chosen"] == t for r, t in zip(rs, truth)])
    sure = np.array([r["label"] == "sure" for r in rs])
    sure_w = np.array([r["label"] == "sure" and r["chosen"] != t for r, t in zip(rs, truth)])
    hub = Counter(r["ids"][0] for r in rs).most_common(1)[0][1] / len(rs)
    return {"t1": t1, "t5": t5, "acc": acc, "acc_w": acc & ~right, "sure": sure, "sure_w": sure_w, "hub": hub}


def synth_tables(labels, truth, novel):
    data = {l: json.load(open(os.path.join(HO, f"synth_{l}.json"), encoding="utf-8")) for l in labels}
    out, F = [], {}
    for l in labels:
        F[l] = {("s", h): flags(data[l]["synth"][h], truth) for h in SYN_H}
        F[l].update({("l", s): flags(data[l]["large"][s], truth) for s in LARGE})
    n = len(truth)
    for title, mask in (("ALL H cards", np.ones(n, bool)), ("name-novel H cards (name absent from every non-H card)", novel),
                        ("name-seen H cards (same name exists outside H)", ~novel)):
        out.append(f"\n#### Synthetic H eval, {title} (n={int(mask.sum())}); top-1 id % [95% Wilson]")
        out.append("| model | " + " | ".join(f"h{h}" for h in SYN_H) + " | " + " | ".join(f"large {s}" for s in LARGE) + " |")
        out.append("|---|" + "---|" * (len(SYN_H) + len(LARGE)))
        for l in labels:
            cells = [wilson(int(F[l][k]["t1"][mask].sum()), int(mask.sum())) for k in [("s", h) for h in SYN_H] + [("l", s) for s in LARGE]]
            out.append(f"| {l} | " + " | ".join(cells) + " |")
    out.append(f"\n#### Synthetic H eval, ALL H (n={n}); top-5 id % [95% Wilson]")
    out.append("| model | " + " | ".join(f"h{h}" for h in SYN_H) + " |")
    out.append("|---|" + "---|" * len(SYN_H))
    for l in labels:
        out.append(f"| {l} | " + " | ".join(wilson(int(F[l][("s", h)]["t5"].sum()), n) for h in SYN_H) + " |")
    out.append("\n#### Accepted-wrong / sure-wrong (id level) and hub share (top-1 modal id share), ALL H")
    out.append("| model | cell | accepted | acc-wrong | sure | sure-wrong | hub1 |")
    out.append("|---|---|---|---|---|---|---|")
    for l in labels:
        for k in [("s", "120"), ("s", "150"), ("s", "200"), ("l", "full"), ("l", "400"), ("l", "300")]:
            f = F[l][k]
            out.append(f"| {l} | {k[0]}{k[1]} | {int(f['acc'].sum())} | {int(f['acc_w'].sum())} | {int(f['sure'].sum())} | "
                       f"{int(f['sure_w'].sum())} | {f['hub']:.3f} |")
    if "E-ho" in F and "F-ho" in F:
        out.append("\n#### Paired E-ho minus F-ho, top-1 / top-5 id pts [95% paired bootstrap CI over cards]")
        out.append("| cell | subset | top1 diff | top5 diff |")
        out.append("|---|---|---|---|")
        cells = [("s", h) for h in SYN_H] + [("l", s) for s in LARGE]
        for k in cells:
            for nm_, mask in (("all", np.ones(n, bool)), ("novel", novel)):
                a, b = F["E-ho"][k], F["F-ho"][k]
                out.append(f"| {k[0]}{k[1]} | {nm_} | {fmt_p(paired(a['t1'][mask], b['t1'][mask]))} | {fmt_p(paired(a['t5'][mask], b['t5'][mask]))} |")
        # pooled over the four synthetic heights (cards resampled jointly)
        for nm_, mask in (("all", np.ones(n, bool)), ("novel", novel)):
            a1 = np.mean([F["E-ho"][("s", h)]["t1"][mask] for h in SYN_H], 0)
            b1 = np.mean([F["F-ho"][("s", h)]["t1"][mask] for h in SYN_H], 0)
            a5 = np.mean([F["E-ho"][("s", h)]["t5"][mask] for h in SYN_H], 0)
            b5 = np.mean([F["F-ho"][("s", h)]["t5"][mask] for h in SYN_H], 0)
            out.append(f"| pooled synth 90-200 | {nm_} | {fmt_p(paired(a1, b1))} | {fmt_p(paired(a5, b5))} |")
        sw = {m: sum(int(F[m][k]["sure_w"].sum()) for k in [("s", "200"), ("l", "full"), ("l", "400"), ("l", "300")]) for m in ("E-ho", "F-ho")}
        out.append(f"\nSure-wrong on H at >=200 px (synthetic h200 + large full/400/300, 4*{n} reads): E-ho {sw['E-ho']}, F-ho {sw['F-ho']}")
    return "\n".join(out), F


def xs_tables(labels, hids):
    boxes = json.load(open(os.path.join(HO, "xs", "boxes.json"), encoding="utf-8"))
    nm = load_names()
    nrm = lambda n: base_name(n or "")  # noqa: E731
    out = []
    cells = {"native>=200": lambda b: b["cell"] == "native" and b["native_h"] >= 200,
             "native<200": lambda b: b["cell"] == "native" and b["native_h"] < 200,
             "250": lambda b: b["cell"] == 250, "200": lambda b: b["cell"] == 200,
             "150": lambda b: b["cell"] == 150, "120": lambda b: b["cell"] == 120}
    store = {}
    for sub, pred in (("H (true id in H)", lambda b: b["gt_id"] in hids), ("non-H", lambda b: b["gt_id"] not in hids)):
        out.append(f"\n#### Cross-source, {sub}; cell: n | top1-name/top5-name/top1-id | accepted-wrong | sure-wrong")
        out.append("| cell | n | " + " | ".join(labels) + " |")
        out.append("|---|---|" + "---|" * len(labels))
        for cname, cp in cells.items():
            bs = [b for b in boxes if pred(b) and cp(b)]
            if not bs:
                continue
            row = []
            for l in labels:
                g = [nrm(b["gt_name"]) for b in bs]
                rd = [b["reads"][l] for b in bs]
                t1 = sum(nrm(r["names"][0]) == x for r, x in zip(rd, g))
                t5 = sum(x in [nrm(n) for n in r["names"]] for r, x in zip(rd, g))
                ti = sum(r["ids"][0] == b["gt_id"] for r, b in zip(rd, bs))
                aw = sum(r["accepted"] and nrm(r["chosen_name"]) != x for r, x in zip(rd, g))
                sw = sum(r["label"] == "sure" and nrm(r["chosen_name"]) != x for r, x in zip(rd, g))
                row.append(f"{t1}/{t5}/{ti} | {aw} | {sw}")
                store[(sub, cname, l)] = (len(bs), t1, t5, ti, aw, sw)
            out.append(f"| {cname} | {len(bs)} | " + " | ".join(row) + " |")
    # paired E-ho vs F-ho, non-H, pooled over all cells (name-level top1)
    pairs = []
    for b in boxes:
        if b["gt_id"] in hids:
            continue
        g = nrm(b["gt_name"])
        pairs.append((nrm(b["reads"]["E-ho"]["names"][0]) == g, nrm(b["reads"]["F-ho"]["names"][0]) == g,
                      g in [nrm(n) for n in b["reads"]["E-ho"]["names"]], g in [nrm(n) for n in b["reads"]["F-ho"]["names"]],
                      b["cell"] == "native"))
    p = np.array(pairs)
    for nm_, m in (("all cells", np.ones(len(p), bool)), ("native only", p[:, 4])):
        out.append(f"\nCross-source non-H, {nm_} (n={int(m.sum())}), E-ho minus F-ho: top1-name {fmt_p(paired(p[m, 0], p[m, 1]))}, "
                   f"top5-name {fmt_p(paired(p[m, 2], p[m, 3]))}")
    return "\n".join(out), store


if __name__ == "__main__":
    nm = load_names()
    hids = json.load(open(os.path.join(HO, "heldout.json"), encoding="utf-8"))["ids"]
    hs = set(hids)
    other = {norm_name(n) for c, n in nm.items() if c not in hs}
    novel = np.array([norm_name(nm[c]) not in other for c in hids])
    which = sys.argv[1:] or ["synth", "xs"]
    if "synth" in which:
        txt, _ = synth_tables(LABELS, hids, novel)
        print(txt)
    if "xs" in which:
        txt, _ = xs_tables(LABELS, hs)
        print(txt)
