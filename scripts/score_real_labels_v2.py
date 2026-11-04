"""Cards-only variant of score_real_labels: drops photos listed in EXCLUDE.json (numbering unchanged)."""
import json
import os
import sys
from collections import Counter

import score_real_labels as sl

USER_SAVED = os.path.join(sl.ROOT, "data", "fb-eval-pack", "user-saved")
EXCLUDE = os.path.join(USER_SAVED, "EXCLUDE.json")
OUT_V2 = os.path.join(sl.REPORTS, "real-labelled", "scores-cards-only.json")
sl.HALF_A = {6, 7}
sl.HALF_B = {3, 4, 5}
SELECTION_PHOTO = 3  # looked at during model selection


def excluded_numbers():
    """Photo numbers (sorted listing of user-saved images, 1-based) of excluded files."""
    with open(EXCLUDE, encoding="utf-8") as fh:
        names = set(json.load(fh)["exclude_photos"])
    imgs = sorted(f for f in os.listdir(USER_SAVED) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    return {i + 1 for i, f in enumerate(imgs) if f in names}


def keep(d, drop):
    return {k: v for k, v in d.items() if k[0] not in drop}


def run(labels, models, drop):
    labels = keep(labels, drop)
    return {m: sl.score_model(keep(b, drop), labels) for m, b in models.items()}, labels


def main():
    drop = excluded_numbers()
    labels = sl.parse_labels()
    models = {m: sl.load_model_results(os.path.join(sl.REPORTS, m, "results.json")) for m in sl.model_dirs()}
    scored, kept = run(labels, models, drop)
    results = {"excluded_photo_numbers": sorted(drop),
               "label_counts": {"n": len(kept), "kind": dict(Counter(l["kind"] for l in kept.values())),
                                "card_conf": dict(Counter(l["conf"] for l in kept.values() if l["kind"] == "card"))},
               "models": scored}
    no3, _ = run(labels, models, drop | {SELECTION_PHOTO})
    results["excluding_photo_3"] = {m: {s: {"overall": r[s]["overall"]} for s in ("ALL_READABLE", "CLEAR")}
                                    for m, r in no3.items()}
    os.makedirs(os.path.dirname(OUT_V2), exist_ok=True)
    with open(OUT_V2, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    print("excluded photo numbers", sorted(drop), results["label_counts"])
    results["disagreement_shipped_vs_B_raw"] = "n/a"
    sl.print_tables(results)
    print("\n== Excluding photo 3 as well (top1B/top5B/top1S %, n)")
    for s in ("ALL_READABLE", "CLEAR"):
        for m, r in results["excluding_photo_3"].items():
            t = r[s]["overall"]
            print(f"{s:13s} {m:28s} {sl.pct(t['top1_base'])}/{sl.pct(t['top5_base'])}/{sl.pct(t['top1_strict'])} n{t['n']}")
    print("saved", OUT_V2)


if __name__ == "__main__":
    sys.exit(main())
