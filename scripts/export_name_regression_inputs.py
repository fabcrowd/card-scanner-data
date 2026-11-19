"""One-off: writes the inputs of tools/shared-data/scripts/build_regression.py's name mode for the 3,363 agreed-name real boxes.

    labels.json         [{"row": <index into emb.npy>, "name": "<agreed label>"}], in the order of cmd_realboxes
                        (scripts/newsets_eval.py): classes.json boxes of class E whose (set, box_id) has an agreed label.
    catalog-names.json  {shipped catalog id: printed card name}, from load_names() (scripts/run_real_photos.py).

The selection is cross-checked against realboxes.json (same count, same (set, box_id) order, same agreed names) and
against features.jsonl (the row's set and box_id). Nothing is filtered by any result. Run with python -I.

    python -I scripts/export_name_regression_inputs.py --out <folder>
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = r"C:\repos\projects\auto card lister\tools\shared-data"
sys.path.insert(0, TEMPLATE)

from pipeline.names import base_name  # noqa: E402

REPORTS = os.path.join(ROOT, "data", "reports")
CLASSES = os.path.join(REPORTS, "japanese-20261117", "classes.json")
REALBOXES = os.path.join(REPORTS, "newsets-20261117", "realboxes.json")
FEATURES = r"C:\repos\projects\auto card lister\.tmp\detect-qa\diag-20261113\features.jsonl"
CARD_DIR = r"C:\repos\projects\auto card lister\data\card-catalog"  # run_real_photos.CARD_DIR
SETS = ["obscure", "fresh-20261111", "check-20261112", "check-20261112c", "check-20261113"]  # japanese_study.SETS


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_names():
    """run_real_photos.load_names: meta.json then meta.full.json (the full one wins)."""
    names = {}
    for f in ("meta.json", "meta.full.json"):
        for m in read_json(os.path.join(CARD_DIR, f)):
            if m.get("name"):
                names[m["id"]] = m["name"]
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    agreed = {s: {int(r["box_id"]): r["name"] for r in read_json(os.path.join(REPORTS, s, "labels", "agreed.json"))} for s in SETS}
    cls = read_json(CLASSES)["boxes"]
    boxes = [b for b in cls if b["cls"] == "E" and b["set"] in agreed and b["box_id"] in agreed[b["set"]]]

    with open(FEATURES, encoding="utf-8") as f:
        features = [json.loads(line) for line in f if line.strip()]
    for b in boxes:
        feat = features[b["row"]]
        if (feat["set"], feat["box_id"]) != (b["set"], b["box_id"]):
            sys.exit(f"features.jsonl row {b['row']} is {feat['set']}/{feat['box_id']}, not {b['set']}/{b['box_id']}")

    real = read_json(REALBOXES)["boxes"]
    if [(b["set"], b["box_id"]) for b in boxes] != [(r["set"], r["box_id"]) for r in real]:
        sys.exit("the reproduced selection differs from realboxes.json in count or order")
    for b, r in zip(boxes, real):
        label = agreed[b["set"]][b["box_id"]]
        if r["agreed"] not in (label, base_name(label)):
            sys.exit(f"{b['set']}/{b['box_id']}: realboxes agreed {r['agreed']!r} is not the label {label!r} or its base name")

    labels = [{"row": b["row"], "name": agreed[b["set"]][b["box_id"]]} for b in boxes]
    names = load_names()
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False)
    with open(os.path.join(args.out, "catalog-names.json"), "w", encoding="utf-8") as f:
        json.dump(names, f, ensure_ascii=False)
    print(f"labels: {len(labels)} rows; catalog names: {len(names)} ids -> {args.out}")


if __name__ == "__main__":
    main()
