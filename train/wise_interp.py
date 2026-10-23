"""WiSE-FT: interpolate pretrained visual tower (theta0) and B raw final (theta1); int8 export + pilot eval at batch 1."""
import copy
import json
import os
import sys

import open_clip
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pilot_eval as V  # noqa: E402
import train_pilot as TP  # noqa: E402

ALPHAS = (0.25, 0.5, 0.75)
HS = (60, 90, 120, 150, 200)
GATE = {"clean": 0.970, "self": 0.983, "cosST": 0.98}


def interpolate(t0, t1, alpha):
    return {k: ((1 - alpha) * v.float() + alpha * t1[k].float()).to(v.dtype) if v.is_floating_point() else v
            for k, v in t0.items()}


def row(name, m):
    d = m["drift"]
    cells = []
    for h in HS:
        a = m["heights"][h]["all"] if h in m["heights"] else m["heights"][str(h)]["all"]
        cells.append(f"{a['top1']:.3f}/{a['accepted_wrong']}/{a['hub1_share']:.3f}")
    c = d.get("cos_ST_large_median")
    return f"{name:12s} " + " ".join(f"{x:>16s}" for x in cells) + f"  {d['clean_median_cos']:.4f} {d['self_match']:.4f} " + (f"{c:.4f}" if c is not None else "  n/a")


def main():
    device = "cuda"
    ctx = V.Context()
    model, teacher, _, _ = TP.build_models(device)
    t0 = {k: v.detach().cpu().clone() for k, v in model.visual.state_dict().items()}
    t1 = torch.load(os.path.join(V.OUT, "B", "raw-final.pt"), map_location="cpu")
    results = {}
    for alpha in ALPHAS:
        stem = os.path.join(V.OUT, "B", f"wise-{alpha}")
        vis = copy.deepcopy(model.visual).cpu()
        vis.load_state_dict(interpolate(t0, t1, alpha))
        exp = copy.deepcopy(model)
        exp.visual.load_state_dict(vis.state_dict())
        V.export_int8(exp, stem + "-fp32.onnx", stem + "-int8.onnx")
        rec = V.eval_int8(stem + "-int8.onnx", ctx)
        rec["drift"]["cos_ST_large_median"] = TP.drift_cos(ctx, teacher, vis.to(device).eval(), device)
        results[str(alpha)] = rec
        print(alpha, "done", flush=True)
    base = json.load(open(os.path.join(V.OUT, "baseline", "metrics.json"), encoding="utf-8"))
    bm = json.load(open(os.path.join(V.OUT, "B", "metrics.json"), encoding="utf-8"))["epochs"][-1]
    rows = {"baseline": base, "B_ema_ep4": {k: bm[k] for k in ("heights", "drift")}, "B_raw_a1.0": bm["raw_student"], **{f"a{a}": results[str(a)] for a in ALPHAS}}
    passes = {}
    for name, m in rows.items():
        d = m["drift"]
        passes[name] = bool(d["clean_median_cos"] >= GATE["clean"] and d["self_match"] >= GATE["self"] and d.get("cos_ST_large_median", 1) >= GATE["cosST"])
    print("cells = top1/accepted-wrong/hub1; then clean median, self-match, cos(S,T)")
    print(f"{'':12s} " + " ".join(f"{'H=' + str(h):>16s}" for h in HS))
    for name, m in rows.items():
        print(row(name, m), "PASS" if passes[name] else "fail")
    with open(os.path.join(V.OUT, "B", "wise-metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "gate": GATE, "passes_drift_gate": passes}, f, indent=2)


if __name__ == "__main__":
    main()
