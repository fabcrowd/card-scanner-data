"""fastC: train_fast with the audit fixes (gpu_data2) and optional real-photo crops; plus CPU-only post-eval and compare.

Modes:
  --name N [--real]      GPU training (8 epochs, EMA + raw checkpoints each epoch; EMA int8 eval1 of epochs 1..7 in the background)
  --post N               CPU only: final-epoch EMA and raw -> int8 -> eval1 + eval2, large-crop catalog check on the EMA
  --compare              CPU only: one table over shipped / B raw / fastB / fastC_cat / fastC_real, saved to fastC_compare.json
Real-crop CE variant: NO no_negative masks. A real crop is treated exactly like a catalog card: positives = every catalog
row whose normalised name equals the crop's label name, all other rows are negatives.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import open_clip
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gpu_data as G  # noqa: E402
import gpu_data2 as G2  # noqa: E402
import pilot_data as D  # noqa: E402
import pilot_eval as V  # noqa: E402
import train_fast as T  # noqa: E402
import train_pilot as P  # noqa: E402

ROOT = D.ROOT
sys.path.insert(0, os.path.join(ROOT, "scripts"))
OUT = V.OUT
COMPARE = os.path.join(OUT, "fastC_compare.json")
AUTOTUNE_CANDIDATES = (128, 192, 256)
REAL_FRAC = 0.25


def limit_ort():
    import onnxruntime as ort

    real = ort.InferenceSession

    def limited(path, *a, **k):
        so = ort.SessionOptions()
        so.intra_op_num_threads = T.EVAL_THREADS
        return real(path, sess_options=so, **k)

    ort.InferenceSession = limited
    torch.set_num_threads(T.EVAL_THREADS)


def train_step(student_c, teacher_c, large, deg, gid, catv, grp, w_ce, device):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        t = F.normalize(teacher_c(large).float(), dim=-1)
    n = len(gid)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        s = student_c(torch.cat([deg, large]))
    s = F.normalize(s.float(), dim=-1)
    sd, sl = s[:n], s[n:]
    l_dist = ((1 - (sd * t).sum(-1)) ** 2).mean()
    l_drift = ((1 - (sl * t).sum(-1)) ** 2).mean()
    l_ce = G2.ce_group(sd, gid, catv, grp, P.TAU) if w_ce > 0 else torch.zeros((), device=device)
    loss = l_dist + l_drift + w_ce * l_ce
    loss.backward()
    return torch.stack([loss.detach(), l_dist.detach(), l_drift.detach(), l_ce.detach()])


def autotune(student, student_c, teacher_c, cards, gid_cat, catv, grp, w_ce, device, rng):
    """Images/s per candidate (3 images per sample); step up only if >5% faster and peak VRAM < limit."""
    best, best_ips = None, 0.0
    for b in AUTOTUNE_CANDIDATES:
        opt = torch.optim.AdamW(P.param_groups(student), lr=0.0, weight_decay=0.0, fused=True)
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            for i in range(6):
                if i == 2:
                    torch.cuda.synchronize()
                    t0 = time.time()
                pos = torch.from_numpy(rng.choice(len(cards), b, replace=False)).to(device)
                large, deg, _, _ = G2.make_batch(cards[pos], rng, device)
                opt.zero_grad(set_to_none=True)
                train_step(student_c, teacher_c, large, deg, gid_cat[pos], catv, grp, w_ce, device)
                opt.step()
            torch.cuda.synchronize()
            ips, peak = 3 * b * 4 / (time.time() - t0), torch.cuda.max_memory_allocated() / 2**30
        except torch.OutOfMemoryError:
            ips, peak = 0.0, float("inf")
        del opt
        student.zero_grad(set_to_none=True)
        print(f"autotune B={b}: {ips:.0f} img/s, peak {peak:.2f} GiB", flush=True)
        if peak >= T.VRAM_LIMIT_GIB or (best is not None and ips < 1.05 * best_ips):
            break
        best, best_ips = b, ips
    torch.cuda.empty_cache()
    return best


def load_real(cat, name_of, eval_rows, k2g, device, args=None):
    # fastD: --extra-packs adds label files (HiBid rows carry absolute photo paths, so no path change is needed);
    # --exclude-eval hard-drops eval-duplicate / eval-SHA photos before caching; --real-cache keeps real320.pt untouched.
    extra = tuple(args.extra_packs) if args is not None and args.extra_packs else ()
    rows = G2.read_pack_rows(packs=G2.PACKS + extra)
    cache = G2.REAL_CACHE
    excl_rep = None
    if args is not None and args.exclude_eval:
        eval_dirs = [os.path.join(ROOT, "data", "hibid-pack", "eval"), os.path.join(ROOT, "data", "fb-eval-pack")]
        if args.all_physical:  # fastE: also the harvest eval splits data/harvest-20261022/*/eval
            hv = os.path.join(ROOT, "data", "harvest-20261022")
            eval_dirs += [os.path.join(hv, d, "eval") for d in sorted(os.listdir(hv)) if os.path.isdir(os.path.join(hv, d, "eval"))]
        rows, excl_rep, eval_sha = G2.exclude_eval_photos(rows, eval_dirs, os.path.join(G2.LABELS_DIR, "hibid-pack-train-eval-duplicates.json"))
        excl_rep["eval_dirs"] = [os.path.relpath(d, ROOT) for d in eval_dirs]
        assert not any(G2._sha256(r["photo"]) in eval_sha for r in rows), "eval SHA overlap after exclusion"
        excl_rep["sha_overlap_after_exclusion"] = 0
    if args is not None and args.real_cache:
        cache = args.real_cache
    crops = G2.load_real_crops(rows, cache)
    row_of = {c: i for i, c in enumerate(cat.ids)}
    names = G2.label_names(rows, name_of, row_of)
    ev_norm = {G.norm_name(name_of[r]) for r in eval_rows if name_of[r]}
    keep, gid, rep = G2.filter_real(names, [r["name"] for r in rows], ev_norm, k2g)
    if args is not None and args.all_physical:  # fastE: name-leakage filter OFF (training photos); report what it used to drop
        rep["would_have_dropped_by_name_leakage_filter"] = rep["dropped_leakage"]
        keep, gid, rep = G2.filter_real(names, [r["name"] for r in rows], set(), k2g)
        rep["would_have_dropped_by_name_leakage_filter"] = int(sum(1 for k, n, r in zip(range(len(names)), names, rows)
                                                                    if G.norm_name(n) in ev_norm or G.norm_name(r["name"]) in ev_norm))
    rep["exclusion"] = excl_rep
    rep["per_pack_leakage"] = {pk: {"candidates": sum(1 for r in rows if r["pack"] == pk),
                                    "used": sum(1 for r, k in zip(rows, keep) if k and r["pack"] == pk)}
                               for pk in dict.fromkeys(r["pack"] for r in rows)}
    rep["by_level_used"] = {lv: int(sum(1 for r, k in zip(rows, keep) if k and r["level"] == lv)) for lv in ("id", "name")}
    k = torch.from_numpy(np.flatnonzero(keep))
    return crops.index_select(0, k).to(device), torch.from_numpy(gid).to(device), rep


def load_all_cards(cat, cache_path):
    """fastE: every catalog id with an image except digital-only ids (synthetic eval cards and same-name cards included)."""
    from concurrent.futures import ThreadPoolExecutor

    have, _ = D.eval_rows(cat)
    ids = [cat.ids[r] for r in have if not D.is_digital(cat.ids[r])]
    if os.path.exists(cache_path):
        blob = torch.load(cache_path)
        assert blob["ids"] == ids, "cards cache does not match the all-physical id list; delete it"
        cards = blob["cards"]
    else:
        with ThreadPoolExecutor(12) as ex:
            cards = torch.stack(list(ex.map(G._load_card, [D.E.image_path(c) for c in ids])))
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({"ids": ids, "cards": cards}, cache_path)
    row_of = {c: i for i, c in enumerate(cat.ids)}
    rep = {"with_image": len(have), "excluded_digital": len(have) - len(ids), "trained_cards": len(ids),
           "note": "synthetic eval cards are in the training pool: synthetic metrics are in-train, drift-check only"}
    return cards, np.array([row_of[c] for c in ids], dtype=np.int64), rep


def train(args):
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    from pipeline.catalog import Catalog
    import score_synth_fusion as S

    cat = Catalog.load(os.path.join(ROOT, "assets", "catalog.bin.gz"), os.path.join(ROOT, "assets", "id_map.json"))
    nm = S.load_names()
    name_of = [nm.get(c) for c in cat.ids]
    _, eval_rows = D.eval_rows(cat)
    if args.all_physical:
        cards_cpu, rows_np, rep = load_all_cards(cat, args.cards_cache)
    else:
        cards_cpu, rows_np, rep = G.load_training_cards(cat, name_of, eval_rows, os.path.join(OUT, "split.json"))
    print("training cards:", json.dumps(rep), flush=True)
    cards = cards_cpu.to(device)
    catv = torch.from_numpy(cat.vectors).to(device)
    grp_cpu, k2g = G2.name_groups(name_of)
    assert torch.equal(grp_cpu, T.norm_groups(name_of)), "group ids differ from train_fast.norm_groups"
    grp = grp_cpu.to(device)
    gid_cat = grp[torch.from_numpy(rows_np).to(device)]
    real, gid_real, real_rep = None, None, None
    if args.real:
        real, gid_real, real_rep = load_real(cat, name_of, eval_rows, k2g, device, args)
        print("real crops:", json.dumps(real_rep), "tensor", tuple(real.shape), flush=True)

    model, teacher, student, ema = P.build_models(device)
    student_c, teacher_c, compile_status = T.try_compile(student, teacher, device)
    batch = int(args.batch) if args.batch != "auto" else autotune(student, student_c, teacher_c, cards, gid_cat, catv, grp, args.w_ce, device, rng)
    n_real = batch * 1 // 4 if args.real else 0
    n_cat = batch - n_real
    steps_per_epoch = len(cards) // batch
    total = args.epochs * steps_per_epoch
    print(f"name {args.name} real={args.real} w_ce={args.w_ce} batch={batch} (catalog {n_cat} + real {n_real}) "
          f"steps/epoch={steps_per_epoch} total={total} compile={compile_status}", flush=True)

    opt = torch.optim.AdamW(P.param_groups(student), lr=P.PEAK_LR, weight_decay=P.WEIGHT_DECAY, fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: P.lr_factor(s, total, T.WARMUP))
    out = os.path.join(OUT, args.name)
    os.makedirs(out, exist_ok=True)
    queue = T.EvalQueue(args.name)
    drift = T.drift_fn(cat, eval_rows, teacher, device)
    cat_stream = G2.Cycle(len(cards), np.random.default_rng(args.seed + 1))
    real_stream = G2.Cycle(len(real), np.random.default_rng(args.seed + 2)) if args.real else None
    drifts = {"ema": {}}
    hs_seen, step, t0 = [], 0, time.time()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, args.epochs + 1):
        student.train()
        acc, n_acc = torch.zeros(4, device=device), 0
        for _ in range(steps_per_epoch):
            ci = torch.from_numpy(cat_stream.take(n_cat)).to(device)
            x, gid = cards[ci], gid_cat[ci]
            if args.real:
                ri = torch.from_numpy(real_stream.take(n_real)).to(device)
                x, gid = torch.cat([x, real[ri]]), torch.cat([gid, gid_real[ri]])
            large, deg, hs, _ = G2.make_batch(x, rng, device)
            opt.zero_grad(set_to_none=True)
            losses = train_step(student_c, teacher_c, large, deg, gid, catv, grp, args.w_ce, device)
            opt.step()
            sched.step()
            P.ema_update(ema, student, min(T.EMA_MAX, (1 + step) / (10 + step)))
            hs_seen.extend(hs.tolist())
            acc += losses
            n_acc += 1
            step += 1
            if step % 10 == 0:
                a = (acc / n_acc).tolist()
                print(f"ep{epoch} step {step}/{total} loss {a[0]:.4f} dist {a[1]:.4f} drift {a[2]:.4f} ce {a[3]:.3f} "
                      f"{step / (time.time() - t0):.2f} it/s peakVRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
                acc, n_acc = torch.zeros(4, device=device), 0
        torch.save(student.state_dict(), os.path.join(out, f"epoch{epoch}-raw.pt"))
        torch.save(ema.state_dict(), os.path.join(out, f"epoch{epoch}-ema.pt"))
        drifts["ema"][epoch] = drift(ema)
        if epoch == args.epochs:
            student.eval()
            drifts["raw"] = drift(student)
        print(f"epoch {epoch} done: cos(S_ema,T) large median {drifts['ema'][epoch]:.4f}", flush=True)
        if epoch < args.epochs and not args.no_epoch_eval:
            queue.submit(epoch, drifts["ema"][epoch])  # the final epoch is evaluated by --post
    torch.cuda.synchronize()
    train_s = time.time() - t0
    hs_arr = np.array(hs_seen)
    metrics = {"arm": args.name, "real": args.real, "real_report": real_rep, "w_ce": args.w_ce, "batch": batch, "batch_catalog": n_cat,
               "batch_real": n_real, "ce_variant": "no no_negative masks; name-group positives for every sample",
               "train_seconds": train_s, "train_it_s": step / train_s, "gpu_images_per_s": 3 * batch * step / train_s,
               "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30, "compile": compile_status,
               "h_fracs": {"70-180": float(np.mean((hs_arr >= 70) & (hs_arr <= 180))), "<70": float(np.mean(hs_arr < 70)),
                           "180-250": float(np.mean((hs_arr > 180) & (hs_arr < 250))), ">=250": float(np.mean(hs_arr >= 250))},
               "drifts": drifts}
    with open(os.path.join(out, "drifts.json"), "w", encoding="utf-8") as f:
        json.dump(drifts, f)
    print(f"training done in {train_s:.0f}s; {metrics['gpu_images_per_s']:.0f} img/s; h fracs {metrics['h_fracs']}", flush=True)
    queue.drain()
    metrics["epochs"] = []
    for e in range(1, args.epochs if not args.no_epoch_eval else 1):
        with open(os.path.join(out, f"metrics-epoch{e}.json"), encoding="utf-8") as f:
            metrics["epochs"].append(json.load(f))
    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


# ---------------------------------------------------------------- post-eval (CPU)
def _jp2(ctx):
    import eval_gen2 as E2

    return E2, E2.build_evalset2(rows=ctx.rows, cat=ctx.cat)


def _dump(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def post(name, epochs):
    limit_ort()
    import large_crop_check as LC
    from pipeline.embed import ClipEmbedder
    from run_real_photos import load_names

    out = os.path.join(OUT, name)
    with open(os.path.join(out, "drifts.json"), encoding="utf-8") as f:
        drifts = json.load(f)
    ctx = V.Context()
    E2, jp2 = _jp2(ctx)
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    for kind in ("ema", "raw"):
        t0 = time.time()
        stem = os.path.join(out, f"epoch{epochs}")
        path = f"{stem}-{kind}-int8.onnx"
        model.visual.load_state_dict(torch.load(f"{stem}-{kind}.pt", map_location="cpu"))
        V.export_int8(model, f"{stem}-{kind}-fp32.onnx", path)
        rec = V.eval_int8(path, ctx)
        rec["drift"]["cos_ST_large_median"] = drifts["ema"][str(epochs)] if kind == "ema" else drifts["raw"]
        rec["eval2"] = E2.eval_set(path, ctx, jp2)
        rec["eval_seconds"] = time.time() - t0
        _dump(os.path.join(out, f"final-{kind}.json"), rec)
        print(f"{name} {kind}: eval1+eval2 done in {rec['eval_seconds']:.0f}s", flush=True)
    large(LC, ClipEmbedder, load_names, os.path.join(out, f"epoch{epochs}-ema-int8.onnx"), os.path.join(out, "final-ema-large.json"))


def large(LC, ClipEmbedder, load_names, onnx_path, dest):
    from pipeline.catalog import Catalog

    cat = Catalog.load(os.path.join(ROOT, "assets", "catalog.bin.gz"), os.path.join(ROOT, "assets", "id_map.json"))
    nm = load_names()
    res = LC.task1(cat, [nm.get(c) for c in cat.ids], {"m": ClipEmbedder(onnx_path)})
    _dump(dest, {k.split("@")[1]: v for k, v in res.items()})


def _extra(label, onnx_path, cache):
    """eval2 and large-crop for a model that lacks them (cached in fastC_extra.json)."""
    if label in cache:
        return cache[label]
    limit_ort()
    import large_crop_check as LC
    from pipeline.embed import ClipEmbedder
    from run_real_photos import load_names

    ctx = V.Context()
    E2, jp2 = _jp2(ctx)
    tmp = os.path.join(OUT, f"_large_{label}.json")
    large(LC, ClipEmbedder, load_names, onnx_path, tmp)
    with open(tmp, encoding="utf-8") as f:
        cache[label] = {"eval2": E2.eval_set(onnx_path, ctx, jp2), "large": json.load(f)}
    os.remove(tmp)
    _dump(os.path.join(OUT, "fastC_extra.json"), cache)
    return cache[label]


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compare():
    extra_path = os.path.join(OUT, "fastC_extra.json")
    cache = _load(extra_path) if os.path.exists(extra_path) else {}
    task1 = _load(os.path.join(ROOT, "data", "reports", "large-crop-check", "task1.json"))
    ship_large = {k.split("@")[1]: v for k, v in task1.items() if k.startswith("ship@")}
    b_large = {k.split("@")[1]: v for k, v in task1.items() if k.startswith("B@")}
    fb = _load(os.path.join(OUT, "fastB", "metrics.json"))["epochs"][-1]
    bm = _load(os.path.join(OUT, "B", "metrics.json"))["epochs"][-1]["raw_student"]
    models = {
        "shipped": (_load(os.path.join(OUT, "baseline", "metrics.json")), _load(os.path.join(OUT, "baseline2", "metrics.json")), ship_large),
        "B_raw": (bm, _extra("B_raw", os.path.join(OUT, "B", "raw-final-int8.onnx"), cache)["eval2"], b_large),
        "fastB": (fb, None, None),
    }
    x = _extra("fastB", os.path.join(OUT, "fastB", f"epoch{fb['epoch']}-int8.onnx"), cache)
    models["fastB"] = (fb, x["eval2"], x["large"])
    for run in ("fastC_cat", "fastC_real"):
        for kind in ("ema", "raw"):
            p = os.path.join(OUT, run, f"final-{kind}.json")
            if os.path.exists(p):
                rec = _load(p)
                lp = os.path.join(OUT, run, "final-ema-large.json")
                models[f"{run}_{kind}"] = (rec, rec["eval2"], _load(lp) if kind == "ema" and os.path.exists(lp) else None)
    h = lambda m, hh, k: m["heights"][str(hh)]["all"][k]  # noqa: E731
    rows = []
    for hh in (60, 90, 120, 150, 200):
        rows.append((f"e1 top1@{hh}", lambda e1, e2, lg, hh=hh: h(e1, hh, "top1")))
    for hh in (150, 200):
        rows.append((f"e1 acc-wrong@{hh}", lambda e1, e2, lg, hh=hh: h(e1, hh, "accepted_wrong")))
    rows.append(("e1 hub1@60", lambda e1, e2, lg: h(e1, 60, "hub1_share")))
    for hh in (120, 150, 200):
        rows.append((f"e2 top1@{hh}", lambda e1, e2, lg, hh=hh: h(e2, hh, "top1")))
    rows.append(("clean median cos", lambda e1, e2, lg: e1["drift"]["clean_median_cos"]))
    rows.append(("cos(S,T) large", lambda e1, e2, lg: e1["drift"].get("cos_ST_large_median")))
    for sz in ("full", "400", "300"):
        for k in ("top1_id", "top1_base"):
            rows.append((f"large {sz} {k[5:]}", lambda e1, e2, lg, sz=sz, k=k: lg[sz][k] if lg else None))
    table, result = [], {}
    names = list(models)
    table.append(f"{'metric':20s}" + "".join(f"{n:>16s}" for n in names))
    for label, fn in rows:
        vals = []
        for n in names:
            e1, e2, lg = models[n]
            try:
                v = fn(e1, e2, lg)
            except (KeyError, TypeError):
                v = None
            vals.append(v)
            result.setdefault(n, {})[label] = v
        table.append(f"{label:20s}" + "".join(f"{'-' if v is None else (f'{v:.4f}' if isinstance(v, float) else str(v)):>16s}" for v in vals))
    print("\n".join(table))
    _dump(COMPARE, {"table": table, "models": result})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--extra-packs", nargs="*", default=[], help="extra label jsonl names in pseudo-labels/v3 (fastD: hibid-pack-train.jsonl)")
    ap.add_argument("--exclude-eval", action="store_true", help="drop eval-duplicate / eval-SHA photos before caching (fastD)")
    ap.add_argument("--real-cache", default=None, help="separate real-crop cache path (fastD)")
    ap.add_argument("--all-physical", action="store_true", help="fastE: all physical-card ids, name filter off for real crops, harvest eval dirs excluded")
    ap.add_argument("--cards-cache", default=os.path.join(OUT, "cache", "cards320_all.pt"))
    ap.add_argument("--no-epoch-eval", action="store_true", help="skip the background int8 eval of epochs 1..N-1")
    ap.add_argument("--w-ce", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", default="auto")
    ap.add_argument("--seed", type=int, default=T.SEED)
    ap.add_argument("--post")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    if args.compare:
        compare()
    elif args.post:
        post(args.post, args.epochs)
    else:
        if not args.name:
            ap.error("--name is required")
        train(args)


if __name__ == "__main__":
    main()
