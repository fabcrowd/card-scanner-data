"""Full-tower asymmetric distillation pilot: student ViT-B/32 tower -> frozen original tower at high resolution."""
import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pilot_data as D  # noqa: E402
import pilot_eval as V  # noqa: E402

E = D.E
SEED = 7001
PEAK_LR = 2e-5
LAYER_DECAY = 0.7
WEIGHT_DECAY = 0.01
WARMUP = 300
EMA_DECAY = 0.999
TAU = 0.04
N_EVAL_DRIFT = 300


def build_models(device):
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    teacher = copy.deepcopy(model.visual).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = copy.deepcopy(model.visual).to(device)
    for n in ("conv1", "ln_pre"):  # patch embedding (+ ln_pre, not in the trainable list) frozen
        for p in getattr(student, n).parameters():
            p.requires_grad_(False)
    student.class_embedding.requires_grad_(False)
    student.positional_embedding.requires_grad_(False)
    ema = copy.deepcopy(student).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    return model, teacher, student, ema


def param_groups(student):
    blocks = student.transformer.resblocks
    n = len(blocks)
    groups = [{"params": list(b.parameters()), "lr": PEAK_LR * LAYER_DECAY ** (n - 1 - i)} for i, b in enumerate(blocks)]
    groups.append({"params": list(student.ln_post.parameters()) + [student.proj], "lr": PEAK_LR})
    return groups


def lr_factor(step, total, warmup):
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


@torch.no_grad()
def ema_update(ema, student, decay):
    ep, sp = list(ema.parameters()), list(student.parameters())
    torch._foreach_mul_(ep, decay)
    torch._foreach_add_(ep, sp, alpha=1 - decay)


def ce_multi_positive(s, rows, catv, grp):
    logits = s @ catv.T / TAU
    mask = grp[rows][:, None] == grp[None, :]
    return (torch.logsumexp(logits, 1) - torch.logsumexp(logits.masked_fill(~mask, float("-inf")), 1)).mean()


def large_view(path):
    with Image.open(path) as im:
        return D.to_tensor(D.resize_to_height(im.convert("RGB"), D.LARGE_H, Image.BOX))


@torch.no_grad()
def drift_cos(ctx, teacher, ema, device):
    paths = [E.image_path(ctx.cat.ids[r]) for r in ctx.rows[:N_EVAL_DRIFT]]
    x = torch.stack([large_view(p) for p in paths]).to(device)
    cs = []
    for s in range(0, len(x), 50):
        t = F.normalize(teacher(x[s:s + 50]).float(), dim=-1)
        e = F.normalize(ema(x[s:s + 50]).float(), dim=-1)
        cs.append((t * e).sum(-1).cpu())
    return float(torch.cat(cs).median())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B"], required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--raw-final", action="store_true", help="also evaluate the non-EMA student after the last epoch")
    args = ap.parse_args()
    w_ce = 0.0 if args.arm == "A" else 0.1
    device = "cuda"
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t_start = time.time()

    ctx = V.Context()
    split, train_rows, _ = D.build_split(ctx.cat, ctx.name_of, os.path.join(V.OUT, "split.json"))
    print("split counts:", json.dumps(split["counts"]), "| digital rule:", split["digital_rule"], flush=True)
    if args.smoke:
        train_rows = train_rows[:400]
    paths = {r: E.image_path(ctx.cat.ids[r]) for r in train_rows}

    model, teacher, student, ema = build_models(device)
    catv = torch.from_numpy(ctx.cat.vectors).to(device)
    names, grp_of, grp = ctx.name_of, {}, []
    for i, nm in enumerate(names):
        grp.append(grp_of.setdefault(nm, len(grp_of)) if nm else -1 - i)
    grp = torch.tensor(grp, device=device)

    micro = args.batch // args.accum
    sampler = D.EpochBatches(train_rows, args.batch, SEED)
    loader = torch.utils.data.DataLoader(
        D.PairDataset(paths), batch_sampler=sampler, num_workers=8, persistent_workers=True,
        pin_memory=True, prefetch_factor=4)
    steps_per_epoch = len(sampler)
    total = 20 if args.smoke else args.epochs * steps_per_epoch
    warmup = 5 if args.smoke else WARMUP
    opt = torch.optim.AdamW(param_groups(student), lr=PEAK_LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, total, warmup))
    print(f"arm {args.arm} w_ce={w_ce} batch={args.batch} accum={args.accum} steps/epoch={steps_per_epoch} total={total}", flush=True)

    outdir = os.path.join(V.OUT, "smoke" if args.smoke else args.arm)
    os.makedirs(outdir, exist_ok=True)
    try:
        base = json.load(open(os.path.join(V.OUT, "baseline", "metrics.json"), encoding="utf-8"))
    except OSError:
        base = None
    metrics = {"arm": args.arm, "w_ce": w_ce, "epochs": []}
    step, hs_seen, t0 = 0, [], time.time()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, args.epochs + 1):
        sampler.epoch = epoch - 1
        student.train()
        acc = np.zeros(4)
        n_acc = 0
        for large, deg, rows, hs in loader:
            large, deg, rows = large.to(device, non_blocking=True), deg.to(device, non_blocking=True), rows.to(device)
            hs_seen.extend(hs.tolist())
            opt.zero_grad(set_to_none=True)
            for m0 in range(0, len(rows), micro):
                sl = slice(m0, m0 + micro)
                with torch.no_grad():
                    t = F.normalize(teacher(large[sl]).float(), dim=-1)  # T(big) == T(large): same 320 px image
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    s = student(torch.cat([deg[sl], large[sl]]))
                s = F.normalize(s.float(), dim=-1)
                sd, sL = s[:micro], s[micro:]
                l_dist = ((1 - (sd * t).sum(-1)) ** 2).mean()
                l_drift = ((1 - (sL * t).sum(-1)) ** 2).mean()
                l_ce = ce_multi_positive(sd, rows[sl], catv, grp) if w_ce > 0 else torch.zeros((), device=device)
                loss = (l_dist + l_drift + w_ce * l_ce) / args.accum
                loss.backward()
                acc += np.array([float(loss) * args.accum, float(l_dist), float(l_drift), float(l_ce)]) / args.accum
                n_acc += 1.0 / args.accum
            opt.step()
            sched.step()
            ema_update(ema, student, EMA_DECAY)
            step += 1
            if step % 10 == 0 or args.smoke:
                a = acc / max(n_acc, 1e-9)
                print(f"ep{epoch} step {step}/{total} loss {a[0]:.4f} dist {a[1]:.4f} drift {a[2]:.4f} ce {a[3]:.3f} "
                      f"lr_last {sched.get_last_lr()[-1]:.2e} {step / (time.time() - t0):.2f} it/s "
                      f"peakVRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
                acc, n_acc = np.zeros(4), 0
            if args.smoke and step >= total:
                break
        if args.smoke:
            if step >= total:
                break
            continue
        torch.save(ema.state_dict(), os.path.join(outdir, f"epoch{epoch}.pt"))
        rec = evaluate(ctx, model, teacher, ema, device, os.path.join(outdir, f"epoch{epoch}"), epoch)
        if args.raw_final and epoch == args.epochs:
            raw = copy.deepcopy(ema)
            raw.load_state_dict(student.state_dict())
            rec["raw_student"] = evaluate(ctx, model, teacher, raw, device, os.path.join(outdir, "raw-final"), epoch)
        metrics["epochs"].append(rec)
        with open(os.path.join(outdir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump({**metrics, "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
                       "train_it_s": step / (time.time() - t0), "h_ge250_frac": float(np.mean(np.array(hs_seen) >= 250)),
                       "h_70_180_frac": float(np.mean((np.array(hs_seen) >= 70) & (np.array(hs_seen) <= 180)))}, f, indent=2)
        print(f"=== arm {args.arm} epoch {epoch} (EMA, int8) ===\n{V.table(rec, base)}", flush=True)
    hs_arr = np.array(hs_seen)
    print(f"deg H fractions: 70-180 {np.mean((hs_arr >= 70) & (hs_arr <= 180)):.3f}, >=250 {np.mean(hs_arr >= 250):.3f}; "
          f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB; {step / (time.time() - t0):.2f} it/s; "
          f"wall {time.time() - t_start:.0f}s", flush=True)


def evaluate(ctx, model, teacher, ema_like, device, stem, epoch):
    t0 = time.time()
    export_model = copy.deepcopy(model)
    export_model.visual.load_state_dict({k: v.cpu() for k, v in ema_like.state_dict().items()})
    V.export_int8(export_model, stem + "-fp32.onnx", stem + "-int8.onnx")
    rec = V.eval_int8(stem + "-int8.onnx", ctx)
    rec["epoch"] = epoch
    rec["drift"]["cos_ST_large_median"] = drift_cos(ctx, teacher, ema_like, device)
    rec["eval_seconds"] = time.time() - t0
    return rec


if __name__ == "__main__":
    main()
