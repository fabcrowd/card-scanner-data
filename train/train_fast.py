"""GPU-saturating version of train_pilot: GPU-resident cards + GPU augmentation (gpu_data), bf16, fused AdamW, auto batch.

Same model and loss as train_pilot (student ViT-B/32 tower vs frozen teacher; 1.0*(1-cos(S(deg),T))^2 +
1.0*(1-cos(S(large),T))^2 + w_ce*CE over frozen catalog rows, tau 0.04, positives = rows with the same NORMALISED name).
Per-epoch int8 eval of the EMA weights runs in a background CPU-only subprocess (this file with --eval-worker).
"""
import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gpu_data as G  # noqa: E402
import pilot_data as D  # noqa: E402
import pilot_eval as V  # noqa: E402
import train_pilot as P  # noqa: E402  (build_models / param_groups / loss helpers are reused unchanged)

SEED = 7001
WARMUP = 30
EMA_MAX = 0.999
VRAM_LIMIT_GIB = 14.5
BATCH_CANDIDATES = (128, 192, 256, 384, 512, 768)
EVAL_THREADS = 6
BENCH_WARM = 10


def norm_groups(name_of):
    groups, ids = {}, []
    for i, nm in enumerate(name_of):
        key = G.norm_name(nm)
        ids.append(groups.setdefault(key, len(groups)) if key else -1 - i)
    return torch.tensor(ids)


class Timers:
    """Per-phase wall timers (synchronise the GPU around each phase, so only enabled for profiling)."""

    def __init__(self, enabled):
        self.enabled, self.t, self.mark = enabled, {}, None

    def lap(self, phase):
        if not self.enabled:
            return
        torch.cuda.synchronize()
        now = time.perf_counter()
        if self.mark is not None:
            self.t[phase] = self.t.get(phase, 0.0) + now - self.mark
        self.mark = now

    def start(self):
        if self.enabled:
            torch.cuda.synchronize()
            self.mark = time.perf_counter()


def try_compile(student, teacher, device):
    """torch.compile both towers; fall back to eager (and say so) if the first call fails (e.g. no triton on Windows)."""
    try:
        cs, ct = torch.compile(student), torch.compile(teacher)
        x = torch.randn(4, 3, 224, 224, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cs(x).sum().backward()
            with torch.no_grad():
                ct(x)
        student.zero_grad(set_to_none=True)
        print("torch.compile: OK", flush=True)
        return cs, ct, "ok"
    except Exception as err:  # noqa: BLE001 - dynamo/inductor raise many types
        student.zero_grad(set_to_none=True)
        msg = str(err).strip().splitlines()[0][:160] if str(err).strip() else type(err).__name__
        print(f"torch.compile FAILED, running eager: {type(err).__name__}: {msg}", flush=True)
        return student, teacher, f"failed: {type(err).__name__}: {msg}"


def train_step(student_c, teacher_c, large, deg, rows, catv, grp, w_ce, device, tm):
    tm.start()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        t = F.normalize(teacher_c(large).float(), dim=-1)
    tm.lap("teacher_fwd")
    n = len(rows)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        s = student_c(torch.cat([deg, large]))
    s = F.normalize(s.float(), dim=-1)
    sd, sl = s[:n], s[n:]
    l_dist = ((1 - (sd * t).sum(-1)) ** 2).mean()
    l_drift = ((1 - (sl * t).sum(-1)) ** 2).mean()
    tm.lap("student_fwd")
    l_ce = P.ce_multi_positive(sd, rows, catv, grp) if w_ce > 0 else torch.zeros((), device=device)
    tm.lap("ce")
    loss = l_dist + l_drift + w_ce * l_ce
    loss.backward()
    tm.lap("student_bwd")
    return torch.stack([loss.detach(), l_dist.detach(), l_drift.detach(), l_ce.detach()])


def autotune_batch(student, student_c, teacher_c, cards, rows_t, catv, grp, w_ce, device, rng):
    """Measured images/s per candidate; step up only if >5% faster and peak allocated VRAM < VRAM_LIMIT_GIB."""
    best, best_ips = None, 0.0
    tm = Timers(False)
    for b in BATCH_CANDIDATES:
        if b > len(cards):
            break
        opt = torch.optim.AdamW(P.param_groups(student), lr=0.0, weight_decay=0.0, fused=True)  # lr 0: weights untouched
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            for i in range(6):
                if i == 2:
                    torch.cuda.synchronize()
                    t0 = time.time()
                pos = torch.from_numpy(rng.choice(len(cards), b, replace=False)).to(device)
                large, deg, _, _ = G.make_batch(cards[pos], rng, device)
                opt.zero_grad(set_to_none=True)
                train_step(student_c, teacher_c, large, deg, rows_t[pos], catv, grp, w_ce, device, tm)
                opt.step()
            torch.cuda.synchronize()
            ips = 3 * b * 4 / (time.time() - t0)
            peak = torch.cuda.max_memory_allocated() / 2**30
        except torch.OutOfMemoryError:
            ips, peak = 0.0, float("inf")
        del opt
        student.zero_grad(set_to_none=True)
        print(f"autotune B={b}: {ips:.0f} img/s, {3 * (len(cards) // b * b) / max(ips, 1e-9):.1f} s/epoch, peak {peak:.2f} GiB", flush=True)
        if peak >= VRAM_LIMIT_GIB or (best is not None and ips < 1.05 * best_ips):
            break
        best, best_ips = b, ips
    torch.cuda.empty_cache()
    return best


def eval_worker(name, epoch, drift):
    """CPU-only: export int8 of the EMA weights (pilot exporter) and run the pilot's eval at ORT batch 1."""
    import onnxruntime as ort

    torch.set_num_threads(EVAL_THREADS)
    real = ort.InferenceSession

    def limited(path, *a, **k):
        so = ort.SessionOptions()
        so.intra_op_num_threads = EVAL_THREADS
        return real(path, sess_options=so, **k)

    ort.InferenceSession = limited
    t0 = time.time()
    out = os.path.join(V.OUT, name)
    stem = os.path.join(out, f"epoch{epoch}")
    ctx = V.Context()
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model.visual.load_state_dict(torch.load(f"{stem}-ema.pt", map_location="cpu"))
    V.export_int8(model, stem + "-fp32.onnx", stem + "-int8.onnx")
    rec = V.eval_int8(stem + "-int8.onnx", ctx)
    rec["epoch"] = epoch
    rec["drift"]["cos_ST_large_median"] = drift
    rec["eval_seconds"] = time.time() - t0
    with open(os.path.join(out, f"metrics-epoch{epoch}.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)


class EvalQueue:
    """Runs eval workers one at a time in the background so training never waits on them."""

    def __init__(self, name):
        self.name, self.pending, self.proc = name, [], None

    def submit(self, epoch, drift):
        self.pending.append((epoch, drift))
        self.poll()

    def poll(self):
        if self.proc is not None and self.proc.poll() is None:
            return
        if self.proc is not None and self.proc.returncode != 0:
            print(f"WARNING: eval worker exited {self.proc.returncode}", flush=True)
        self.proc = None
        if self.pending:
            epoch, drift = self.pending.pop(0)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": str(EVAL_THREADS)}
            log = open(os.path.join(V.OUT, self.name, f"eval-epoch{epoch}.log"), "w")
            self.proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--eval-worker", self.name, str(epoch), repr(drift)],
                                         env=env, stdout=log, stderr=subprocess.STDOUT)

    def drain(self):
        while self.proc is not None or self.pending:
            self.poll()
            time.sleep(5)


@torch.no_grad()
def drift_fn(cat, rows, teacher, device):
    paths = [D.E.image_path(cat.ids[r]) for r in rows[:P.N_EVAL_DRIFT]]
    x = torch.stack([P.large_view(p) for p in paths]).to(device)
    t = torch.cat([F.normalize(teacher(x[s:s + 50]).float(), dim=-1) for s in range(0, len(x), 50)])

    def drift(ema):
        e = torch.cat([F.normalize(ema(x[s:s + 50]).float(), dim=-1) for s in range(0, len(x), 50)])
        return float((t * e).sum(-1).cpu().median())

    return drift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--w-ce", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", default="auto")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--bench-steps", type=int, default=0, help="timed steps after a short warm-up; no saves/evals")
    ap.add_argument("--phase-timers", action="store_true")
    ap.add_argument("--eval-worker", nargs=3, metavar=("NAME", "EPOCH", "DRIFT"))
    args = ap.parse_args()
    if args.eval_worker:
        eval_worker(args.eval_worker[0], int(args.eval_worker[1]), float(args.eval_worker[2]))
        return

    if not args.name:
        ap.error("--name is required")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    from pipeline.catalog import Catalog
    import score_synth_fusion as S

    cat = Catalog.load(os.path.join(D.ROOT, "assets", "catalog.bin.gz"), os.path.join(D.ROOT, "assets", "id_map.json"))
    nm = S.load_names()
    name_of = [nm.get(c) for c in cat.ids]
    _, eval_rows = D.eval_rows(cat)
    cards_cpu, rows_np, rep = G.load_training_cards(cat, name_of, eval_rows, os.path.join(V.OUT, "split.json"))
    print("training cards:", json.dumps(rep), flush=True)
    cards = cards_cpu.to(device)
    rows_t = torch.from_numpy(rows_np).to(device)
    catv = torch.from_numpy(cat.vectors).to(device)
    grp = norm_groups(name_of).to(device)

    model, teacher, student, ema = P.build_models(device)
    student_c, teacher_c, compile_status = (student, teacher, "disabled") if args.no_compile else try_compile(student, teacher, device)
    batch = int(args.batch) if args.batch != "auto" else autotune_batch(student, student_c, teacher_c, cards, rows_t, catv, grp, args.w_ce, device, rng)
    steps_per_epoch = len(cards) // batch
    total = BENCH_WARM + args.bench_steps if args.bench_steps else args.epochs * steps_per_epoch
    print(f"name {args.name} w_ce={args.w_ce} batch={batch} steps/epoch={steps_per_epoch} total={total} compile={compile_status}", flush=True)

    opt = torch.optim.AdamW(P.param_groups(student), lr=P.PEAK_LR, weight_decay=P.WEIGHT_DECAY, fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: P.lr_factor(s, total, WARMUP))
    out = os.path.join(V.OUT, args.name)
    queue = None
    drift = None
    if not args.bench_steps:
        os.makedirs(out, exist_ok=True)
        queue = EvalQueue(args.name)
        drift = drift_fn(cat, eval_rows, teacher, device)
    tm = Timers(args.phase_timers)
    metrics = {"arm": args.name, "w_ce": args.w_ce, "epochs": []}
    drifts, hs_seen, step, t0, t_bench = {}, [], 0, time.time(), None
    torch.cuda.reset_peak_memory_stats()
    perm_rng = np.random.default_rng(args.seed + 1)
    epoch = 0
    while step < total:
        epoch += 1
        order = perm_rng.permutation(len(cards))
        student.train()
        acc = torch.zeros(4, device=device)
        n_acc = 0
        for b in range(steps_per_epoch):
            if step >= total:
                break
            if args.bench_steps and step == BENCH_WARM:
                torch.cuda.synchronize()
                t_bench = time.time()
                print("BENCH_START", flush=True)
            tm.start()
            pos = torch.from_numpy(order[b * batch:(b + 1) * batch]).to(device)
            large, deg, hs, _ = G.make_batch(cards[pos], rng, device)
            rows = rows_t[pos]
            tm.lap("data_aug")
            opt.zero_grad(set_to_none=True)
            losses = train_step(student_c, teacher_c, large, deg, rows, catv, grp, args.w_ce, device, tm)
            opt.step()
            sched.step()
            P.ema_update(ema, student, min(EMA_MAX, (1 + step) / (10 + step)))
            tm.lap("opt_ema")
            hs_seen.extend(hs.tolist())
            acc += losses
            n_acc += 1
            step += 1
            if step % 10 == 0:
                a = (acc / n_acc).tolist()
                print(f"ep{epoch} step {step}/{total} loss {a[0]:.4f} dist {a[1]:.4f} drift {a[2]:.4f} ce {a[3]:.3f} "
                      f"{step / (time.time() - t0):.2f} it/s peakVRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
                acc, n_acc = torch.zeros(4, device=device), 0
        if args.bench_steps:
            continue
        torch.save(student.state_dict(), os.path.join(out, f"epoch{epoch}-raw.pt"))
        torch.save(ema.state_dict(), os.path.join(out, f"epoch{epoch}-ema.pt"))
        drifts[epoch] = drift(ema)
        print(f"epoch {epoch} done: drift cos(S_ema,T) large median {drifts[epoch]:.4f}; eval queued", flush=True)
        queue.submit(epoch, drifts[epoch])
    torch.cuda.synchronize()
    if args.bench_steps:
        dt = time.time() - t_bench
        n = args.bench_steps
        print("BENCH_END", flush=True)
        print("BENCH_RESULT " + json.dumps({
            "batch": batch, "steps": n, "steps_per_s": n / dt, "images_per_s": 3 * batch * n / dt,
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30, "compile": compile_status,
            "phase_seconds": tm.t, "nvjpeg": G._USE_NVJPEG["on"]}), flush=True)
        return
    train_s = time.time() - t0
    hs_arr = np.array(hs_seen)
    print(f"training done in {train_s:.0f}s; waiting for background evals", flush=True)
    queue.drain()
    for e in range(1, epoch + 1):
        with open(os.path.join(out, f"metrics-epoch{e}.json"), encoding="utf-8") as f:
            metrics["epochs"].append(json.load(f))
    metrics.update({"peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30, "train_it_s": step / train_s, "batch": batch,
                    "compile": compile_status, "h_ge250_frac": float(np.mean(hs_arr >= 250)),
                    "h_70_180_frac": float(np.mean((hs_arr >= 70) & (hs_arr <= 180)))})
    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    base = json.load(open(os.path.join(V.OUT, "baseline", "metrics.json"), encoding="utf-8"))
    for rec in metrics["epochs"]:
        print(f"=== {args.name} epoch {rec['epoch']} (EMA, int8) ===\n{V.table(rec, base)}", flush=True)


if __name__ == "__main__":
    main()
