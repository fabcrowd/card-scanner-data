"""GPU saturation benchmark: train_fast for N steps while sampling nvidia-smi at 1 Hz; optional old-pilot CPU-pipeline timing."""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
FIELDS = "utilization.gpu,power.draw,memory.used,clocks.sm,power.limit"


def gpu_busy():
    util = float(subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], text=True).split()[0])
    return util > 10


def wait_idle(max_minutes=40):
    ok, deadline = 0, time.time() + max_minutes * 60
    while time.time() < deadline:
        ok = 0 if gpu_busy() else ok + 1
        if ok >= 2:
            return True
        time.sleep(60 if ok == 0 else 5)
    return False


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.rows, self.stop = [], threading.Event()

    def run(self):
        while not self.stop.is_set():
            t = time.time()
            line = subprocess.check_output(["nvidia-smi", f"--query-gpu={FIELDS}", "--format=csv,noheader,nounits"], text=True).strip()
            self.rows.append((t, *[float(v) for v in line.split(",")]))
            self.stop.wait(max(0.0, 1.0 - (time.time() - t)))


def run_fast(steps, extra):
    cmd = [PY, "-u", os.path.join(ROOT, "train", "train_fast.py"), "--name", "bench", "--bench-steps", str(steps), *extra]
    s = Sampler()
    s.start()
    t_start = t_end = None
    result = None
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT)
    for line in proc.stdout:
        line = line.rstrip()
        if line.startswith("BENCH_START"):
            t_start = time.time()
        elif line.startswith("BENCH_END"):
            t_end = time.time()
        elif line.startswith("BENCH_RESULT"):
            result = json.loads(line.split(" ", 1)[1])
        elif line.startswith(("autotune", "name ", "torch.compile")):
            print(line, flush=True)
    proc.wait()
    s.stop.set()
    s.join()
    if result is None:
        raise SystemExit("train_fast produced no BENCH_RESULT")
    a = np.array([r for r in s.rows if t_start <= r[0] <= t_end])
    util, power = a[:, 1], a[:, 2]
    result["sampler"] = {
        "samples": len(a), "util_mean": float(util.mean()), "util_p10": float(np.percentile(util, 10)), "util_min": float(util.min()),
        "power_mean_w": float(power.mean()), "power_peak_w": float(power.max()), "power_limit_w": float(a[0, 5]),
        "mem_peak_gib_smi": float(a[:, 3].max() / 1024), "sm_clock_mean_mhz": float(a[:, 4].mean())}
    return result


def old_pilot_bench(steps=50, warm=5, batch=128):
    """Pilot loop (train_pilot settings: B=128, 8 PIL workers, bf16) timed over `steps` after `warm` steps."""
    sys.path.insert(0, os.path.join(ROOT, "train"))
    import torch
    import torch.nn.functional as F
    import pilot_data as D
    import pilot_eval as V
    import train_pilot as P

    ctx = V.Context()
    _, train_rows, _ = D.build_split(ctx.cat, ctx.name_of)
    paths = {r: D.E.image_path(ctx.cat.ids[r]) for r in train_rows}
    _, teacher, student, _ = P.build_models("cuda")
    sampler = D.EpochBatches(train_rows, batch, 7001)
    loader = torch.utils.data.DataLoader(D.PairDataset(paths), batch_sampler=sampler, num_workers=8, pin_memory=True, prefetch_factor=4)
    opt = torch.optim.AdamW(P.param_groups(student), lr=P.PEAK_LR, weight_decay=P.WEIGHT_DECAY)
    student.train()
    t0 = None
    for i, (large, deg, rows, hs) in enumerate(loader):
        if i == warm:
            torch.cuda.synchronize()
            t0 = time.time()
        large, deg = large.cuda(non_blocking=True), deg.cuda(non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            t = F.normalize(teacher(large).float(), dim=-1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s = student(torch.cat([deg, large]))
        s = F.normalize(s.float(), dim=-1)
        (((1 - (s[:batch] * t).sum(-1)) ** 2).mean() + ((1 - (s[batch:] * t).sum(-1)) ** 2).mean()).backward()
        opt.step()
        if i + 1 >= warm + steps:
            break
    torch.cuda.synchronize()
    dt = time.time() - t0
    return {"batch": batch, "steps": steps, "steps_per_s": steps / dt, "images_per_s": 3 * batch * steps / dt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch", default="auto")
    ap.add_argument("--phase-timers", action="store_true")
    ap.add_argument("--old", action="store_true", help="also time the old CPU-pipeline pilot (B=128, 50 steps)")
    ap.add_argument("--no-wait", action="store_true")
    a = ap.parse_args()
    if not a.no_wait and not wait_idle():
        raise SystemExit("GPU never became idle within 40 minutes")
    extra = ["--batch", a.batch] + (["--phase-timers"] if a.phase_timers else [])
    res = run_fast(a.steps, extra)
    print("FAST", json.dumps(res, indent=1))
    if a.old:
        if not a.no_wait and not wait_idle(5):
            print("GPU busy; skipping old-pilot timing")
            return
        old = old_pilot_bench()
        print("OLD", json.dumps(old))
        print(f"speed-up: {res['steps_per_s'] * res['batch'] / (old['steps_per_s'] * old['batch']):.1f}x in cards/s "
              f"({res['images_per_s'] / old['images_per_s']:.1f}x images/s)")


if __name__ == "__main__":
    main()
