"""The one gate: does the candidate bundle beat the live bundle on the fixed regression set, hold on the exam,
and add no wrong "high" reads? Replays search + fusion like exam.score_exam, but with the adapter in the loop.

candidate / live: dicts of optional parts {"memory": pack bytes or Pack, "adapter": adapter.bin bytes, "check": ignored
here: it drops false boxes and does not change what a card is answered}. exam / regression: (embeddings, card ids)
like regression.load_regression. The adapter maps the query and every catalog row; memory rows are compared with the
raw query and must beat the RAW catalog top score (lib/references.js applyMemory does the same). An answer that the adapter or memory changed is advisory, never "high" (self-learning plan, "One gate").
"""
import numpy as np

from .catalog import Catalog
from .exam import TOP_K, is_high
from .pack import Pack, read_pack
from .train_adapter import apply_adapter, parse_adapter

MIN_EXAM_CASES = 30
REF_MIN, REF_MARGIN = 0.93, 0.02  # exam.score_exam defaults (lib/references.js)
ACCEPT_MIN, CLEAR_GAP, HIGH_SIM = 0.72, 0.02, 0.85  # lib/fusion.js (pins.json "accept")
MAX_ABSTAIN_RISE = 0.02  # abstain rate may not rise by more than 2 points


def _memory(part):
    if part is None:
        return [], np.zeros((0, 512), dtype=np.float32)
    pack = part if isinstance(part, Pack) else read_pack(part)
    return pack.card_ids, pack.dequantized()


def _accepts(top, gap):
    return top >= ACCEPT_MIN and (gap >= CLEAR_GAP or top >= HIGH_SIM)


def _replay(parts, embeddings, truths, catalog):
    """Counts for one bundle: top-1 right, wrong "high" reads, and the accepted-right / accepted-wrong / abstained split."""
    counts = {"right": 0, "wrongHigh": 0, "acceptedRight": 0, "acceptedWrong": 0, "abstained": 0, "n": len(truths)}
    if len(truths) == 0:
        return counts
    embeddings = np.asarray(embeddings, dtype=np.float32).reshape(-1, 512)
    raw_idx, raw_scores = catalog.top_k_batch(embeddings, TOP_K)
    idx, scores, ranks = raw_idx, raw_scores, raw_scores
    if parts.get("adapter") is not None:
        adapter = parse_adapter(parts["adapter"])
        idx, ranks = Catalog(apply_adapter(adapter, catalog.vectors), catalog.ids).top_k_batch(apply_adapter(adapter, embeddings), TOP_K)
        scores = np.einsum("nd,nkd->nk", embeddings, catalog.vectors[idx])
    mem_ids, mem_vecs = _memory(parts.get("memory"))
    for i, truth in enumerate(truths):
        top_ids = [catalog.ids[j] for j in idx[i]]
        predicted = top_ids[0]
        gap = float(ranks[i][0] - ranks[i][1]) if len(ranks[i]) > 1 else float(scores[i][0])
        high = is_high(raw_scores[i]) and predicted == catalog.ids[raw_idx[i][0]]
        if len(mem_ids):
            sims = mem_vecs @ embeddings[i]
            best = int(np.argmax(sims))
            if sims[best] >= REF_MIN and sims[best] - raw_scores[i][0] >= REF_MARGIN and mem_ids[best] in top_ids:
                predicted, high, gap = mem_ids[best], False, 0.0  # promoted at the top score: a tie with the old top
        counts["right"] += predicted == truth
        counts["wrongHigh"] += high and predicted != truth
        if _accepts(float(scores[i][0]), gap):
            counts["acceptedRight" if predicted == truth else "acceptedWrong"] += 1
        else:
            counts["abstained"] += 1
    return counts


def better_outcome(before, after, top1_before, top1_after):
    """Whether a bundle is better for the user on one set: top-1 up, accepted-right up, accepted-wrong not up, and the
    abstain rate up by no more than 2 points. before / after are outcome dicts with the same n."""
    n = before["n"]
    return bool(
        n > 0
        and top1_after > top1_before
        and after["acceptedRight"] > before["acceptedRight"]
        and after["acceptedWrong"] <= before["acceptedWrong"]
        and after["abstained"] / n - before["abstained"] / n <= MAX_ABSTAIN_RISE + 1e-9
    )


def _no_worse(before, after, top1_before, top1_after):
    """The exam's weaker rule: no drop in top-1, and accepted-wrong and abstain rates not up beyond the same limit."""
    n = before["n"]
    return bool(
        top1_after >= top1_before
        and after["acceptedWrong"] <= before["acceptedWrong"]
        and after["abstained"] / n - before["abstained"] / n <= MAX_ABSTAIN_RISE + 1e-9
    )


def _outcome(counts):
    return {k: int(counts[k]) for k in ("acceptedRight", "acceptedWrong", "abstained", "n")}


def _rate(count, n):
    return round(count / n, 4) if n else 0.0


def evaluate(candidate, live, exam, regression, catalog):
    reg_emb, reg_ids = regression
    exam_emb, exam_ids = exam
    replays, wrong_high = {}, {}
    for label, parts in (("after", candidate), ("before", live)):
        reg, ex = _replay(parts, reg_emb, reg_ids, catalog), _replay(parts, exam_emb, exam_ids, catalog)
        replays[label] = (reg, ex)
        wrong_high[label] = _rate(reg["wrongHigh"] + ex["wrongHigh"], len(reg_ids) + len(exam_ids))
    n_reg, n_exam = len(reg_ids), len(exam_ids)
    (reg_after, exam_after), (reg_before, exam_before) = replays["after"], replays["before"]
    improved = (
        n_reg > 0
        and better_outcome(_outcome(reg_before), _outcome(reg_after), reg_before["right"], reg_after["right"])
        and wrong_high["after"] <= wrong_high["before"]
        and (n_exam < MIN_EXAM_CASES or _no_worse(_outcome(exam_before), _outcome(exam_after), exam_before["right"], exam_after["right"]))
    )
    return {
        "improved": bool(improved),
        "regression": {"before": _rate(reg_before["right"], n_reg), "after": _rate(reg_after["right"], n_reg), "n": n_reg},
        "exam": {"before": _rate(exam_before["right"], n_exam), "after": _rate(exam_after["right"], n_exam), "n": n_exam},
        "wrongHigh": wrong_high,
        "outcomes": {
            "regression": {"before": _outcome(reg_before), "after": _outcome(reg_after)},
            "exam": {"before": _outcome(exam_before), "after": _outcome(exam_after)},
        },
    }
