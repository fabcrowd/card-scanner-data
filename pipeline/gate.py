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
from .names import is_right
from .new_sets import read_extra
from .pack import SCALE, Pack, read_pack
from .train_adapter import apply_adapter, parse_adapter

MIN_EXAM_CASES = 30
REF_MIN, REF_MARGIN = 0.93, 0.02  # exam.score_exam defaults (lib/references.js)
# Per-height confirm rule, mirrored from lib/fusion.js (tests/test_drift.py checks the values). A read is accepted iff
# height >= CONFIRM_MIN_HEIGHT and top >= T(height) and top - best different-name card >= G(height). Rows are
# (min_height, T, G), tallest first; the first row whose min_height the height reaches applies.
# Regression and exam rows carry no read height (only embeddings and card ids), so every replayed row takes the
# 200 row (UNKNOWN_HEIGHT_ROW, as lib/fusion.js does for an unknown height: thumbnails are about 200 px and the
# native row's G=0 was fitted on large crops); no height is invented. Nightly rows carry no
# title line either, so TITLE_NAME_PENALTY (the extension's demotion for a contradicting title) is mirrored but never applied.
# Rows are the floor-analysis values (fusion-study-E/floor/README.md, 2026-10-23). NOT mirrored, on purpose: SMALL_ROWS
# (100-199 px, flagged smallRow) and the price rules of lib/stakes.js (promote at <= $5, demote at >= $50). Nightly rows
# carry neither a read height nor a price, so neither rule can be replayed here.
CONFIRM_MIN_HEIGHT = 200
TITLE_NAME_PENALTY = 0.05
CONFIRM_ROWS = ((350, 0.76, 0.02), (300, 0.74, 0.02), (250, 0.72, 0.03), (200, 0.68, 0.03))
UNKNOWN_HEIGHT_ROW = next(r for r in CONFIRM_ROWS if r[0] == CONFIRM_MIN_HEIGHT)
CONFIRM_EPSILON = 1e-9  # float slack so a value exactly at T or G passes
MAX_ABSTAIN_RISE = 0.02  # abstain rate may not rise by more than 2 points


def _with_extra(catalog, part):
    """The catalog plus the rows of a catalogExtra.bin (None: the catalog itself) and {id: printed name} of those rows. The
    rows are dequantised like a memory pack's. ValueError when the file is malformed."""
    if part is None:
        return catalog, {}
    cards, rows = read_extra(part)
    known = set(catalog.ids)
    keep = [i for i, c in enumerate(cards) if c["id"] not in known]  # the extension drops rows whose id the catalog has
    cards, rows = [cards[i] for i in keep], rows[keep]
    if not cards:
        return catalog, {}
    vectors = rows.astype(np.float32) / SCALE
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    names = None if catalog.names is None else catalog.names + [str(c.get("name", c["id"])) for c in cards]
    ids = [c["id"] for c in cards]
    return Catalog(np.vstack([catalog.vectors, vectors]), catalog.ids + ids, names), {c["id"]: str(c.get("name", c["id"])) for c in cards}


def _memory(part):
    if part is None:
        return [], np.zeros((0, 512), dtype=np.float32)
    pack = part if isinstance(part, Pack) else read_pack(part)
    return pack.card_ids, pack.dequantized()


def confirms(top, margin, height=0):
    """lib/fusion.js confirmDecision without a title: height 0 (unknown) uses the 200 row, a known height under
    CONFIRM_MIN_HEIGHT never confirms."""
    if height > 0 and height < CONFIRM_MIN_HEIGHT:
        return False
    _, min_sim, min_gap = UNKNOWN_HEIGHT_ROW if height <= 0 else next(r for r in CONFIRM_ROWS if height >= r[0])
    return top + CONFIRM_EPSILON >= min_sim and margin + CONFIRM_EPSILON >= min_gap


def rival_margin(ranks, ids, names=None):
    """Top rank minus the best card with ANOTHER name (same-name printings never block). `ranks` is best first and
    `ids` the card ids in the same order; names=None treats every card as a different name. inf when no rival is left,
    as in fusion.js (-Infinity rival)."""
    if len(ranks) == 0:
        return float("inf")
    top_key = ids[0] if names is None else names[0]
    for rank, key in zip(ranks[1:], ids[1:] if names is None else names[1:]):
        if key != top_key:
            return float(ranks[0] - rank)
    return float("inf")


def _replay(parts, embeddings, truths, catalog, new_card_names=None):
    """Counts for one bundle: top-1 right, wrong "high" reads, and the accepted-right / accepted-wrong / abstained split.
    A truth is a card id or a names.NameTruth; names.is_right reads both (new_card_names: id -> printed name of the new-card rows)."""
    new_card_names = new_card_names or {}
    counts = {"right": 0, "wrongHigh": 0, "acceptedRight": 0, "acceptedWrong": 0, "abstained": 0, "n": len(truths), "predicted": [], "highWrong": []}
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
    by_id = None if catalog.names is None else dict(zip(catalog.ids, catalog.names))
    for i, truth in enumerate(truths):
        top_ids = [catalog.ids[j] for j in idx[i]]
        predicted = top_ids[0]
        row_names = None if by_id is None else [by_id[c] for c in top_ids]
        gap = rival_margin(ranks[i], top_ids, row_names)
        high = is_high(raw_scores[i]) and predicted == catalog.ids[raw_idx[i][0]]
        if len(mem_ids):
            sims = mem_vecs @ embeddings[i]
            best = int(np.argmax(sims))
            if sims[best] >= REF_MIN and sims[best] - raw_scores[i][0] >= REF_MARGIN and mem_ids[best] in top_ids:
                predicted, high, gap = mem_ids[best], False, 0.0  # promoted at the top score: a tie with the old top
        counts["predicted"].append(predicted)
        right = is_right(predicted, truth, new_card_names)
        counts["right"] += right
        counts["wrongHigh"] += high and not right
        counts["highWrong"].append(bool(high and not right))
        if confirms(float(scores[i][0]), gap):
            counts["acceptedRight" if right else "acceptedWrong"] += 1
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
    replays, wrong_high, extra_ids = {}, {}, {}
    for label, parts in (("after", candidate), ("before", live)):
        try:
            searched, extra_ids[label] = _with_extra(catalog, parts.get("catalogExtra"))
        except ValueError:
            if label == "after":
                raise
            print("live catalogExtra is unreadable: replayed without its rows")
            searched, extra_ids[label] = catalog, {}
        reg, ex = _replay(parts, reg_emb, reg_ids, searched, extra_ids[label]), _replay(parts, exam_emb, exam_ids, searched)
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
    was_right = [is_right(p, t, extra_ids["before"]) for p, t in zip(reg_before["predicted"], reg_ids)]
    now_right = [is_right(p, t, extra_ids["after"]) for p, t in zip(reg_after["predicted"], reg_ids)]
    new_cards = {
        "firstHit": {
            "before": sum(p in extra_ids["before"] for p in reg_before["predicted"]),
            "after": sum(p in extra_ids["after"] for p in reg_after["predicted"]),
        },
        "lost": sum(b and not a for b, a in zip(was_right, now_right)),
        "gained": sum(a and not b for b, a in zip(was_right, now_right)),
    }
    # The rows behind a failing measure, for run.py's holding rule (never written to the manifest): (row index, first-hit id
    # after, reason). "confidentWrong": a confident wrong read that was not one before, listed only when the confident-wrong
    # rate rose. "lostRead": right before and wrong after, listed only when the regression rate fell.
    rate_fell = _rate(reg_after["right"], n_reg) < _rate(reg_before["right"], n_reg)
    high_rose = wrong_high["after"] > wrong_high["before"]
    blame = []
    for i, (was, now, high_before, high_after) in enumerate(zip(was_right, now_right, reg_before["highWrong"], reg_after["highWrong"])):
        if high_rose and high_after and not high_before:
            blame.append((i, reg_after["predicted"][i], "confidentWrong"))
        if rate_fell and was and not now:
            blame.append((i, reg_after["predicted"][i], "lostRead"))
    return {
        "improved": bool(improved),
        "blame": blame,
        # wrongHigh below is one rate over both sets; these are the confident-wrong counts of each set on its own, so a rise
        # on the exam cannot be hidden by a regression row that a held card brings back.
        "confidentWrong": {
            "regression": {"before": int(reg_before["wrongHigh"]), "after": int(reg_after["wrongHigh"])},
            "exam": {"before": int(exam_before["wrongHigh"]), "after": int(exam_after["wrongHigh"])},
        },
        "newCards": new_cards,
        "regression": {"before": _rate(reg_before["right"], n_reg), "after": _rate(reg_after["right"], n_reg), "n": n_reg},
        "exam": {"before": _rate(exam_before["right"], n_exam), "after": _rate(exam_after["right"], n_exam), "n": n_exam},
        "wrongHigh": wrong_high,
        "outcomes": {
            "regression": {"before": _outcome(reg_before), "after": _outcome(reg_after)},
            "exam": {"before": _outcome(exam_before), "after": _outcome(exam_after)},
        },
    }
