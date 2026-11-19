"""EXAM scoring: does catalog + pack beat the plain catalog on photos the pack never saw?

Only contributions with split "exam" count, and only when every card in them was answered (none "unknown").
Cards answered "not a card" have no card id to score and are skipped without disqualifying the contribution.
Exam samples never enter promote(), which refuses non-teach samples, so exam crops cannot reach the pack.
"""
import math

import numpy as np

from .contributions import build_samples
from .manifest import MIN_EXAM_CASES  # gate.py imports this module, so it cannot be read from there

TOP_K = 20
# The exam rule applies once MIN_EXAM_CASES cases exist; capping each install keeps one install from filling it alone.
EXAM_MIN_INSTALLS = 4
MAX_EXAM_CASES_PER_INSTALL = math.ceil(MIN_EXAM_CASES / EXAM_MIN_INSTALLS)


def exam_cases(items, valid_ids):
    """[(Sample, truth card id)] for scoring, at most MAX_EXAM_CASES_PER_INSTALL per install.

    Cases are taken in key order (photo id, box), so the same inbox always gives the same exam. A kept case counts
    against every install that voted on it, and a case is skipped only when all of its voters are already at the cap
    (so an install that co-votes on every case cannot push honest cases out of the exam, and a case always has an
    install with room, which keeps the total under cap x installs)."""
    answered = [
        it for it in items
        if it["split"] == "exam" and not any("unknown" in c["answer"] for c in it["cards"])
    ]
    cases = []
    taken = {}
    for s in sorted(build_samples(answered, valid_ids=valid_ids, split="exam"), key=lambda sample: sample.key):
        votes = [v for v in s.votes.values() if v.kind != "unknown"]
        ids = {v.card_id for v in votes if v.kind == "card"}
        if len(ids) == 1 and all(v.kind == "card" for v in votes):
            installs = {v.install for v in votes}
            if all(taken.get(i, 0) >= MAX_EXAM_CASES_PER_INSTALL for i in installs):
                continue
            for i in installs:
                taken[i] = taken.get(i, 0) + 1
            cases.append((s, next(iter(ids))))
    return cases


def is_high(scores, sim=0.82, gap=0.025):
    """Whether a catalog read earns the "high" stripe (lib/fusion.js): a clear lead and a high similarity.

    Reads with a tied or near-tied runner-up never qualify. Hub penalty and name rules are not modelled.
    """
    top = float(scores[0])
    lead = top - float(scores[1]) if len(scores) > 1 else top
    return top >= sim and lead >= gap


def score_exam(embeddings, truths, catalog, pack_ids, pack_vectors, ref_min=0.93, ref_margin=0.02, high_sim=0.82, high_gap=0.025):
    """Top-1 accuracy and wrong-at-high rate of the catalog alone ("before") and catalog + pack ("after").

    A pack row wins as in lib/references.js referenceHit and background.js withReferenceHit: the best row's
    cosine must reach ref_min and beat the catalog's top score by ref_margin, and its card must already be in
    the catalog's top 20. A winning pack card becomes the answer, tied with the catalog top, so it is never "high".
    pack_vectors must be L2-normalised floats (Pack.dequantized()).
    """
    n = len(truths)
    if n == 0:
        return {"before": 0.0, "after": 0.0, "n": 0, "wrongAtHigh": {"before": 0.0, "after": 0.0}}
    idx, scores = catalog.top_k_batch(embeddings, TOP_K)
    pack_vectors = np.asarray(pack_vectors, dtype=np.float32).reshape(-1, catalog.vectors.shape[1])
    right_before = right_after = wrong_high_before = wrong_high_after = 0
    for i, truth in enumerate(truths):
        top_ids = [catalog.ids[j] for j in idx[i]]
        predicted = top_ids[0]
        high = is_high(scores[i], high_sim, high_gap)
        right_before += predicted == truth
        wrong_high_before += high and predicted != truth
        if len(pack_ids):
            sims = pack_vectors @ np.asarray(embeddings[i], dtype=np.float32)
            best = int(np.argmax(sims))
            if sims[best] >= ref_min and sims[best] - scores[i][0] >= ref_margin and pack_ids[best] in top_ids:
                predicted, high = pack_ids[best], False
        right_after += predicted == truth
        wrong_high_after += high and predicted != truth
    return {
        "before": round(right_before / n, 4),
        "after": round(right_after / n, 4),
        "n": n,
        "wrongAtHigh": {"before": round(wrong_high_before / n, 4), "after": round(wrong_high_after / n, 4)},
    }
