"""EXAM scoring: does catalog + pack beat the plain catalog on photos the pack never saw?

Only contributions with split "exam" count, and only when every card in them was answered (none "unknown").
Cards answered "not a card" have no card id to score and are skipped without disqualifying the contribution.
Exam samples never enter promote(), which refuses non-teach samples, so exam crops cannot reach the pack.
"""
import numpy as np

from .contributions import build_samples

TOP_K = 20


def exam_cases(items, valid_ids):
    """[(Sample, truth card id)] for scoring."""
    answered = [
        it for it in items
        if it["split"] == "exam" and not any("unknown" in c["answer"] for c in it["cards"])
    ]
    cases = []
    for s in build_samples(answered, valid_ids=valid_ids, split="exam"):
        votes = [v for v in s.votes.values() if v.kind != "unknown"]
        ids = {v.card_id for v in votes if v.kind == "card"}
        if len(ids) == 1 and all(v.kind == "card" for v in votes):
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
