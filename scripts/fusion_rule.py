"""Offline port of tools/extension-wasm/lib/fusion.js (accept decision, high tier, small-source caps).

Ported: fuseUntiered (clear_gap, name_match, high_sim, name_vote, low_sim, no_hits), the high-tier
test of fuseResults, and capSmallSource's height caps.

NOT ported: reference rows, adapter terms (capAdapterChange), title/OCR second opinion (e554499),
set choice / setChoices / versionChoices caps (capUnconfirmedSet), hub penalty (rank = score here),
page prior, gallery dedupe, markRepeatedNames.
Names: `names` is a list aligned with ids; None/"" disables the name paths for that hit only.
"""
from typing import NamedTuple, Optional, Sequence

# fusion.js line numbers in comments.
MIN_ACCEPT_SIM = 0.72   # L11
CLEAR_GAP = 0.02        # L12
HIGH_SIM = 0.85         # L13
NAME_MATCH_SIM = 0.75   # L14
HIGH_TIER_GAP = 0.025   # L26
HIGH_TIER_SIM = 0.82    # L27
BELOW_HIGH_CONFIDENCE = 0.79  # L29
NEAR_BAND = 0.02        # L111
MIN_RELIABLE_HEIGHT = 300     # L178
SMALL_SOURCE_CONFIDENCE_CAP = 0.49  # L180
MIN_NAME_HEIGHT = 200   # L186


class Read(NamedTuple):
    label: str            # 'sure' (identified), 'guess' (accepted but capped to suggestion), 'abstain'
    id: Optional[str]     # chosen id (top-1 id when abstaining)
    reason: str
    accepted: bool        # fuseUntiered accepted (view a)
    high: bool            # high tier before height caps
    confidence: float     # after high-tier and height caps


def _name(names: Optional[Sequence], i: int):
    return names[i] if names is not None and i < len(names) and names[i] else None


def _leading_name(sims, names):
    """Port of leadingName (L113-142); returns hit index or None."""
    best = sims[0]
    near = []
    for i, s in enumerate(sims):
        if best - s > NEAR_BAND:
            break
        near.append(i)
        if len(near) == 3:
            break
    counts = {}
    for i in near:
        n = _name(names, i)
        if not n:
            continue
        e = counts.setdefault(n, [0, i])
        e[0] += 1
        if sims[i] > sims[e[1]]:
            e[1] = i
    winner, tied = None, False
    for e in counts.values():
        if winner is None or e[0] > winner[0]:
            winner, tied = e, False
        elif e[0] == winner[0]:
            tied = True
    if winner is None or tied or winner[0] < 2:
        return None
    return winner[1]


def _fuse_untiered(sims, ids, names):
    """Returns (reason, chosen_index_or_None, confidence); None index = abstain."""
    if not len(sims):
        return "no_hits", None, 0.0
    top = sims[0]
    gap = sims[0] - sims[1] if len(sims) > 1 else top
    if top < MIN_ACCEPT_SIM:
        return "low_sim", None, top
    if gap >= CLEAR_GAP:
        return "clear_gap", 0, top
    n0 = _name(names, 0)
    if len(sims) > 1 and n0 and n0 == _name(names, 1) and top >= NAME_MATCH_SIM:
        return "name_match", 0, top * 0.9
    if top >= HIGH_SIM:
        return "high_sim", 0, top * 0.85
    agreed = _leading_name(sims, names)
    if agreed is not None:
        return "name_vote", agreed, sims[agreed] * 0.9
    return "ambiguous", None, top


def classify(sims_topk: Sequence[float], ids_topk: Sequence[str],
             names: Optional[Sequence] = None, height: int = 0) -> Read:
    """Classify one query. height=0 means unknown (no small-source cap)."""
    reason, idx, conf = _fuse_untiered(sims_topk, ids_topk, names)
    if idx is None:
        return Read("abstain", ids_topk[0] if len(ids_topk) else None, reason, False, False, conf)
    chosen = ids_topk[idx]
    high = False
    if conf > BELOW_HIGH_CONFIDENCE:  # fuseResults L46-49
        lead = sims_topk[0] - sims_topk[1] if len(sims_topk) > 1 else sims_topk[0]
        high = reason == "clear_gap" and sims_topk[0] >= HIGH_TIER_SIM and lead >= HIGH_TIER_GAP
        if not high:
            conf = BELOW_HIGH_CONFIDENCE
    pre_cap_high = high
    label = "sure"
    if 0 < height < MIN_RELIABLE_HEIGHT and conf > SMALL_SOURCE_CONFIDENCE_CAP:  # capSmallSource
        if height >= MIN_NAME_HEIGHT:
            conf = min(conf, BELOW_HIGH_CONFIDENCE)
        else:
            conf, label = SMALL_SOURCE_CONFIDENCE_CAP, "guess"
    return Read(label, chosen, reason, True, pre_cap_high, conf)
