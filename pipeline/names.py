"""Card base names and the one right/wrong rule for the regression replay.

base_name is a port of normalise / base_name in card-scanner-data/scripts/score_real_labels.py (the preregistered study);
tests/test_names.py runs both over a list of names. A regression row is labelled either with one exact catalog id (a str)
or with a NameTruth: a base name plus the shipped catalog ids that have it, resolved when the file is built.
"""
import re
from dataclasses import dataclass

VARIANT_TOKENS = frozenset({"ex", "v", "vmax", "vstar", "vunion", "gx", "star", "delta", "δ"})
MAX_ID_LEN = 64  # regression.npz stores ids as U64
MAX_NAME_LEN = 128


def normalise(name):
    s = name.lower().replace("’", "'")
    s = re.sub(r"'s\b", "", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def base_name(name):
    return " ".join(t for t in normalise(name).split() if t not in VARIANT_TOKENS)


@dataclass(frozen=True)
class NameTruth:
    """A name-labelled row: the label's base name and the shipped catalog ids whose base name equals it (may be empty:
    a name no shipped card has can only be read right through a new card)."""

    name: str
    accept: frozenset

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or len(self.name) > MAX_NAME_LEN:
            raise ValueError("a name-labelled row needs a base name of 1 to 128 characters")
        object.__setattr__(self, "accept", frozenset(self.accept))


def accept_lists(catalog_names, shipped_ids):
    """{base name: [shipped ids with that base name]} from {card id: printed name}. Only ids in shipped_ids count, in its
    order, so an accept list never holds a card the shipped catalog does not have."""
    by_base = {}
    for card_id in shipped_ids:
        printed = catalog_names.get(card_id)
        if printed:
            by_base.setdefault(base_name(printed), []).append(card_id)
    return by_base


def is_right(predicted, truth, new_card_names):
    """Whether the first hit `predicted` (a card id) is the right read for `truth`. An id truth is exact equality. A
    NameTruth is right when predicted is in its accept list, or is a new-card row (new_card_names: id -> printed name,
    tonight's or the live bundle's) whose own base name equals the label."""
    if isinstance(truth, str):
        return predicted == truth
    if predicted in truth.accept:
        return True
    printed = new_card_names.get(predicted)
    return printed is not None and base_name(printed) == truth.name
