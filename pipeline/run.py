"""Nightly pipeline: contributions/ -> the bundle parts + manifest.json (unsigned) in --out.

    python -m pipeline.run --data data --assets assets --out out --base-url <url ending in /> --seq <run number>

Parts: community-pack.bin (memory) always; adapter.bin, check.bin, catalog-extra.bin when their trainer or the new-sets
step produced one. The trainers, seed, gate and status modules are optional: while one is missing its part is simply
left out and the gate falls back to the memory-only rule, so the pipeline still builds a memory-only bundle.

This step holds no secrets and signs nothing: the separate `sign` job verifies the output and signs the manifest
(scripts/sign_release.py). --data is only read as JSON files; nothing in it is executed or imported.
"""
import argparse
import datetime
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import sys

import numpy as np

from .catalog import Catalog
from .consensus import promote
from .contributions import build_samples, count_box_signals, load_contributions, without_box_signals
from .embed import ClipEmbedder
from .exam import TOP_K, exam_cases, score_exam
from .manifest import ASSET_FILES, MIN_EXAM_CASES, build_manifest, is_improved, manifest_bytes
from .new_sets import extra_tag, read_extra, write_extra
from .pack import read_pack, write_pack
from .regression import load_regression

OPTIONAL_MODULES = ("train_adapter", "train_check", "gate", "status")
# A night that ships nothing republishes the live bundle once its manifest is older than this. The extension drops a
# manifest older than 30 days (MAX_MANIFEST_AGE_MS in tools/extension-wasm/lib/bundle.js; tests/test_drift.py keeps the margin).
REISSUE_AFTER_DAYS = 14
# The learned parts a night can change; catalogExtra is separate because it adds cards, not learning.
LEARNED = ("memory", "adapter")
MIN_NOT_A_CARD_INSTALLS = 2
# The holding rule for a night that only adds cards (see _plan_hold): at most this many hold-and-retest rounds, and at
# most this many held cards in total.
HOLD_MAX_ROUNDS = 3
HOLD_MAX_CARDS = 10
HOLD_REASONS = {"confidentWrong": "confident wrong read", "lostRead": "lost a correct read"}


def optional_modules():
    """{name: module} for each optional pipeline module that exists. A module that exists but fails to import is an error."""
    found = {}
    for name in OPTIONAL_MODULES:
        full = f"{__package__}.{name}"
        try:
            found[name] = importlib.import_module(full)
        except ModuleNotFoundError as err:
            if err.name != full:
                raise
    return found


def _embed(samples, embedder):
    """Samples whose crop embeds, with their vectors; the rest are dropped and counted."""
    kept, vectors = [], []
    for s in samples:
        try:
            vectors.append(np.asarray(embedder(s.crop), dtype=np.float32))
            kept.append(s)
        except ValueError:
            continue
    dim = vectors[0].shape[0] if vectors else 512
    return kept, (np.stack(vectors) if vectors else np.zeros((0, dim), dtype=np.float32)), len(samples) - len(kept)


def _not_a_card(samples, embeddings):
    """Embeddings of crops that two or more installs called "not a card" and nobody called a card. One install alone
    cannot teach the check what is not a card."""
    keep = [
        i for i, s in enumerate(samples)
        if not s.card_votes and sum(v.kind == "notACard" for v in s.votes.values()) >= MIN_NOT_A_CARD_INSTALLS
    ]
    return embeddings[keep] if keep else np.zeros((0, embeddings.shape[1]), dtype=np.float32)


def _scores(block):
    return {k: block[k] for k in ("before", "after", "n")}


def _issue_stamp(clock):
    """The manifest's issuedAt: the clock's time in UTC, as YYYY-MM-DDTHH:MM:SSZ. Read when the manifest is made."""
    now = clock() if clock else datetime.datetime.now(datetime.timezone.utc)
    return now.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _signer_improved_rule(path=None):
    """The signer's own check that `improved: true` follows from the numbers, loaded from scripts/sign_release.py so the
    build proposes only what the signer can accept (one copy of the rule). None when the file cannot be loaded, for any
    reason: a broken signer file must mean "do not re-issue", not a failed night. `path` is for tests."""
    path = path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "sign_release.py")
    try:
        spec = importlib.util.spec_from_file_location("sign_release_rule", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module._improved_is_supported
    except Exception as err:  # noqa: BLE001 - executing the file can raise anything (SyntaxError included)
        print(f"the signer's rule could not be loaded ({type(err).__name__}: {err}): not re-issuing")
        return None


def _reissue_parts(live_manifest, live, pins, clip_hash, issued_at):
    """The live bundle's parts {name: bytes} when tonight should publish it again, else None.

    That needs a live manifest that is improved (and whose numbers the signer's rule backs, or the signer would
    refuse the re-issue), made for this recognizer model and crop, older than REISSUE_AFTER_DAYS, and whose listed
    parts are all in `live` byte for byte (sha256 and size as the manifest says)."""
    if not isinstance(live_manifest, dict) or live_manifest.get("improved") is not True:
        return None
    if live_manifest.get("model") != clip_hash or live_manifest.get("crop") != pins["cropVersion"]:
        return None
    try:
        made = datetime.datetime.strptime(live_manifest["issuedAt"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)
        assets = live_manifest["assets"]
        now = datetime.datetime.strptime(issued_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)
        if now - made <= datetime.timedelta(days=REISSUE_AFTER_DAYS) or not assets:
            return None
        rule = _signer_improved_rule()
        if rule is None:
            return None
        if not rule(live_manifest["exam"], live_manifest["regression"]):
            print("live bundle is due for a re-issue, but its numbers do not back improved under the signer's rule: not re-issuing")
            return None
        parts = {}
        for name, asset in assets.items():
            data = live.get(name)
            if data is None or hashlib.sha256(data).hexdigest() != asset["sha256"] or len(data) != asset["bytes"]:
                print(f"live bundle is due for a re-issue, but its {name} file does not match its manifest: not re-issuing")
                return None
            parts[name] = data
    except (KeyError, TypeError, ValueError):
        print("live manifest could not be read for a re-issue: not re-issuing")
        return None
    return parts


def run_pipeline(data_dir, out_dir, pins, embedder, catalog, clip_hash, base_url, seq, clock=None, regression=None,
                 modules=None, live=None, seed_pairs=None, new_sets_fn=None, status_path=None, status_history=None, live_manifest=None,
                 _hold=None):
    """Runs every stage and writes the parts and the unsigned manifest. Returns a summary dict.

    clock: () -> aware datetime; read once, right before the manifest is made (after the long stages), for issuedAt.
        Default: now in UTC.
    regression: (embeddings, card ids) of the fixed server-owned set, or None. Without it the manifest can never
        say improved.
    modules: {name: module} of the optional pipeline modules (default: those that exist).
    live: {"memory": bytes, "adapter": bytes} of the live bundle, which the gate compares against. Missing means
        the plain catalog.
    live_manifest: the live bundle's parsed manifest (load_live_manifest), or None. When the night ships nothing and
        the live manifest is older than REISSUE_AFTER_DAYS, the output is the live bundle again: the same part files and
        a manifest equal to the live one except seq and issuedAt (the sign job re-checks this against the release).
    seed_pairs: [(embedding, card id)] from seed.npz (scripts/build_seed.py), or None for no seed.
    new_sets_fn: () -> new_sets.build result, or None to add no cards.
    status_path / status_history: where STATUS.md is written, and the earlier status.json to carry forward.
    _hold: internal, set only when this call retests a candidate with some new cards held back ({"round", "held", "probe"});
        such a call writes nothing and returns None unless the reduced candidate ships.
    """
    modules = optional_modules() if modules is None else modules
    live = live or {}
    hold = _hold or {"round": 0, "held": [], "probe": False}
    items, dropped = load_contributions(data_dir)
    box_signals = count_box_signals(items)
    items = without_box_signals(items)
    new_cards = new_sets_fn() if new_sets_fn else None
    valid_ids = set(catalog.ids) | _extra_ids(new_cards, live.get("catalogExtra"))

    teach, teach_emb, teach_failed = _embed(build_samples(items, valid_ids=valid_ids, split="teach"), embedder)
    if len(teach):
        idx, _ = catalog.top_k_batch(teach_emb, TOP_K)
        neighbours = [{catalog.ids[j] for j in row} for row in idx]
    else:
        neighbours = []
    promotion = promote(teach, teach_emb, neighbours)
    promoted_vectors = np.stack([r.embedding for r in promotion.rows]) if promotion.rows else np.zeros((0, 512), dtype=np.float32)
    pack_bytes = write_pack([r.card_id for r in promotion.rows], promoted_vectors)
    pack = read_pack(pack_bytes)
    parts = {"memory": pack_bytes}

    pairs = [(r.embedding, r.card_id) for r in promotion.rows] + list(seed_pairs or [])
    if "train_adapter" in modules:
        adapter = modules["train_adapter"].train(pairs, catalog)
        if adapter:
            parts["adapter"] = adapter
    if "train_check" in modules:
        check = modules["train_check"].train(promoted_vectors, _not_a_card(teach, teach_emb))
        if check:
            parts["check"] = check

    if new_cards and new_cards.get("bytes"):
        parts["catalogExtra"] = new_cards["bytes"]

    cases = exam_cases(items, valid_ids)
    exam_samples, exam_emb, exam_failed = _embed([s for s, _ in cases], embedder)
    truths = {s.key: t for s, t in cases}
    exam_truths = [truths[s.key] for s in exam_samples]
    has_regression = regression is not None and len(regression[1]) > 0
    outcomes = None

    if "gate" in modules:
        empty = (np.zeros((0, 512), dtype=np.float32), [])
        # catalogExtra goes in too: the replay searches the catalog plus these rows, before (live's) and after (tonight's).
        verdict = modules["gate"].evaluate(
            {k: parts[k] for k in (*LEARNED, "catalogExtra") if k in parts}, live, (exam_emb, exam_truths), regression if has_regression else empty,
            catalog=catalog,
        )
        replay = verdict.get("newCards")
        if replay and has_regression:
            print(
                f"new-card rows are the first hit on {replay['firstHit']['before']} of {regression_result_n(verdict)} regression rows before and "
                f"{replay['firstHit']['after']} after; correct reads lost {replay['lost']}, gained {replay['gained']}"
            )
        exam = {**_scores(verdict["exam"]), "wrongAtHigh": verdict["wrongHigh"]}
        regression_result = _scores(verdict["regression"]) if has_regression else None
        improved = bool(verdict["improved"])
        outcomes = verdict.get("outcomes")
        # A night that only adds cards changes what the learned parts do not. It may ship when what was learned is
        # byte-identical to the live bundle and the replay over the catalog plus the new-card rows (gate.evaluate) shows
        # no fewer correct reads and no more confident wrong reads than before. An empty memory pack counts as
        # equal to no live memory part (the first night after a model change, when the old live bundle was ignored).
        if not improved and new_cards and new_cards["added"] and regression_result and regression_result["n"] > 0:
            same = all(
                parts.get(k) == live.get(k) or (k == "memory" and k not in live and not pack.card_ids)
                for k in LEARNED
            )
            improved = (
                same
                and regression_result["after"] >= regression_result["before"]
                and verdict["wrongHigh"]["after"] <= verdict["wrongHigh"]["before"]
                # the signer (scripts/sign_release.py _improved_is_supported) refuses an exam drop once the exam has enough cases
                and not (exam["n"] >= MIN_EXAM_CASES and exam["after"] < exam["before"])
            )
            blamed = _plan_hold(verdict, regression_result, exam, new_cards, hold) if same and not improved else None
            if blamed:
                held_now = _held_records(new_cards, blamed)
                final = run_pipeline(
                    data_dir, out_dir, pins, embedder, catalog, clip_hash, base_url, seq, clock=clock, regression=regression,
                    modules=modules, live=live, seed_pairs=seed_pairs, new_sets_fn=lambda: _reduced_new_cards(new_cards, {h["id"] for h in held_now}),
                    status_path=status_path, status_history=status_history, live_manifest=live_manifest,
                    _hold={"round": hold["round"] + 1, "held": hold["held"] + held_now, "probe": True},
                )
                if final is not None:
                    return final
                if not hold["probe"]:
                    print("holding back the blamed new cards did not make the night pass: nothing is held and nothing ships")
    else:
        exam = score_exam(
            exam_emb, exam_truths, catalog, pack.card_ids, pack.dequantized(),
            ref_min=pins["reference"]["min"], ref_margin=pins["reference"]["margin"],
            high_sim=pins["highTier"]["sim"], high_gap=pins["highTier"]["gap"],
        )
        regression_result = None
        if has_regression:
            scored = score_exam(
                regression[0], regression[1], catalog, pack.card_ids, pack.dequantized(),
                ref_min=pins["reference"]["min"], ref_margin=pins["reference"]["margin"],
                high_sim=pins["highTier"]["sim"], high_gap=pins["highTier"]["gap"],
            )
            regression_result = _scores(scored)
        improved = is_improved(exam, regression_result)

    if improved and _shrinks(live.get("catalogExtra"), parts.get("catalogExtra"), new_cards):  # dropped rows are accounted for inside
        improved = False
    if hold["probe"] and improved:
        # A candidate trimmed by holding must also keep the exam's own confident-wrong count at or below "before": the combined
        # count can pass because a regression row got better while the exam got worse.
        exam_wrong = _exam_confident_wrong(verdict)
        if exam_wrong is None or exam_wrong[1] > exam_wrong[0]:
            print("the trimmed candidate has more confident wrong reads on the exam than before: nothing ships")
            improved = False
    if hold["probe"] and not improved:
        return None  # a retest that does not ship leaves no trace: the night is reported as the unreduced candidate
    if hold["held"]:
        for h in hold["held"]:
            print(f"held back {h['id']} ({h['name']}, set {h['set']}): {h['reason']}; blamed by {h['rows']} regression rows")
        print(f"held back {len(hold['held'])} new cards in {hold['round']} rounds; {len(new_cards['added'])} new cards ship; held cards are tried again tomorrow")

    issued_at = _issue_stamp(clock)
    manifest = build_manifest(pins, clip_hash, parts, base_url, exam, regression_result, improved, seq, issued_at)
    reissued = None if manifest["improved"] else _reissue_parts(live_manifest, live, pins, clip_hash, issued_at)
    if reissued is not None:
        print(f"nothing ships tonight and the live bundle is older than {REISSUE_AFTER_DAYS} days: publishing it again as seq {seq}")
        manifest = {**live_manifest, "seq": seq, "issuedAt": issued_at}
        out_parts = reissued
    else:
        out_parts = parts
    raw = manifest_bytes(manifest)

    os.makedirs(out_dir, exist_ok=True)
    for name, data in out_parts.items():
        with open(os.path.join(out_dir, ASSET_FILES[name]), "wb") as f:
            f.write(data)
    with open(os.path.join(out_dir, "manifest.json"), "wb") as f:
        f.write(raw)

    summary = {
        "entries": len(pack.card_ids),
        "parts": sorted(out_parts),
        "improved": bool(improved),
        "reissue": reissued is not None,
        "exam": exam,
        "regression": regression_result,
        "outcomes": outcomes,
        "newCards": len(new_cards["added"]) if new_cards else 0,
        "newSets": new_cards["sets"] if new_cards else [],
        "contributionsDropped": dropped,
        "boxSignals": box_signals,
        "embedFailed": teach_failed + exam_failed,
        **promotion.stats,
    }
    if hold["held"]:
        summary["held"] = [{k: h[k] for k in ("id", "name", "set", "reason", "rows")} for h in hold["held"]]
    if "status" in modules and status_path:
        if status_history and os.path.exists(status_history):
            shutil.copyfile(status_history, os.path.join(os.path.dirname(os.path.abspath(status_path)), "status.json"))
        modules["status"].write(status_path, {
            "date": issued_at[:10],
            "seq": seq,
            "contributors": len({it["install"] for it in items}),
            "promoted": len(promotion.rows),
            "newSets": summary["newSets"],
            "regression": regression_result and {k: regression_result[k] for k in ("before", "after")},
            "exam": {k: exam[k] for k in ("before", "after", "n")},
            "shipped": bool(improved),
            **({"held": [{k: h[k] for k in ("id", "name", "reason")} for h in hold["held"]]} if hold["held"] else {}),
        })
    return summary


def regression_result_n(verdict):
    return verdict["regression"]["n"]


def _exam_confident_wrong(verdict):
    """(before, after) confident-wrong counts of the exam alone, from gate.evaluate's verdict; None when the verdict does not
    carry two non-negative integers there (the holding rule then fails closed)."""
    try:
        counts = verdict["confidentWrong"]["exam"]
        before, after = counts["before"], counts["after"]
    except (KeyError, TypeError):
        return None
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (before, after)):
        return None
    return before, after


def _plan_hold(verdict, regression_result, exam, new_cards, hold):
    """For a cards-only night that failed the replay: {card id: {"rows": set of regression rows, "reasons": [..]}} of the new
    cards to hold back, or None when nothing may be held (a message says why).

    Only the regression rate (after < before) and the confident-wrong count (after > before) can be blamed; a failing exam
    holds nothing. Every blamed row's first hit must be one of tonight's new cards (gate.evaluate "blame": (row, first-hit
    id, reason) for the rows behind a failing measure). At most HOLD_MAX_ROUNDS rounds, at most HOLD_MAX_CARDS held cards in
    total, and at least one new card must be left."""
    rate_fail = regression_result["after"] < regression_result["before"]
    high_fail = verdict["wrongHigh"]["after"] > verdict["wrongHigh"]["before"]
    if (exam["n"] >= MIN_EXAM_CASES and exam["after"] < exam["before"]) or not (rate_fail or high_fail):
        return None
    exam_wrong = _exam_confident_wrong(verdict)
    if exam_wrong is None:
        print("not holding any new card: the gate did not report the exam's confident wrong reads")
        return None
    if exam_wrong[1] > exam_wrong[0]:
        print("not holding any new card: the exam itself has more confident wrong reads, which holding a card cannot undo")
        return None
    blame = verdict.get("blame")
    if not blame:
        print("not holding any new card: no regression row is blamed on one")
        return None
    tonight = set(new_cards["added"])
    stray = sorted({card for _, card, _ in blame if card not in tonight})
    if stray:
        print(f"not holding any new card: a failing row's first hit is not one of tonight's new cards ({', '.join(stray[:5])})")
        return None
    if hold["round"] >= HOLD_MAX_ROUNDS:
        print(f"not holding any new card: still failing after {HOLD_MAX_ROUNDS} rounds")
        return None
    by_card = {}
    for row, card, reason in blame:
        entry = by_card.setdefault(card, {"rows": set(), "reasons": []})
        entry["rows"].add(row)
        if reason not in entry["reasons"]:
            entry["reasons"].append(reason)
    if len(hold["held"]) + len(by_card) > HOLD_MAX_CARDS:
        print(f"not holding any new card: that would hold {len(hold['held']) + len(by_card)} cards, the limit is {HOLD_MAX_CARDS}")
        return None
    if len(tonight) - len(by_card) < 1:
        print("not holding any new card: no new card would be left")
        return None
    return by_card


def _held_records(new_cards, blamed):
    """[{id, name, set, reason, rows}] for the blamed card ids, from tonight's catalogExtra rows."""
    by_id = {c["id"]: c for c in read_extra(new_cards["bytes"])[0]}
    return [
        {"id": cid, "name": by_id[cid].get("name", cid), "set": by_id[cid].get("set", ""),
         "reason": " and ".join(HOLD_REASONS[r] for r in entry["reasons"]), "rows": len(entry["rows"])}
        for cid, entry in sorted(blamed.items())
    ]


def _reduced_new_cards(new_cards, drop):
    """new_sets.build's result without the cards in `drop`: the part rebuilt from the remaining rows (same tag), and
    "added", "sets" (of the added cards that remain) and "total" agreeing with it."""
    cards, rows = read_extra(new_cards["bytes"])
    keep = [i for i, c in enumerate(cards) if c["id"] not in drop]
    kept = [cards[i] for i in keep]
    model, crop = extra_tag(new_cards["bytes"])
    added = [i for i in new_cards["added"] if i not in drop]
    return {
        **new_cards,
        "bytes": write_extra(kept, rows[keep], None if model is None or crop is None else {"model": model, "crop": crop}),
        "added": added,
        "sets": sorted({c["set"] for c in kept if c["id"] in set(added)}),
        "total": len(kept),
    }


def _extra_count(data):
    """Cards in a catalogExtra.bin, or None when it cannot be read."""
    try:
        return len(read_extra(data)[0])
    except (ValueError, TypeError):
        return None


def _extra_ids(new_cards, live_extra):
    """Ids of the new-set cards this run knows: tonight's file when there is one, else the live one, plus the ids tonight added.
    Votes for them are valid, so a correction to a new card is counted instead of silently vanishing."""
    data = (new_cards or {}).get("bytes") or live_extra
    ids = set((new_cards or {}).get("added") or [])
    if data:
        try:
            ids |= {c["id"] for c in read_extra(data)[0]}
        except (ValueError, TypeError):
            print("the new-card file could not be read, so votes for new cards are not counted tonight")
    return ids


def _shrinks(live_extra, candidate_extra, new_cards):
    """True (with a log line) when the live bundle is ours and has catalogExtra cards the candidate does not keep. `live` is
    {} when the live bundle was ignored (another model or crop), so that night is exempt and never gets here."""
    if not live_extra:
        return False
    before = _extra_count(live_extra)
    if before is None:
        print("not shipping: the live catalogExtra is unreadable, so tonight's file cannot be shown to keep its cards")
        return True
    after = (new_cards or {}).get("total")
    if after is None:
        after = _extra_count(candidate_extra) if candidate_extra else 0
    # Rows `build` dropped on purpose tonight (id now in the catalog, failed re-validation) are not a shrink.
    allowed = before - ((new_cards or {}).get("dropped") or 0)
    if after is None or after < allowed:
        print(f"not shipping: the new-card file has {after or 0} cards, the live one {before} ({before - allowed} dropped on purpose): a night must not drop cards")
        return True
    return False


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


MAX_SEED_ROWS = 400000


def load_seed(path):
    """[(float32[512] unit embedding, card id)] from seed.npz, or [] when the file is missing. The file comes from the
    data repo's main branch (owner-committed), is read without pickle, and a malformed one stops the run."""
    if not path or not os.path.exists(path):
        return []
    with np.load(path, allow_pickle=False) as z:
        vectors, ids = np.asarray(z["embeddings"], dtype=np.float32), z["cardIds"]
    if vectors.ndim != 2 or vectors.shape[1] != 512 or len(ids) != len(vectors) or len(ids) > MAX_SEED_ROWS or not np.all(np.isfinite(vectors)):
        raise ValueError(f"{path} is not a valid seed file")
    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    return [(vectors[i], str(ids[i])) for i in range(len(ids))]


def _is_tag(stored_model, stored_crop, model, crop):
    """A single string equal to `model` and a single integer equal to `crop` (a float or a string is not a match)."""
    return (
        stored_model.shape == () and stored_model.dtype.kind == "U" and str(stored_model) == model
        and stored_crop.shape == () and stored_crop.dtype.kind in "iu" and int(stored_crop) == crop
    )


def model_tag_ok(path, model, crop):
    """True when the npz at `path` carries a `model` and `crop` equal to ours. Otherwise says so and returns False, so
    a regression set or seed made with another recognizer (or an untagged one) is never used."""
    try:
        with np.load(path, allow_pickle=False) as z:
            ok = "model" in z.files and "crop" in z.files and _is_tag(z["model"], z["crop"], model, crop)
    except Exception:  # noqa: BLE001 - a corrupt file or a pickled object array is "not ours", never a crash
        ok = False
    if not ok:
        print(f"{path} was made with another model (or is untagged): ignored")
    return ok


def load_live(folder, model, crop):
    """The live bundle's parts that exist in `folder` (its release assets), as {name: bytes}. A bundle whose manifest is
    unreadable or names another model or crop is ignored ({}), so nothing made with an old model is carried over."""
    if folder and os.path.isdir(folder):
        try:
            with open(os.path.join(folder, "manifest.json"), encoding="utf-8") as f:
                live_manifest = json.load(f)
            live_model, live_crop = live_manifest["model"], live_manifest["crop"]
        except (OSError, ValueError, KeyError, TypeError):
            print("live bundle has no readable manifest: starting from the plain catalog")
            return {}
        if live_model != model or live_crop != crop:
            print(f"live bundle is model {live_model} crop {live_crop}, not ours: starting from the plain catalog")
            return {}
    live = {}
    for name, filename in ASSET_FILES.items():
        path = os.path.join(folder, filename) if folder else None
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                live[name] = f.read()
    return live


def load_live_manifest(folder, model, crop):
    """The live bundle's parsed manifest.json from `folder`, or None when it is missing, unreadable, or for another model
    or crop (the same bundles load_live ignores). The sign job re-checks it against the published release; this copy
    only decides whether a re-issue is worth proposing."""
    if not folder or not os.path.isdir(folder):
        return None
    try:
        with open(os.path.join(folder, "manifest.json"), encoding="utf-8") as f:
            live_manifest = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(live_manifest, dict) or live_manifest.get("model") != model or live_manifest.get("crop") != crop:
        return None
    return live_manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=".", help="data repo root (holds contributions/)")
    ap.add_argument("--assets", default="assets", help="folder written by scripts/fetch_assets.py")
    ap.add_argument("--out", default="out")
    ap.add_argument("--pins", default="pins.json")
    ap.add_argument("--base-url", required=True, help="where the extension will download the parts; ends with /")
    ap.add_argument("--seq", type=int, required=True, help="the workflow run number; only ever goes up")
    ap.add_argument("--regression", default="regression.npz", help="fixed regression set (scripts/build_regression.py); missing means never improved")
    ap.add_argument("--live-dir", default="live", help="the live release's assets, for the gate and to extend catalog-extra.bin; may be missing")
    ap.add_argument("--status-history", default="live-status/status.json", help="earlier status.json to carry forward; may be missing")
    ap.add_argument("--seed", default="seed.npz", help="seed embeddings from scripts/build_seed.py, committed to main; may be missing")
    ap.add_argument("--no-new-sets", action="store_true", help="skip the card-source requests")
    args = ap.parse_args(argv)
    if not args.base_url.endswith("/"):
        sys.exit("--base-url must end with /")

    with open(args.pins, encoding="utf-8") as f:
        pins = json.load(f)
    paths = {a["name"]: os.path.join(args.assets, a["name"]) for a in pins["assets"]}
    for a in pins["assets"]:
        if _sha256_file(paths[a["name"]]) != a["sha256"]:
            sys.exit(f"{a['name']} does not match its pinned sha256; refusing to run")
    hashes = {a["name"]: a["sha256"][:16] for a in pins["assets"]}

    embedder = ClipEmbedder(paths["clip_visual_int8.onnx"])
    catalog = Catalog.load(paths["catalog.bin.gz"], paths["id_map.json"])
    model, crop = hashes["clip_visual_int8.onnx"], pins["cropVersion"]
    live = load_live(args.live_dir, model, crop)
    new_sets_fn = None
    if not args.no_new_sets:
        from . import new_sets

        new_sets_fn = lambda: new_sets.build(catalog.ids, live.get("catalogExtra"), embedder, tag={"model": model, "crop": crop})  # noqa: E731

    os.makedirs(args.out, exist_ok=True)
    summary = run_pipeline(
        data_dir=args.data, out_dir=args.out, pins=pins, embedder=embedder, catalog=catalog,
        clip_hash=hashes["clip_visual_int8.onnx"], base_url=args.base_url, seq=args.seq,
        live_manifest=load_live_manifest(args.live_dir, model, crop),
        regression=load_regression(args.regression) if os.path.exists(args.regression) and model_tag_ok(args.regression, model, crop) else None,
        live=live, new_sets_fn=new_sets_fn,
        seed_pairs=load_seed(args.seed) if os.path.exists(args.seed) and model_tag_ok(args.seed, model, crop) else [],
        status_path=os.path.join(args.out, "STATUS.md"), status_history=args.status_history,
    )
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
