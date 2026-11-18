"""Unit tests for scripts/build_ja_catalog.py. No network: fetch_json / fetch_bytes are replaced by stubs."""
import hashlib
import io
import json
import os
import sys
import threading
import time

import numpy as np
import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import build_ja_catalog as B  # noqa: E402

GROUPS_URL = "https://tcgcsv.com/tcgplayer/85/groups"


def products_url(group_id):
    return f"https://tcgcsv.com/tcgplayer/85/{group_id}/products"


def card(pid, number="001/076", name="Pikachu", rarity="Common", image_count=1, group_id=1):
    ext = [{"name": "Number", "value": number}] if number is not None else []
    if rarity:
        ext.append({"name": "Rarity", "value": rarity})
    return {"productId": pid, "groupId": group_id, "name": name, "imageCount": image_count, "extendedData": ext}


GROUP = {"groupId": 1, "name": "Test Set", "abbreviation": "TS1", "publishedOn": "2019-03-15T00:00:00"}


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(B, "sleep", lambda s: None)


def stub_json(monkeypatch, table, calls=None):
    def fake(url):
        if calls is not None:
            calls.append(url)
        value = table[url]
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(B, "fetch_json", fake)


# ---- filtering, cleaning, row fields ----

def test_product_without_number_is_dropped():
    assert B.card_row(card(5, number=None), GROUP) is None
    assert B.card_row({"productId": 5, "extendedData": []}, GROUP) is None


def test_name_cleaning_strips_number_suffix_only():
    assert B.clean_name("Masquerain - 003/076", "003/076") == "Masquerain"
    assert B.clean_name("Pikachu - Special Art", "003/076") == "Pikachu - Special Art"
    assert B.clean_name("Pikachu - 004/076", "003/076") == "Pikachu - 004/076"


def test_row_fields():
    row = B.card_row(card(42, number="003/076", name="Masquerain - 003/076", rarity="Rare", image_count=2), GROUP)
    assert row == {
        "productId": 42, "groupId": 1, "group": "Test Set", "abbr": "TS1", "published": "2019-03-15",
        "name": "Masquerain", "number": "003/076", "rarity": "Rare", "imageCount": 2,
    }
    assert B.card_row(card(43, rarity=None), GROUP)["rarity"] is None


# ---- list stage ----

def list_table(extra=None):
    table = {
        GROUPS_URL: {"results": [GROUP, {**GROUP, "groupId": 2, "name": "Set Two", "abbreviation": "TS2",
                                          "publishedOn": "2020-01-01T00:00:00"}]},
        products_url(1): {"results": [card(1), card(2, number=None, name="Booster Box")]},
        products_url(2): {"results": [card(3, group_id=2, image_count=0)]},
    }
    table.update(extra or {})
    return table


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_list_stage_writes_cards_and_counts(monkeypatch, tmp_path):
    stub_json(monkeypatch, list_table())
    summary = B.list_stage(str(tmp_path))
    rows = read_jsonl(tmp_path / "products.jsonl")
    assert [r["productId"] for r in rows] == [1, 3]
    assert summary["groups_total"] == 2 and summary["groups_done"] == 2 and summary["groups_failed"] == 0
    assert summary["cards"] == 2 and summary["dropped_no_number"] == 1 and summary["image_count_zero"] == 1
    assert summary["per_year"] == {"2019": 1, "2020": 1}


def test_list_stage_resume_skips_finished_group(monkeypatch, tmp_path):
    calls = []
    stub_json(monkeypatch, list_table(), calls)
    B.list_stage(str(tmp_path))
    calls.clear()
    summary = B.list_stage(str(tmp_path))
    assert calls == [GROUPS_URL]
    assert summary["cards"] == 2 and len(read_jsonl(tmp_path / "products.jsonl")) == 2


def test_list_stage_failed_group_is_recorded_skipped_and_retried_next_run(monkeypatch, tmp_path):
    calls = []
    stub_json(monkeypatch, list_table({products_url(2): B.FetchError("boom", 500)}), calls)
    summary = B.list_stage(str(tmp_path))
    assert summary["groups_failed"] == 1 and summary["groups_done"] == 1 and summary["cards"] == 1
    assert calls.count(products_url(2)) == 2  # one retry
    stub_json(monkeypatch, list_table())
    assert B.list_stage(str(tmp_path))["cards"] == 2


def test_retry_once_then_succeeds(monkeypatch):
    attempts = []

    def flaky(url):
        attempts.append(url)
        if len(attempts) == 1:
            raise B.FetchError("429", 429)
        return {"ok": True}

    monkeypatch.setattr(B, "fetch_json", flaky)
    assert B.fetch_json_retry("https://x") == {"ok": True} and len(attempts) == 2


# ---- images stage ----

IMG = b"x" * 3000


def cards_for(n):
    return [{"productId": i, "imageCount": 1} for i in range(1, n + 1)]


def test_downloads_skip_existing_and_count(monkeypatch, tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "1.jpg").write_bytes(IMG)
    (images / "2.jpg").write_bytes(b"tiny")  # too small: refetched
    monkeypatch.setattr(B, "fetch_bytes", lambda url: IMG)
    stats = B.download_images(cards_for(3), str(tmp_path), workers=1, rate=6, limiter=lambda: None)
    assert (stats["downloaded"], stats["already"], stats["failed"], stats["stopped"]) == (2, 1, 0, False)
    assert stats["bytes"] == 6000


def test_images_stage_skips_cards_without_pictures_and_honors_limit(monkeypatch, tmp_path):
    rows = [{"productId": i, "imageCount": 0 if i == 2 else 1} for i in range(1, 6)]
    B.append_jsonl(str(tmp_path / "products.jsonl"), rows)
    fetched = []
    monkeypatch.setattr(B, "fetch_bytes", lambda url: fetched.append(url) or IMG)
    monkeypatch.setattr(B, "make_limiter", lambda rate: (lambda: None))
    stats = B.images_stage(str(tmp_path), limit=3, workers=1, rate=6)
    assert stats["downloaded"] == 3 and len(fetched) == 3
    assert not (tmp_path / "images" / "2.jpg").exists()


def test_failure_is_logged(monkeypatch, tmp_path):
    def fake(url):
        raise B.FetchError("gone", 404)

    monkeypatch.setattr(B, "fetch_bytes", fake)
    stats = B.download_images(cards_for(1), str(tmp_path), workers=1, rate=6, limiter=lambda: None)
    assert stats["failed"] == 1 and not stats["stopped"]
    rows = read_jsonl(tmp_path / "image_failures.jsonl")
    assert rows[0]["productId"] == 1 and rows[0]["status"] == 404


def test_stop_after_25_refusals_in_a_row(monkeypatch, tmp_path):
    calls = []

    def fake(url):
        calls.append(url)
        raise B.FetchError("forbidden", 403)

    monkeypatch.setattr(B, "fetch_bytes", fake)
    stats = B.download_images(cards_for(40), str(tmp_path), workers=1, rate=6, limiter=lambda: None)
    assert stats["stopped"] is True
    assert len(calls) == 25


def test_24_refusals_then_success_does_not_stop(monkeypatch, tmp_path):
    def fake(url):
        pid = int(url.split("/product/")[1].split("_")[0])
        if pid <= 24:
            raise B.FetchError("slow down", 429)
        return IMG

    monkeypatch.setattr(B, "fetch_bytes", fake)
    stats = B.download_images(cards_for(30), str(tmp_path), workers=1, rate=6, limiter=lambda: None)
    assert stats["stopped"] is False and stats["downloaded"] == 6 and stats["failed"] == 24


def test_a_success_resets_the_refusal_streak(monkeypatch, tmp_path):
    def fake(url):
        pid = int(url.split("/product/")[1].split("_")[0])
        if pid % 25 == 0:
            return IMG
        raise B.FetchError("no", 403)

    monkeypatch.setattr(B, "fetch_bytes", fake)
    stats = B.download_images(cards_for(60), str(tmp_path), workers=1, rate=6, limiter=lambda: None)
    assert stats["stopped"] is False


def test_caps():
    assert B.clamp_workers(99) == 4 and B.clamp_workers(2) == 2 and B.clamp_workers(0) == 1
    assert B.clamp_rate(99) == 6 and B.clamp_rate(3) == 3
    with pytest.raises(ValueError):
        B.clamp_rate(0)


def test_worker_cap_is_enforced(monkeypatch, tmp_path):
    state = {"now": 0, "peak": 0}
    lock = threading.Lock()

    def fake(url):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.02)
        with lock:
            state["now"] -= 1
        return IMG

    monkeypatch.setattr(B, "fetch_bytes", fake)
    stats = B.download_images(cards_for(24), str(tmp_path), workers=50, rate=100, limiter=lambda: None)
    assert stats["downloaded"] == 24 and state["peak"] <= 4


def test_rate_limiter_spaces_requests():
    slept = []
    wait = B.make_limiter(5, clock=lambda: 0.0, sleep=slept.append)
    for _ in range(4):
        wait()
    assert slept == pytest.approx([0.2, 0.4, 0.6])  # first call is free


def test_rate_limiter_is_capped_at_six():
    slept = []
    wait = B.make_limiter(1000, clock=lambda: 0.0, sleep=slept.append)
    wait()
    wait()
    assert slept == pytest.approx([1 / 6])


# ---- dedupe ----

def test_placeholder_threshold():
    same = {i: "aa" for i in range(1, 22)}
    assert set(B.find_placeholders(same)) == {"aa"}
    assert B.find_placeholders(same)["aa"]["count"] == 21
    assert len(B.find_placeholders(same)["aa"]["examples"]) == 5
    twenty = {i: "bb" for i in range(1, 21)}
    assert B.find_placeholders(twenty) == {}


def test_card_shape_filter():
    assert B.is_card_shaped(716, 1000)
    assert B.is_card_shaped(680, 1000) and B.is_card_shaped(750, 1000)
    assert not B.is_card_shaped(1000, 1000)
    assert not B.is_card_shaped(679, 1000) and not B.is_card_shaped(751, 1000)
    assert not B.is_card_shaped(0, 0)


def jpeg(width, height, color=(200, 30, 30)):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, "JPEG")
    return buf.getvalue()


def test_dedupe_stage_reports(tmp_path, capsys):
    images = tmp_path / "images"
    images.mkdir()
    for pid in range(1, 23):
        (images / f"{pid}.jpg").write_bytes(jpeg(71, 100, (9, 9, 9)))
    (images / "100.jpg").write_bytes(jpeg(100, 100, (1, 2, 3)))
    (images / "101.jpg").write_bytes(jpeg(71, 100, (4, 5, 6)))
    summary = B.dedupe_stage(str(tmp_path))
    assert summary["placeholder_products"] == 22 and summary["not_card_shaped"] == 1
    assert summary["not_card_shaped_examples"] == [100]
    saved = json.loads((tmp_path / "placeholders.json").read_text())
    assert len(saved) == 1 and list(saved.values())[0]["count"] == 22


# ---- embed ----

def fake_embedder(calls):
    def embed(data, trusted=False):
        calls.append(len(data))
        v = np.arange(1, 513, dtype=np.float32) + len(data)
        return v / np.linalg.norm(v)

    return embed


def test_embed_stage_skips_placeholders_and_bad_shapes_and_resumes(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    ph = jpeg(71, 100, (9, 9, 9))
    for pid in (1, 2, 3):
        (images / f"{pid}.jpg").write_bytes(jpeg(71, 100, (pid * 10, 0, 0)))
    (images / "4.jpg").write_bytes(ph)
    (images / "5.jpg").write_bytes(jpeg(100, 100))  # not card shaped
    (images / "6.jpg").write_bytes(b"not a jpeg" * 300)  # cannot decode
    (tmp_path / "placeholders.json").write_text(json.dumps({hashlib.sha256(ph).hexdigest(): {"count": 21}}))
    calls = []
    # a partial file from an earlier run already holds id 1
    np.savez(tmp_path / "emb_t.partial.npz", ids=np.array([1], dtype=np.int64),
             vectors=np.ones((1, 512), dtype=np.float32) / np.sqrt(512))
    summary = B.embed_stage(str(tmp_path), "t", fake_embedder(calls), save_every=2)
    assert len(calls) == 2  # ids 2 and 3 only
    out = np.load(tmp_path / "emb_t.npz")
    assert out["ids"].tolist() == [1, 2, 3] and out["ids"].dtype == np.int64
    assert out["vectors"].shape == (3, 512) and out["vectors"].dtype == np.float32
    assert np.allclose(np.linalg.norm(out["vectors"], axis=1), 1, atol=1e-5)
    assert summary["decode_failed"] == 1 and not (tmp_path / "emb_t.partial.npz").exists()


# ---- pack ----

def unit_rows(n, seed=0):
    v = np.random.default_rng(seed).normal(size=(n, 512)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def product(pid, gid=1, name="Pikachu", group="Test Set", number="001/076"):
    return {"productId": pid, "groupId": gid, "group": group, "abbr": "TS", "published": "2019-03-15",
            "name": name, "number": number, "rarity": None, "imageCount": 1}


def make_catalog(tmp_path, products, emb_ids, placeholders=None, images=None, tag="T"):
    """products.jsonl, emb_<tag>.npz (random unit rows for emb_ids), placeholders.json (None = not written)."""
    B.append_jsonl(str(tmp_path / "products.jsonl"), products)
    vectors = unit_rows(len(emb_ids))
    np.savez(tmp_path / f"emb_{tag}.npz", ids=np.array(emb_ids, dtype=np.int64), vectors=vectors)
    if placeholders is not None:
        (tmp_path / "placeholders.json").write_text(json.dumps(placeholders))
    if images:
        (tmp_path / "images").mkdir(exist_ok=True)
        for pid, data in images.items():
            (tmp_path / "images" / f"{pid}.jpg").write_bytes(data)
    return dict(zip(emb_ids, vectors))


@pytest.fixture
def models(tmp_path):
    rows_model, query_model = tmp_path / "rows.onnx", tmp_path / "query.onnx"
    rows_model.write_bytes(b"rows model bytes")
    query_model.write_bytes(b"query model bytes")
    return str(rows_model), str(query_model)


def run_pack(tmp_path, models, dest=None, built="2026-11-17"):
    return B.pack_stage(str(tmp_path), "T", models[0], models[1], 224, built, dest)


def test_pack_layout_header_and_roundtrip(tmp_path, models):
    originals = make_catalog(tmp_path, [product(10, 1, "ピカチュウ"), product(11, 1, "Eevee", number="002/076")],
                             [10, 11], placeholders={})
    summary = run_pack(tmp_path, models)
    header, scales, rows = B.read_pack(str(tmp_path / "catalog-ja.bin.gz"))
    assert header == {
        "v": 1, "dim": 512, "entries": 2,
        "rows": hashlib.sha256(b"rows model bytes").hexdigest()[:16],
        "query": hashlib.sha256(b"query model bytes").hexdigest()[:16],
        "crop": 224, "built": "2026-11-17",
        "cards": [{"pid": 10, "gid": 1, "name": "ピカチュウ", "set": "Test Set", "number": "001/076"},
                  {"pid": 11, "gid": 1, "name": "Eevee", "set": "Test Set", "number": "002/076"}],
    }
    assert scales.dtype == np.float32 and scales.shape == (2,)
    assert rows.dtype == np.int8 and rows.shape == (2, 512)
    for i, pid in enumerate([10, 11]):
        x = rows[i].astype(np.float32) * scales[i]
        assert np.linalg.norm(x) == pytest.approx(1.0, abs=1e-5)
        assert float(x @ originals[pid]) > 0.999
    assert summary["entries"] == 2


def test_pack_raw_layout_and_gzip_header(tmp_path, models):
    import gzip
    import struct
    make_catalog(tmp_path, [product(1)], [1], placeholders={})
    run_pack(tmp_path, models)
    blob = (tmp_path / "catalog-ja.bin.gz").read_bytes()
    assert blob[:2] == b"\x1f\x8b" and blob[3] == 0 and blob[4:8] == b"\x00\x00\x00\x00"  # no FNAME, mtime 0
    raw = gzip.decompress(blob)
    (hlen,) = struct.unpack("<I", raw[:4])
    header = json.loads(raw[4:4 + hlen].decode("utf-8"))
    assert raw[4:4 + hlen] == json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    assert len(raw) == 4 + hlen + 4 * 1 + 512 * 1


def test_pack_leaves_out_and_counts(tmp_path, models):
    ph = jpeg(71, 100, (9, 9, 9))
    ok = jpeg(71, 100, (50, 0, 0))
    products = [
        product(1),                                  # kept
        product(2),                                  # placeholder picture
        product(3),                                  # no embedding row
        product(4, name=""),                         # empty name
        product(5, group="g" * 201),                 # group too long
        product(6, number=None),                     # no number
        product(7, name="n" * 200),                  # exactly 200: kept
    ]
    make_catalog(tmp_path, products, [1, 2, 4, 5, 6, 7, 99],  # 99 has a row but no product
                 placeholders={hashlib.sha256(ph).hexdigest(): {"count": 21, "examples": [2]}},
                 images={1: ok, 2: ph, 7: ok})
    summary = run_pack(tmp_path, models)
    header, _, _ = B.read_pack(str(tmp_path / "catalog-ja.bin.gz"))
    assert [c["pid"] for c in header["cards"]] == [1, 7]
    assert summary["leftOut"] == {"placeholder": 1, "noRow": 1, "badText": 3}


def test_pack_order_is_group_number_product(tmp_path, models):
    products = [product(5, gid=2, number="001/009"), product(4, gid=1, number="010/076"),
                product(3, gid=1, number="002/076"), product(2, gid=1, number="002/076"),
                product(1, gid=2, number="001/009")]
    make_catalog(tmp_path, products, [1, 2, 3, 4, 5], placeholders={})
    run_pack(tmp_path, models)
    header, _, _ = B.read_pack(str(tmp_path / "catalog-ja.bin.gz"))
    assert [c["pid"] for c in header["cards"]] == [2, 3, 4, 1, 5]


def test_pack_rows_follow_the_card_order(tmp_path, models):
    originals = make_catalog(tmp_path, [product(7, gid=2), product(8, gid=1)], [7, 8], placeholders={})
    run_pack(tmp_path, models)
    header, scales, rows = B.read_pack(str(tmp_path / "catalog-ja.bin.gz"))
    assert [c["pid"] for c in header["cards"]] == [8, 7]
    for i, pid in enumerate([8, 7]):
        assert float((rows[i].astype(np.float32) * scales[i]) @ originals[pid]) > 0.999


def test_pack_same_inputs_same_bytes(tmp_path, models):
    make_catalog(tmp_path, [product(1), product(2, number="002/076")], [1, 2], placeholders={})
    run_pack(tmp_path, models, dest=str(tmp_path / "a.gz"))
    run_pack(tmp_path, models, dest=str(tmp_path / "b.gz"))
    assert (tmp_path / "a.gz").read_bytes() == (tmp_path / "b.gz").read_bytes()


def test_pack_sidecar_matches_file(tmp_path, models):
    import gzip
    make_catalog(tmp_path, [product(1)], [1], placeholders={})
    summary = run_pack(tmp_path, models)
    dest = tmp_path / "catalog-ja.bin.gz"
    side = json.loads((tmp_path / "catalog-ja.bin.gz.json").read_text(encoding="utf-8"))
    blob = dest.read_bytes()
    assert side["sha256"] == hashlib.sha256(blob).hexdigest()
    assert side["bytes"] == len(blob) and side["rawBytes"] == len(gzip.decompress(blob))
    assert (side["entries"], side["crop"], side["built"]) == (1, 224, "2026-11-17")
    assert side["rows"] == hashlib.sha256(b"rows model bytes").hexdigest()[:16]
    assert side["query"] == hashlib.sha256(b"query model bytes").hexdigest()[:16]
    assert side["leftOut"] == {"placeholder": 0, "noRow": 0, "badText": 0}
    assert summary == side


def test_pack_requires_placeholders_json(tmp_path, models):
    make_catalog(tmp_path, [product(1)], [1], placeholders=None)
    with pytest.raises(B.PackError, match="dedupe"):
        run_pack(tmp_path, models)
    assert not (tmp_path / "catalog-ja.bin.gz").exists()


def test_pack_refuses_too_many_cards(tmp_path, models, monkeypatch):
    make_catalog(tmp_path, [product(1), product(2), product(3)], [1, 2, 3], placeholders={})
    monkeypatch.setattr(B, "PACK_MAX_CARDS", 2)
    with pytest.raises(B.PackError, match="2"):
        run_pack(tmp_path, models)
    assert not (tmp_path / "catalog-ja.bin.gz").exists()


def test_pack_zero_row_is_an_error(tmp_path, models):
    make_catalog(tmp_path, [product(1)], [1], placeholders={})
    np.savez(tmp_path / "emb_T.npz", ids=np.array([1], dtype=np.int64), vectors=np.zeros((1, 512), dtype=np.float32))
    with pytest.raises(B.PackError, match="zero"):
        run_pack(tmp_path, models)


def test_pack_command_line(tmp_path, models, capsys):
    make_catalog(tmp_path, [product(1)], [1], placeholders={})
    code = B.main(["--out", str(tmp_path), "pack", "--tag", "T", "--rows-model", models[0],
                   "--query-model", models[1], "--crop", "224", "--built", "2026-11-17"])
    assert code == 0 and (tmp_path / "catalog-ja.bin.gz").exists()
    assert "sha256" in capsys.readouterr().out
