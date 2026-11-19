"""The sign job's gate: it re-checks what the unprivileged build job produced before it signs anything."""
import base64
import datetime
import importlib.util
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from cryptography.hazmat.primitives.asymmetric import ec

import helpers as h
from pipeline import image_hosts, new_sets
from pipeline.manifest import ASSET_FILES, build_manifest, manifest_bytes
from pipeline.new_sets import write_extra
from pipeline.pack import write_pack
from pipeline.signing import generate_keypair, sign_bytes, verify_signature

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "sign_release.py")
spec = importlib.util.spec_from_file_location("sign_release", SCRIPT)
sign_release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sign_release)

BASE = "https://github.com/fabcrowd/card-scanner-data/releases/download/pack-20261015-9/"
NOW = datetime.datetime(2026, 10, 15, 3, 30, tzinfo=datetime.timezone.utc)
PINS = {"cropVersion": 2, "dim": 512}
EXAM = {"before": 0.5, "after": 0.6, "n": 40, "wrongAtHigh": {"before": 0.1, "after": 0.05}}
REG = {"before": 0.8, "after": 0.8, "n": 20}
CARD = {"id": "sv9-1", "name": "Pikachu", "set": "sv9", "setName": "Journey Together", "number": "1", "image": "https://assets.tcgdex.net/en/sv/sv9/1/high.webp"}


def with_header(header, body):
    head = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack("<I", len(head)) + head + body


def adapter(rank=4, scale=0.01):
    return with_header({"v": 1, "dim": 512, "rank": rank, "scale": scale}, bytes(2 * 512 * rank))


def check(bias=-1.0, threshold=0.4, weights=None):
    w = np.zeros(512, dtype="<f4") if weights is None else weights
    return with_header({"v": 1, "dim": 512, "bias": bias, "threshold": threshold}, w.tobytes())


def parts(**over):
    out = {
        "memory": write_pack(["c-1", "c-2"], np.stack([h.unit(1), h.unit(2)])),
        "adapter": adapter(),
        "check": check(),
        "catalogExtra": write_extra([CARD], np.zeros((1, 512), dtype=np.int8)),
    }
    out.update(over)
    return {k: v for k, v in out.items() if v is not None}


def release(files=None, **over):
    files = parts() if files is None else files
    args = dict(pins=PINS, clip="77538aade655907f", assets=files, base_url=BASE, exam=EXAM, regression=REG, improved=True, seq=9, issued_at="2026-10-15T03:17:00Z")
    args.update(over)
    return manifest_bytes(build_manifest(**args)), files


def tamper(raw, edit):
    m = json.loads(raw)
    edit(m)
    return json.dumps(m, sort_keys=True, separators=(",", ":")).encode()


def verify(manifest, files, **over):
    kw = dict(base_url=BASE, run_number=9, now=NOW)
    kw.update(over)
    return sign_release.verify_release(manifest, files, **kw)


class HeldBackNight(unittest.TestCase):
    def test_the_output_of_a_night_that_held_a_card_back_passes_the_unchanged_signer_checks(self):
        import contextlib
        import io
        import types

        import test_run
        from pipeline import gate
        from pipeline.run import run_pipeline

        crop = h.unit(3, noise=0.05)
        cards = [{"id": i, "name": n, "set": "sv9", "setName": "S", "number": str(k), "image": "https://assets.tcgdex.net/x"}
                 for k, (i, n) in enumerate((("sv9-1", "Thief"), ("sv9-2", "Calm")))]
        extra = write_extra(cards, np.rint(np.stack([crop, h.unit(100)]) * 127).astype(np.int8))
        new_cards = {"bytes": extra, "added": ["sv9-1", "sv9-2"], "sets": ["sv9"], "total": 2}
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as out, contextlib.redirect_stdout(io.StringIO()):
            run_pipeline(data_dir=root, out_dir=out, pins=test_run.PINS, embedder=test_run.FakeEmbedder(), catalog=test_run.catalog(),
                         modules={"gate": gate}, new_sets_fn=lambda: new_cards, live={}, regression=(np.stack([crop, crop]), ["c-3", "c-3"]),
                         **{**test_run.COMMON, "base_url": BASE, "seq": 9, "clock": lambda: NOW})
            raw = open(os.path.join(out, "manifest.json"), "rb").read()
            files = {name: open(os.path.join(out, filename), "rb").read() for name, filename in ASSET_FILES.items() if os.path.exists(os.path.join(out, filename))}
        manifest = verify(raw, files)
        self.assertIs(manifest["improved"], True)
        self.assertEqual([c["id"] for c in new_sets.read_extra(files["catalogExtra"])[0]], ["sv9-2"])
        self.assertEqual(manifest["assets"]["catalogExtra"]["bytes"], len(files["catalogExtra"]))


class Verify(unittest.TestCase):
    def refused(self, manifest, files, pattern, **over):
        with self.assertRaisesRegex(sign_release.Refused, pattern):
            verify(manifest, files, **over)

    def test_a_good_release_passes_with_every_part_and_with_any_one(self):
        manifest, files = release()
        self.assertEqual(sorted(verify(manifest, files)["assets"]), ["adapter", "catalogExtra", "check", "memory"])
        for name in ASSET_FILES:
            only = {name: parts()[name]}
            manifest, files = release(files=only)
            self.assertEqual(list(verify(manifest, files)["assets"]), [name])

    def test_a_file_that_does_not_match_its_entry_is_refused(self):
        manifest, files = release()
        self.refused(manifest, {**files, "adapter": files["adapter"] + b"\0"}, "adapter")
        self.refused(manifest, {k: v for k, v in files.items() if k != "check"}, "check file is missing")
        swapped = tamper(manifest, lambda m: m["assets"]["check"].update(bytes=3))
        self.refused(swapped, files, "check")

    def test_urls_seq_and_issue_time_must_belong_to_this_run(self):
        manifest, files = release()
        self.refused(manifest, files, "url", base_url=BASE.replace("pack-20261015-9", "pack-20261015-10"))
        self.refused(tamper(manifest, lambda m: m["assets"]["memory"].update(url="https://evil.test/community-pack.bin")), files, "url")
        self.refused(manifest, files, "seq", run_number=10)
        self.refused(manifest, files, "issuedAt", now=NOW + datetime.timedelta(hours=3))
        old, files_old = release(issued_at="2026-01-01T00:00:00Z")
        self.refused(old, files_old, "issuedAt")

    def test_improved_cannot_be_claimed_against_the_numbers(self):
        for name, over in {
            "no regression set": dict(regression=None),
            "regression dropped": dict(regression={"before": 0.8, "after": 0.7, "n": 20}),
            "empty regression": dict(regression={"before": 0.0, "after": 0.0, "n": 0}),
            "exam dropped with 30 cards": dict(exam={**EXAM, "after": 0.4}),
            "more wrong highs": dict(exam={**EXAM, "wrongAtHigh": {"before": 0.1, "after": 0.2}}),
        }.items():
            manifest, files = release(**over)
            with self.assertRaisesRegex(sign_release.Refused, "improved", msg=name):
                verify(manifest, files)

    def test_improved_with_a_small_exam_is_fine_and_honest_not_improved_is_always_signable(self):
        manifest, files = release(exam={**EXAM, "n": 5, "after": 0.1})
        verify(manifest, files)  # the exam is not trusted below 30 cards
        for over in (dict(regression=None), dict(regression={"before": 0.8, "after": 0.7, "n": 20}), dict(exam={**EXAM, "after": 0.4})):
            manifest, files = release(improved=False, **over)
            self.assertIs(verify(manifest, files)["improved"], False)

    def test_unknown_fields_parts_and_old_manifests_are_refused(self):
        manifest, files = release()
        self.refused(tamper(manifest, lambda m: m.update(evil=1)), files, "fields")
        self.refused(tamper(manifest, lambda m: m.update(catalog="0" * 16)), files, "fields")
        self.refused(tamper(manifest, lambda m: m.update(v=1)), files, "v or crop")
        self.refused(tamper(manifest, lambda m: m["assets"].update(evil=m["assets"]["memory"])), files, "unknown")
        self.refused(tamper(manifest, lambda m: m["assets"]["memory"].update(extra=1)), files, "wrong fields")
        self.refused(tamper(manifest, lambda m: m.update(assets={})), files, "assets")
        self.refused(b"not json", files, "manifest")

    def test_memory_and_extra_rows_are_validated(self):
        bad_ids = write_pack(["", "c-2"], np.stack([h.unit(1), h.unit(2)]))
        self.refused(*release(files=parts(memory=bad_ids)), "card id")
        big = [f"c-{i}" for i in range(sign_release.MAX_ENTRIES + 1)]
        self.refused(*release(files=parts(memory=write_pack(big, np.tile(h.unit(1), (len(big), 1))))), "too many")
        self.refused(*release(files=parts(memory=parts()["memory"] + b"x")), "size")

    def test_extra_cards_must_match_their_ids_and_be_well_formed(self):
        zero = np.zeros((1, 512), dtype=np.int8)
        for bad in (
            {**CARD, "name": ""},
            {**CARD, "image": "http://assets.tcgdex.net/x.webp"},
            {**CARD, "image": "https://x/" + "a" * 400},
            {**CARD, "extra": 1},
            {k: v for k, v in CARD.items() if k != "number"},
        ):
            self.refused(*release(files=parts(catalogExtra=write_extra([bad], zero))), "catalogExtra")
        no_cards = with_header({"v": 1, "dim": 512, "entries": 1, "cardIds": ["sv9-1"]}, bytes(512))
        self.refused(*release(files=parts(catalogExtra=no_cards)), "cards")

    def test_new_card_pictures_must_come_from_the_hosts_the_build_allows(self):
        zero = np.zeros((1, 512), dtype=np.int8)
        for host in new_sets.TCGDEX_IMAGE_HOST, new_sets.POKEMONTCG_IMAGE_HOST:
            ok = {**CARD, "image": host + "en/sv/sv9/1/high.webp"}
            verify(*release(files=parts(catalogExtra=write_extra([ok], zero))))
        for bad in ("https://evil.test/x.webp", "https://assets.tcgdex.net.evil.test/x.webp", "https://images.pokemontcg.io.evil.test/x"):
            self.refused(*release(files=parts(catalogExtra=write_extra([{**CARD, "image": bad}], zero))), "catalogExtra")

    def test_the_signer_uses_the_build_s_host_list_not_a_copy(self):
        self.assertEqual(sign_release.IMAGE_HOSTS, image_hosts.IMAGE_HOSTS)
        self.assertEqual(new_sets.TCGDEX_IMAGE_HOST, image_hosts.TCGDEX_IMAGE_HOST)
        self.assertEqual(new_sets.POKEMONTCG_IMAGE_HOST, image_hosts.POKEMONTCG_IMAGE_HOST)
        self.assertEqual(set(image_hosts.IMAGE_HOSTS), {"https://assets.tcgdex.net/", "https://images.pokemontcg.io/"})

    def test_adapter_and_check_are_validated(self):
        self.refused(*release(files=parts(adapter=adapter(rank=65))), "rank")
        self.refused(*release(files=parts(adapter=adapter(rank=0))), "rank")
        self.refused(*release(files=parts(adapter=adapter(scale=-1))), "scale")
        self.refused(*release(files=parts(adapter=adapter()[:-1])), "size")
        self.refused(*release(files=parts(check=check(threshold=1.5))), "threshold")
        self.refused(*release(files=parts(check=check(bias=float("nan")))), "bias")
        self.refused(*release(files=parts(check=check(weights=np.full(512, np.inf, dtype="<f4")))), "finite")
        self.refused(*release(files=parts(check=check()[:-4])), "size")


OLD_BASE = "https://github.com/fabcrowd/card-scanner-data/releases/download/pack-20260920-5/"


def live_release(**over):
    """(signed bytes of a live manifest made on 2026-09-20 as seq 5, its part files)."""
    args = dict(base_url=OLD_BASE, seq=5, issued_at="2026-09-20T03:17:00Z")
    args.update(over)
    return release(**args)


def reissue_of(live_bytes, **fields):
    """The candidate a quiet night proposes: the live manifest with this run's seq and issuedAt."""
    def edit(m):
        m.update(seq=9, issuedAt="2026-10-15T03:17:00Z")
        m.update(fields)
    return tamper(live_bytes, edit)


class Reissue(unittest.TestCase):
    def setUp(self):
        self.live_bytes, self.files = live_release()
        self.live = json.loads(self.live_bytes)

    def refused(self, candidate, pattern, files=None, live=None, **over):
        with self.assertRaisesRegex(sign_release.Refused, pattern):
            verify(candidate, self.files if files is None else files, live=self.live if live is None else live, **over)

    def test_the_live_manifest_with_a_new_seq_and_time_and_the_same_files_passes(self):
        manifest = verify(reissue_of(self.live_bytes), self.files, live=self.live)
        self.assertEqual((manifest["seq"], manifest["issuedAt"]), (9, "2026-10-15T03:17:00Z"))
        self.assertEqual(manifest["assets"], self.live["assets"])  # the urls stay the live release's, not this run's

    def test_any_field_other_than_seq_and_time_that_differs_from_the_live_manifest_is_refused(self):
        for name, edit in {
            "improved": lambda m: m.update(improved=False),
            "exam": lambda m: m["exam"].update(after=0.9),
            "regression": lambda m: m.update(regression={"before": 0.8, "after": 0.9, "n": 20}),
            "model": lambda m: m.update(model="0123456789abcdef"),
            "crop": lambda m: m.update(crop=7),
            "url": lambda m: m["assets"]["memory"].update(url=BASE + "community-pack.bin"),
            "a part dropped": lambda m: m["assets"].pop("check"),
        }.items():
            with self.subTest(name):
                self.refused(tamper(reissue_of(self.live_bytes), edit), "re-issue|files|improved")

    def test_an_older_signed_manifest_is_not_a_re_issue_of_the_live_one(self):
        older_bytes, older_files = live_release(seq=3, issued_at="2026-08-01T00:00:00Z", exam={**EXAM, "after": 0.55})
        self.refused(reissue_of(older_bytes), "re-issue", files=older_files)  # the live one is self.live, not the older one

    def test_the_new_seq_must_be_above_the_live_seq_and_the_time_must_not_go_back(self):
        self.refused(reissue_of(self.live_bytes), "seq", live={**self.live, "seq": 9})
        self.refused(reissue_of(self.live_bytes, seq=4), "seq", run_number=4)
        self.refused(reissue_of(self.live_bytes, issuedAt="2026-09-01T00:00:00Z"), "issuedAt", now=datetime.datetime(2026, 9, 1, tzinfo=datetime.timezone.utc))

    def test_the_checks_a_fresh_release_gets_still_apply(self):
        self.refused(reissue_of(self.live_bytes), "adapter", files={**self.files, "adapter": self.files["adapter"] + b"\0"})
        self.refused(reissue_of(self.live_bytes), "check file is missing", files={k: v for k, v in self.files.items() if k != "check"})
        self.refused(reissue_of(self.live_bytes), "seq", run_number=10)
        self.refused(reissue_of(self.live_bytes), "issuedAt", now=NOW + datetime.timedelta(hours=3))
        not_improved_bytes, _ = live_release(improved=False)
        self.refused(reissue_of(not_improved_bytes), "improved", live=json.loads(not_improved_bytes))

    def test_a_new_card_picture_off_the_allowed_hosts_in_a_live_part_is_still_refused(self):
        zero = np.zeros((1, 512), dtype=np.int8)
        bad_files = parts(catalogExtra=write_extra([{**CARD, "image": "https://evil.test/x.webp"}], zero))
        live_bytes, files = live_release(files=bad_files)
        self.refused(reissue_of(live_bytes), "catalogExtra", files=files, live=json.loads(live_bytes))


class ReissueSigning(unittest.TestCase):
    def setUp(self):
        self.pem, _ = generate_keypair()
        self.live_bytes, self.files = live_release()
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.out = os.path.join(self.dir.name, "out")
        self.live_dir = os.path.join(self.dir.name, "live")
        os.makedirs(self.out)
        os.makedirs(self.live_dir)

    def put(self, folder, manifest, files=None, signature=None):
        with open(os.path.join(folder, "manifest.json"), "wb") as f:
            f.write(manifest)
        if signature is not None:
            with open(os.path.join(folder, "manifest.sig"), "w") as f:
                f.write(signature)
        for name, data in (files or {}).items():
            with open(os.path.join(folder, ASSET_FILES[name]), "wb") as f:
                f.write(data)

    def run_signer(self, pem=None, live_dir=True):
        args = [sys.executable, SCRIPT, "--dir", self.out, "--base-url", BASE, "--run-number", "9", "--now", "2026-10-15T03:30:00Z"]
        if live_dir:
            args += ["--reissue-live-dir", self.live_dir]
        return subprocess.run(args, env={**os.environ, "PACK_SIGNING_KEY": (pem or self.pem).decode()}, capture_output=True, text=True)

    def publish_live(self, manifest=None, signature=None):
        manifest = manifest or self.live_bytes
        self.put(self.live_dir, manifest, self.files, signature or sign_bytes(manifest, self.pem))

    def test_a_re_issue_of_the_live_manifest_is_signed_and_the_signature_verifies(self):
        self.publish_live()
        candidate = reissue_of(self.live_bytes)
        self.put(self.out, candidate, self.files)
        r = self.run_signer()
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(os.path.join(self.out, "manifest.sig")) as f:
            self.assertTrue(verify_signature(candidate, f.read(), spki_of(self.pem)))

    def test_without_the_live_dir_a_re_issue_is_refused_as_a_normal_release(self):
        self.publish_live()
        self.put(self.out, reissue_of(self.live_bytes), self.files)
        r = self.run_signer(live_dir=False)
        self.assertNotEqual(r.returncode, 0)  # its urls are not this run's
        self.assertFalse(os.path.exists(os.path.join(self.out, "manifest.sig")))

    def refused_run(self, expect):
        r = self.run_signer()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn(expect, r.stderr)
        self.assertFalse(os.path.exists(os.path.join(self.out, "manifest.sig")))

    def test_a_live_manifest_whose_signature_does_not_verify_is_refused(self):
        other_pem, _ = generate_keypair()
        self.publish_live(signature=sign_bytes(self.live_bytes, other_pem))
        self.put(self.out, reissue_of(self.live_bytes), self.files)
        self.refused_run("live manifest signature")

    def test_a_live_manifest_with_no_signature_file_or_a_tampered_body_is_refused(self):
        self.put(self.live_dir, self.live_bytes, self.files)  # no manifest.sig
        self.put(self.out, reissue_of(self.live_bytes), self.files)
        self.refused_run("live manifest")
        self.publish_live()
        self.put(self.live_dir, tamper(self.live_bytes, lambda m: m.update(seq=1)), self.files, sign_bytes(self.live_bytes, self.pem))
        self.refused_run("live manifest signature")

    def test_a_signed_manifest_that_is_not_the_live_one_cannot_be_re_issued(self):
        """The live release moved on (seq 7); the build handed over a re-issue of the older signed seq-3 manifest."""
        newer_bytes, _ = live_release(seq=7, issued_at="2026-09-25T00:00:00Z", exam={**EXAM, "after": 0.7})
        older_bytes, older_files = live_release(seq=3, issued_at="2026-08-01T00:00:00Z")
        self.publish_live(manifest=newer_bytes)
        self.put(self.out, reissue_of(older_bytes), older_files)
        self.refused_run("re-issue")

    def test_a_build_that_changes_a_part_file_is_refused(self):
        self.publish_live()
        self.put(self.out, reissue_of(self.live_bytes), {**self.files, "check": check(bias=-2.0)})
        self.refused_run("check")

    def wrong_curve_pem(self):
        from cryptography.hazmat.primitives import serialization

        key = ec.generate_private_key(ec.SECP384R1())
        return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())

    def test_a_problem_with_the_key_exits_1_in_every_mode_not_the_refusal_code(self):
        manifest, files = release()
        for name, pem in {"unparsable": b"not a key", "wrong curve": self.wrong_curve_pem()}.items():
            with self.subTest(name, mode="normal"):
                self.put(self.out, manifest, files)
                r = self.run_signer(pem=pem, live_dir=False)
                self.assertEqual(r.returncode, 1, r.stderr)
                self.assertIn("signing key", r.stderr)
                self.assertFalse(os.path.exists(os.path.join(self.out, "manifest.sig")))
            with self.subTest(name, mode="re-issue"):
                self.publish_live()
                self.put(self.out, reissue_of(self.live_bytes), self.files)
                r = self.run_signer(pem=pem)
                self.assertEqual(r.returncode, 1, r.stderr)
                self.assertIn("signing key", r.stderr)
                self.assertFalse(os.path.exists(os.path.join(self.out, "manifest.sig")))

    def refused_reissue_with_live_issued(self, issued_at):
        """A content refusal (the live signature is by another key) of a re-issue whose live manifest was issued at issued_at."""
        other_pem, _ = generate_keypair()
        live_bytes, files = live_release(issued_at=issued_at)
        self.put(self.live_dir, live_bytes, files, sign_bytes(live_bytes, other_pem))
        self.put(self.out, reissue_of(live_bytes), files)
        return self.run_signer()

    def test_a_refused_re_issue_of_a_young_live_bundle_exits_3_and_of_an_old_or_unreadable_one_exits_1(self):
        self.assertEqual(sign_release.REISSUE_FAIL_AFTER_DAYS, 21)
        young = self.refused_reissue_with_live_issued("2026-09-24T03:30:01Z")  # one second under 21 days before --now
        self.assertEqual(young.returncode, 3, young.stderr)
        self.assertIn("live manifest signature", young.stderr)
        old = self.refused_reissue_with_live_issued("2026-09-24T03:30:00Z")  # exactly 21 days
        self.assertEqual(old.returncode, 1, old.stderr)
        self.assertIn("30 days", old.stderr)
        os.remove(os.path.join(self.live_dir, "manifest.json"))  # age cannot be determined
        unknown = self.run_signer()
        self.assertEqual(unknown.returncode, 1, unknown.stderr)
        self.assertIn("30 days", unknown.stderr)


def spki_of(pem):
    from cryptography.hazmat.primitives import serialization

    key = serialization.load_pem_private_key(pem, password=None)
    der = key.public_key().public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    return base64.b64encode(der).decode()


class Signing(unittest.TestCase):
    def write(self, out, manifest, files):
        with open(os.path.join(out, "manifest.json"), "wb") as f:
            f.write(manifest)
        for name, data in files.items():
            with open(os.path.join(out, ASSET_FILES[name]), "wb") as f:
                f.write(data)

    def args(self, out):
        return [sys.executable, SCRIPT, "--dir", out, "--base-url", BASE, "--run-number", "9", "--now", "2026-10-15T03:30:00Z"]

    def test_main_signs_the_exact_manifest_bytes(self):
        pem, spki = generate_keypair()
        manifest, files = release()
        with tempfile.TemporaryDirectory() as out:
            self.write(out, manifest, files)
            r = subprocess.run(self.args(out), env={**os.environ, "PACK_SIGNING_KEY": pem.decode()}, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(os.path.join(out, "manifest.sig")) as f:
                sig = f.read()
            self.assertEqual(len(base64.b64decode(sig)), 64)
            self.assertTrue(verify_signature(manifest, sig, spki))
            self.assertFalse(verify_signature(manifest + b" ", sig, spki))

    def test_refused_release_writes_no_signature_and_a_missing_key_stops_early(self):
        pem, _ = generate_keypair()
        manifest, files = release()
        with tempfile.TemporaryDirectory() as out:
            self.write(out, manifest, {**files, "adapter": files["adapter"] + b"x"})
            r = subprocess.run(self.args(out), env={**os.environ, "PACK_SIGNING_KEY": pem.decode()}, capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertFalse(os.path.exists(os.path.join(out, "manifest.sig")))
            env = {k: v for k, v in os.environ.items() if k != "PACK_SIGNING_KEY"}
            self.assertNotEqual(subprocess.run(self.args(out), env=env, capture_output=True, text=True).returncode, 0)

    def test_a_refusal_has_its_own_exit_code_so_the_workflow_can_tell_it_from_a_broken_run(self):
        self.assertEqual(sign_release.REFUSED_EXIT, 3)
        pem, _ = generate_keypair()
        manifest, files = release()
        with tempfile.TemporaryDirectory() as out:
            self.write(out, manifest, {**files, "adapter": files["adapter"] + b"x"})
            r = subprocess.run(self.args(out), env={**os.environ, "PACK_SIGNING_KEY": pem.decode()}, capture_output=True, text=True)
            self.assertEqual(r.returncode, 3)
            self.assertIn("refusing to sign: ", r.stderr)
            env = {k: v for k, v in os.environ.items() if k != "PACK_SIGNING_KEY"}
            self.assertEqual(subprocess.run(self.args(out), env=env, capture_output=True, text=True).returncode, 1)  # no key: not a refusal

    def test_a_part_the_manifest_lists_but_the_folder_lacks_is_refused(self):
        pem, _ = generate_keypair()
        manifest, files = release()
        with tempfile.TemporaryDirectory() as out:
            self.write(out, manifest, {k: v for k, v in files.items() if k != "check"})
            r = subprocess.run(self.args(out), env={**os.environ, "PACK_SIGNING_KEY": pem.decode()}, capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("check", r.stderr)

    def test_the_script_imports_only_the_standard_library_cryptography_and_the_host_list(self):
        with open(SCRIPT, encoding="utf-8") as f:
            text = f.read()
        imports = {line.split()[1].split(".")[0] for line in text.splitlines() if line.startswith(("import ", "from "))}
        self.assertTrue(imports <= {"argparse", "base64", "datetime", "hashlib", "json", "math", "os", "re", "struct", "sys", "cryptography", "pipeline"}, imports)
        # The one pipeline module it may import holds constants only and imports nothing itself.
        self.assertEqual([line.split("  #")[0] for line in text.splitlines() if line.startswith("from pipeline")], ["from pipeline.image_hosts import IMAGE_HOSTS"])
        with open(os.path.join(os.path.dirname(SCRIPT), "..", "pipeline", "image_hosts.py"), encoding="utf-8") as f:
            self.assertEqual([line for line in f.read().splitlines() if line.startswith(("import ", "from "))], [])


if __name__ == "__main__":
    unittest.main()
