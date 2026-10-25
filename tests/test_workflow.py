"""Structure checks on the nightly workflow and the pinned requirements (no YAML parser needed)."""
import os
import re
import unittest

import _paths

with open(os.path.join(_paths.ROOT, ".github", "workflows", "nightly.yml"), encoding="utf-8") as _f:
    TEXT = _f.read()
BUILD, SIGN = TEXT.split("\n  sign:\n")


class Workflow(unittest.TestCase):
    def test_every_action_is_pinned_to_a_commit_sha(self):
        uses = re.findall(r"uses:\s*(\S+)", TEXT)
        self.assertGreaterEqual(len(uses), 6)
        for ref in uses:
            self.assertRegex(ref, r"^[\w.-]+/[\w.-]+@[0-9a-f]{40}$", ref)

    def test_the_build_job_has_no_secrets_and_the_key_is_only_in_the_sign_job(self):
        self.assertNotIn("secrets.", BUILD.split("jobs:\n  build:")[1])
        self.assertNotIn("PACK_SIGNING_KEY", BUILD.split("jobs:\n  build:")[1])
        self.assertEqual(TEXT.count("secrets.PACK_SIGNING_KEY"), 1)
        self.assertIn("secrets.PACK_SIGNING_KEY", SIGN)
        self.assertIn("environment: pack-signing", SIGN)
        self.assertIn("needs: build", SIGN)

    def test_code_comes_from_main_and_contributions_from_inbox_into_a_separate_folder(self):
        self.assertIn("ref: main", BUILD)
        self.assertIn("ref: inbox", BUILD)
        self.assertIn("path: data", BUILD)
        self.assertIn("--data data", BUILD)

    def test_the_build_job_cannot_write_the_repository(self):
        self.assertRegex(TEXT.split("jobs:")[0], r"permissions:\n  contents: read")
        self.assertNotIn("contents: write", BUILD.split("jobs:")[1])

    def test_the_sign_job_installs_only_the_signing_requirements(self):
        self.assertIn("requirements-sign.txt", SIGN)
        self.assertNotIn("requirements.txt", SIGN)
        self.assertIn("scripts/sign_release.py", SIGN)

    def test_the_build_job_reads_the_live_release_with_the_read_only_token_and_passes_it_to_the_pipeline(self):
        self.assertIn("gh release download", BUILD)
        self.assertIn("github.token", BUILD)
        self.assertIn("--live-dir live", BUILD)
        self.assertIn("--seed seed.npz", BUILD)
        self.assertIn("--base-url", BUILD)

    def test_the_sign_job_signs_per_run_urls_and_publishes_unapproved_bundles_as_prereleases(self):
        self.assertIn("--base-url", SIGN)
        self.assertNotIn("--url ", SIGN)
        self.assertIn("--prerelease", SIGN)
        self.assertIn("manifest.sig", SIGN)

    def test_daily_prices_are_their_own_prerelease_and_a_failed_fetch_does_not_stop_the_pack(self):
        self.assertIn("python -m pipeline.prices", BUILD)
        self.assertIn("continue-on-error: true", BUILD)
        self.assertIn('gh release create prices --repo "${RELEASE_REPO}" --prerelease', SIGN)
        self.assertIn("gh release upload prices prices/prices.json --clobber", SIGN)

    def test_every_part_is_uploaded_from_the_build_job(self):
        for name in ("community-pack.bin", "adapter.bin", "check.bin", "catalog-extra.bin", "manifest.json", "STATUS.md"):
            self.assertIn(f"out/{name}", BUILD)


class Requirements(unittest.TestCase):
    def check(self, name):
        with open(os.path.join(_paths.ROOT, name), encoding="utf-8") as f:
            text = f.read()
        entries = re.findall(r"^([A-Za-z0-9_.-]+)(==[^\s\\]+)?", re.sub(r"(?m)^#.*$", "", text), re.M)
        names = [n for n, _ in entries if n]
        self.assertTrue(names)
        for block in re.split(r"\n(?=[A-Za-z])", re.sub(r"(?m)^#.*\n", "", text).strip()):
            self.assertRegex(block, r"^[A-Za-z0-9_.-]+==\d[^\s]*")
            self.assertRegex(block, r"--hash=sha256:[0-9a-f]{64}")

    def test_requirements_are_exact_and_hashed(self):
        self.check("requirements.txt")

    def test_signing_requirements_are_exact_and_hashed(self):
        self.check("requirements-sign.txt")


if __name__ == "__main__":
    unittest.main()
