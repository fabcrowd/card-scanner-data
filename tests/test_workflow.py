"""Structure checks on the nightly workflow and the pinned requirements (no YAML parser needed)."""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

import _paths

with open(os.path.join(_paths.ROOT, ".github", "workflows", "nightly.yml"), encoding="utf-8") as _f:
    TEXT = _f.read()
BUILD, SIGN = TEXT.split("\n  sign:\n")


def _working_bash():
    """The first bash on PATH that runs (on Windows `bash` can resolve to a WSL stub that does not)."""
    path = shutil.which("bash")
    if not path:
        return None
    probe = subprocess.run([path, "-c", "echo ok"], capture_output=True, text=True)
    return path if probe.stdout.strip() == "ok" else None


BASH = _working_bash()


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

    def test_the_download_of_the_latest_release_names_a_pattern_because_gh_refuses_without_one(self):
        lines = [
            line for line in BUILD.splitlines()
            if "gh release download" in line and re.search(r"--dir live(\s|$)", line)
        ]
        self.assertEqual(len(lines), 1)
        self.assertNotRegex(lines[0], r"gh release download\s+[^-\s]")
        self.assertIn("--pattern", lines[0])

    def fetch_script(self):
        step = BUILD.split("- name: Fetch the live bundle and status history")[1].split("\n      - name:")[0]
        body = step.split("run: |\n")[1]
        return textwrap.dedent(body)

    def test_the_download_fallbacks_forgive_only_a_missing_release(self):
        script = self.fetch_script()
        downloads = [line for line in script.splitlines() if "gh release download" in line]
        self.assertEqual(len(downloads), 2)
        for line in downloads:
            self.assertTrue(line.startswith("if ! gh release download"), line)
            self.assertNotIn("||", line)
        self.assertEqual(script.count('grep -qi'), 2)
        self.assertEqual(script.count("exit 1"), 2)
        self.assertNotRegex(script, r"\|\|\s*echo")
        live, status = [line for line in script.splitlines() if "grep -qi" in line]
        self.assertIn('grep -qi "release not found"', live)  # the live bundle: only a missing release is forgiven
        self.assertNotIn("no assets match", live)
        self.assertIn("release not found", status)  # the status history also forgives a release with no status.json yet
        self.assertIn("no assets match the file pattern", status)

    def run_fetch(self, live_message, status_message, live_exit=1, status_exit=1):
        """Runs the step's script with a fake `gh` whose live-bundle and status calls print the given stderr and exit with the
        given codes; returns (exit code, stdout)."""
        with tempfile.TemporaryDirectory() as d:
            gh = os.path.join(d, "gh")
            with open(gh, "w", newline="\n") as f:
                f.write(
                    "#!/bin/sh\n"
                    'case "$*" in\n'
                    f"  *' status '*) echo '{status_message}' >&2; exit {status_exit};;\n"
                    f"  *) echo '{live_message}' >&2; exit {live_exit};;\n"
                    "esac\n"
                )
            os.chmod(gh, 0o755)
            work = os.path.join(d, "work")
            os.mkdir(work)
            r = subprocess.run(
                [BASH, "-e", "-c", self.fetch_script()], cwd=work, capture_output=True, text=True,
                env={**os.environ, "PATH": d + os.pathsep + os.environ["PATH"], "RELEASE_REPO": "o/r"},
            )
            return r.returncode, r.stdout

    @unittest.skipUnless(BASH, "no working bash")
    def test_the_download_step_runs_on_a_missing_release_and_fails_on_anything_else(self):
        missing, no_asset = "release not found", "no assets match the file pattern"
        code, out = self.run_fetch(missing, missing)
        self.assertEqual(code, 0)
        self.assertIn("no live bundle yet", out)
        self.assertIn("no status history yet", out)
        for message in ("HTTP 401: Bad credentials", "could not resolve host", "API rate limit exceeded", no_asset):
            code, out = self.run_fetch(message, missing)
            self.assertNotEqual(code, 0, f"live: {message}")  # the live bundle download is strict
            self.assertNotIn("no live bundle yet", out)
        self.assertEqual(self.run_fetch("", "", 0, 0)[0], 0)

    @unittest.skipUnless(BASH, "no working bash")
    def test_a_status_release_without_a_status_file_does_not_lock_the_build(self):
        code, out = self.run_fetch("", "no assets match the file pattern", live_exit=0)
        self.assertEqual(code, 0)
        self.assertIn("no status history yet", out)
        for message in ("HTTP 401: Bad credentials", "could not resolve host", "API rate limit exceeded"):
            self.assertNotEqual(self.run_fetch("", message, live_exit=0)[0], 0, f"status: {message}")

    def sign_script(self):
        step = SIGN.split("- name: Verify and sign")[1].split("\n      # Only the parts")[0]
        return textwrap.dedent(step.split("run: |\n")[1])

    def run_sign(self, reissue, signer_exit, signer_message="refusing to sign: live manifest signature does not verify"):
        """Runs the 'Verify and sign' script with a fake `python` (the signer) that exits signer_exit. Returns
        (exit code, stdout, the lines it wrote to GITHUB_OUTPUT)."""
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "python"), "w", newline="\n") as f:
                f.write(f"#!/bin/sh\necho '{signer_message}' >&2\nexit {signer_exit}\n")
            os.chmod(os.path.join(d, "python"), 0o755)
            output = os.path.join(d, "github_output")
            open(output, "w").close()
            r = subprocess.run(
                [BASH, "-e", "-c", self.sign_script()], cwd=d, capture_output=True, text=True,
                env={**os.environ, "PATH": d + os.pathsep + os.environ["PATH"], "REISSUE": reissue, "GITHUB_OUTPUT": output,
                     "GITHUB_RUN_NUMBER": "9", "RELEASE_REPO": "o/r", "TAG": "pack-x-9"},
            )
            with open(output) as f:
                return r.returncode, r.stdout, f.read().split()

    def test_a_refused_re_issue_is_not_fatal_and_publishing_the_pack_waits_for_a_signature(self):
        step = SIGN.split("- name: Verify and sign")[1].split("\n      # Only the parts")[0]
        self.assertIn("id: sign", step)
        self.assertNotIn("continue-on-error", step)
        publish = SIGN.split("- name: Publish release")[1].split("\n      - name: Publish status")[0]
        self.assertIn("steps.sign.outputs.signed == 'true'", publish.split("run:")[0])
        for later in ("- name: Publish status", "- name: Publish the price snapshot"):
            header = SIGN.split(later)[1].split("run:")[0]
            self.assertNotIn("steps.sign", header)  # status and prices go on whatever happened to the pack
            self.assertNotIn("always()", header)

    @unittest.skipUnless(BASH, "no working bash")
    def test_only_a_refusal_in_re_issue_mode_is_forgiven_with_a_warning(self):
        code, out, written = self.run_sign("true", 3)
        self.assertEqual(code, 0)
        self.assertIn("::warning", out + "")
        self.assertIn("live manifest signature does not verify", out)
        self.assertEqual(written, ["signed=false"])
        code, out, written = self.run_sign("true", 0)
        self.assertEqual((code, written), (0, ["signed=true"]))
        self.assertNotIn("::warning", out)
        # A refusal on a normal night, and any failure that is not a refusal, still fail the job and sign nothing.
        for reissue, exit_code in (("false", 3), ("", 3), ("true", 1), ("true", 2), ("false", 1)):
            code, out, written = self.run_sign(reissue, exit_code)
            self.assertEqual(code, exit_code, (reissue, exit_code))
            self.assertNotIn("signed=true", written)
            self.assertNotIn("::warning", out)
        self.assertEqual(self.run_sign("false", 0)[2], ["signed=true"])

    def warn_script(self):
        step = BUILD.split("- name: Warn about held new cards")[1].split("\n      - ")[0]
        return textwrap.dedent(step.split("run: |\n")[1])

    def run_warn(self, summary_text):
        """Runs the 'Warn about held new cards' script in a folder whose out/summary.json holds summary_text (None: no file).
        `python` is the interpreter running the tests. Returns (exit code, stdout)."""
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "python"), "w", newline="\n") as f:
                f.write(f'#!/bin/sh\nexec "{sys.executable.replace(chr(92), "/")}" "$@"\n')
            os.chmod(os.path.join(d, "python"), 0o755)
            if summary_text is not None:
                os.mkdir(os.path.join(d, "out"))
                with open(os.path.join(d, "out", "summary.json"), "w") as f:
                    f.write(summary_text)
            r = subprocess.run([BASH, "-e", "-c", self.warn_script()], cwd=d, capture_output=True, text=True,
                               env={**os.environ, "PATH": d + os.pathsep + os.environ["PATH"]})
            return r.returncode, r.stdout

    def test_the_held_cards_step_sits_after_the_pipeline_and_before_the_artifact_upload(self):
        self.assertLess(BUILD.index("python -m pipeline.run"), BUILD.index("- name: Warn about held new cards"))
        self.assertLess(BUILD.index("- name: Warn about held new cards"), BUILD.index("name: pack"))
        self.assertNotIn("continue-on-error", BUILD.split("- name: Warn about held new cards")[1].split("\n      - ")[0])

    @unittest.skipUnless(BASH, "no working bash")
    def test_the_held_cards_step_warns_once_when_cards_were_held_and_is_silent_and_harmless_otherwise(self):
        code, out = self.run_warn('{"held": [{"id": "sv9-1"}, {"id": "sv9-2"}], "newCards": 8}')
        self.assertEqual(code, 0)
        self.assertEqual(out.count("::warning::"), 1)
        self.assertIn("2 new cards were held back", out)
        self.assertIn("STATUS.md", out)
        for quiet in ('{"newCards": 8}', '{"held": []}', "not json", None):
            code, out = self.run_warn(quiet)
            self.assertEqual((code, out), (0, ""), quiet)

    def test_the_price_snapshot_is_built_before_the_pack_so_nothing_slow_sits_between_the_stamp_and_the_upload(self):
        prices = BUILD.index("python -m pipeline.prices")
        pack_build = BUILD.index("python -m pipeline.run")
        pack_upload = BUILD.index("name: pack")
        self.assertLess(prices, pack_build)
        self.assertEqual(BUILD.count("python -m pipeline.prices"), 1)
        between = BUILD[pack_build:pack_upload]
        for slow in ("pipeline.prices", "gh release", "pip install", "fetch_assets"):
            self.assertNotIn(slow, between)
        self.assertIn("continue-on-error: true", BUILD[prices - 200 : prices])  # a failed price fetch still does not stop the pack

    def test_the_price_step_has_its_own_15_minute_timeout(self):
        step = BUILD.split("- name: Build the daily price snapshot")[1].split("\n      - ")[0]
        self.assertIn("python -m pipeline.prices", step)
        self.assertRegex(step, r"(?m)^        timeout-minutes: 15$")

    def test_a_quiet_night_re_issue_is_checked_by_the_sign_job_against_a_release_it_downloads_itself(self):
        self.assertIn("reissue: ${{ steps.meta.outputs.reissue }}", BUILD)
        self.assertIn("out/summary.json", BUILD)
        download = SIGN.index("gh release download")
        sign = SIGN.index("python scripts/sign_release.py")
        self.assertLess(download, sign)
        self.assertEqual(SIGN.count("gh release download"), 1)
        self.assertRegex(SIGN, r"gh release download --repo \"\$\{RELEASE_REPO\}\" --pattern '\*' --dir live-src\n")  # the latest release, no tag
        self.assertIn("if: env.REISSUE == 'true'", SIGN)
        self.assertIn("--reissue-live-dir live-src", SIGN)
        self.assertNotIn("||", SIGN[download : SIGN.index("\n", download)])  # a failed download stops the night
        self.assertNotIn("live-src", BUILD)  # the build job's files are never the re-issue's source
        # The key is still only in the one signing step; the build job still has no secrets.
        self.assertEqual(TEXT.count("secrets.PACK_SIGNING_KEY"), 1)
        self.assertNotIn("secrets.", BUILD.split("jobs:\n  build:")[1])

    def test_the_sign_job_checks_out_the_host_list_it_imports(self):
        sparse = SIGN.split("sparse-checkout: |\n")[1].split("sparse-checkout-cone-mode")[0]
        self.assertEqual(sorted(sparse.split()), ["pipeline/__init__.py", "pipeline/image_hosts.py", "requirements-sign.txt", "scripts/sign_release.py"])

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
