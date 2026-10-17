# Shared learning: data repo template and relay

Users who opt in share the corrections they make. A Cloudflare Worker (`tools/relay/`) accepts them and
commits them to a separate GitHub data repo. A nightly GitHub Action in that repo (this folder) builds a
signed community pack of reference embeddings and scores it on held-out photos. Nothing here is deployed
by the code repo; you copy it.

## What goes where

| This repo | Copy to |
|---|---|
| `tools/relay/` | Deploy as a Cloudflare Worker with wrangler |
| `tools/shared-data/` (everything, including `.github/`) | The root of the `main` branch of the data repo |

## Two branches, on purpose

- `main` holds the code that runs with your signing key's help (the workflow, `pipeline/`, `scripts/`). Only you
  push to it.
- `inbox` holds contributions. The relay's token writes here and nowhere else (the relay refuses to start if
  `DATA_BRANCH` is `main`, `master` or the base branch). The nightly job reads `inbox` into a separate folder as
  data only; nothing in it is ever executed or imported.

A GitHub fine-grained token cannot be limited to one branch, so the guarantee comes from a branch ruleset on `main`
(step 2).

## Owner setup

1. **Data repo.** Use `fabcrowd/card-scanner-data` (public, empty). It needs one commit on `main` before the
   relay can create `inbox` (the Git Data API cannot write to an empty repository): copy the contents of
   `tools/shared-data/` to its root and push to `main`
   (`.github/workflows/nightly.yml`, `pipeline/`, `scripts/`, `tests/`, `pins.json`, `requirements*.txt`).
   The repo is public, so contributions (card pictures, answers, hashed install id) are public. The privacy policy says so.
2. **Protect `main` (exact steps).** Repo, Settings, Rules, Rulesets, New ruleset, New branch ruleset.
   - Ruleset name: `main is admin-only`. Enforcement status: Active.
   - Bypass list: add yourself (your user or the Repository admin role). Add nothing else.
   - Target branches: Add target, Include default branch.
   - Branch rules: tick "Restrict updates", "Restrict deletions", "Block force pushes" and "Restrict creations" is not needed.
   - Create. The relay's token has no bypass, so it can write `inbox` but cannot change `main` even if it leaks.
   Also add a second ruleset for `inbox`: target `refs/heads/inbox`, tick "Restrict deletions" and "Block force pushes"
   (the relay only adds commits).
3. **GitHub token for the relay.** Settings, Developer settings, Fine-grained tokens: resource owner you,
   repository access "Only select repositories" = the data repo, permission "Contents: Read and write".
   Nothing else. Set a calendar reminder for its expiry.
4. **Cloudflare.** Create an account, then `npm i -g wrangler` (or use `npx wrangler`) and `wrangler login`.
5. **Bindings.** In `tools/relay/`:
   - `npx wrangler r2 bucket create card-scanner-pending`
   - `npx wrangler kv namespace create RATE`, then paste the printed id into `wrangler.toml`.
6. **wrangler.toml values.** Set `DATA_REPO` (`fabcrowd/card-scanner-data`), keep `DATA_BRANCH = "inbox"`, and
   `ALLOWED_EXTENSION_IDS` (your extension's id from `chrome://extensions`; list the dev id and the store id
   separated by commas).
7. **Secrets and deploy.**
   - `npx wrangler secret put GITHUB_TOKEN`
   - `npx wrangler secret put RELAY_ID_SECRET` with a long random string (32+ characters; for example the output
     of `openssl rand -hex 32`). Keep a copy in your password manager: it is needed to answer deletion requests.
   - `npx wrangler deploy`. Note the `https://card-scanner-relay.<subdomain>.workers.dev` URL. The hourly Cron
     Trigger is in `wrangler.toml`. Until all settings are valid the Worker answers HTTP 500 and the cron run throws.
8. **Signing key.** On your own machine, outside any repo:
   `python scripts/keygen.py --out C:\secure\pack-signing-key.pem`. It prints the public key (base64 SPKI) on
   its last line and refuses to write inside a repo or overwrite an existing file. Keep the PEM backed up; losing
   it means shipping a new public key in an extension update.
9. **Protect the key (exact steps).** Repo, Settings, Environments, New environment, name `pack-signing`.
   - Deployment branches and tags: "Selected branches and tags", add `main`.
   - Required reviewers: add yourself (optional; it makes every nightly run wait for your approval, so leave it off
     for unattended runs).
   - Environment secrets, Add secret: `PACK_SIGNING_KEY` = the full PEM text. Do **not** add it as a repository secret:
     the `build` job, which reads strangers' data, must not be able to see it.
10. **Regression set.** Build it once on your machine from labelled, non-sealed HiBid development crops, and commit
    `regression.npz` to the root of `main`:
    `python scripts/build_regression.py --labels labels.json --root <crops folder> --model clip_visual_int8.onnx --out regression.npz`.
    It holds embeddings and catalog ids only, no images. Without it the manifest never says `improved`, so no
    user ever receives the pack.

    **How blind is this gate?** The adapter's learning rate and epoch count were chosen while looking at HiBid scores,
    so the HiBid regression gate is not fully blind to the adapter. The independent checks are the community exam
    (it gates once it has n >= 30 cards that were never trained on) and the sealed score (`scripts/sealed-score.mjs`,
    run on request, never tuned against). Treat a HiBid gain as necessary, not sufficient.
10a. **Seed (build once, re-run when you have more labelled crops).** On the machine that has the catalog art and the
    labelled crops: `python scripts/build_seed.py --data-dir data --meta data/card-catalog/meta.json --catalog-images data/card-catalog/images --model clip_visual_int8.onnx --out seed.npz`.
    It embeds labelled real crops plus synthetic photo-like variants of catalog images and stores embeddings and card
    ids only, no images. Any input path containing `hibid`, `sealed` or `realworld` is refused. Commit `seed.npz` to the
    root of `main`; the nightly reads it from the `main` checkout (`--seed seed.npz`) and trains the adapter on it
    together with promoted contributions. Without the file the adapter trains on contributions alone.
11. **Where packs are published.** The Action publishes releases in the data repo itself, which is public, so
    nothing more is needed. Only if you ever make the data repo private: set the Actions variable
    `PACK_RELEASE_REPO` to a public repo and the secret `PACK_RELEASE_TOKEN` to a fine-grained token with
    "Contents: Read and write" on that repo only.
12. **Extension config.** In the extension's `lib/shared-config.js` set:
    - `RELAY_URL` = `https://<worker>.workers.dev/v1/contributions` (the full endpoint)
    - `PACK_MANIFEST_URL` = `https://github.com/fabcrowd/card-scanner-data/releases/latest/download/manifest.json`
    - `PACK_PUBLIC_KEY` = the base64 SPKI line from step 8
13. **First run.** Actions tab, "nightly-pack", Run workflow. With no contributions it publishes an empty,
    valid, not-improved bundle (a pre-release), which proves the signing key and release path work.

## Contribution schema v1

What the extension sends:

```
{ "v": 1, "install": "<uuid>", "split": "teach"|"exam", "photoId": "<16 hex>", "day": "YYYY-MM-DD",
  "model": { "clip": "<16 hex>", "crop": <int> }, "catalog": "<16 hex>",
  "cards": [ { "box": [x,y,w,h], "crop": "<base64 JPEG, at most 40000 chars>",
               "answer": { "cardId": "<id>" } | { "notACard": true } | { "unknown": true },
               "scanner": { "cardId": "<id|null>", "confidence": 0..1, "stripe": "high"|"medium"|"low"|"none" },
               "source": "pick"|"wrong-card"|"confirm" } ] }
```

Request: `POST /v1/contributions` with `{ "v": 1, "contributions": [ up to 20 contributions ] }`, at most 1 MB. Any other
body shape is a 400. Cards: 1 to 30 per contribution, box values in 0..1. Unknown fields are rejected. `day` must be
within 2 days of the server's clock.

**Split rule (exact).** `split` must equal `parseInt(photoId.slice(-8), 16) % 5 === 0 ? "exam" : "teach"`.
That is the low 32 bits of the photoId (its last 8 hex digits) mod 5: about 20% of photos are exam. The relay
rejects any other combination.

`model.clip` and `catalog` are the first 16 hex digits of the sha256 pinned in the extension's `setup.js` for
`clip_visual_int8.onnx` and `catalog.bin.gz` (currently `77538aade655907f` and `2f3c50cec84937ff`).
`model.crop` is `EMBEDDING_VERSION` (currently 2). The crop is the rectified card image, the same canvas
`embedCard` receives; the pipeline takes its own art strip from it.

**Crop limits.** A real JPEG: SOF0/SOF1/SOF2 header with at most 3 components, width and height at most 400, long
side at least 64, and the data must end with `FF D9`. The pipeline repeats these limits before decoding anything.

**What is stored.** The relay replaces `install` with `HMAC-SHA256(RELAY_ID_SECRET, install)` truncated to 16 hex
digits, and adds `received`, its own UTC timestamp. Files land on `inbox` as
`contributions/<server day>/<photoId>-<first 8 of hashed install>-<n>.json`, in hourly commits. The client's `day` is
kept as a label only; filing, ordering and day counts all use the server's `received`. `n` continues after
existing files so nothing is overwritten.

## Relay behaviour

- CORS and access: only `chrome-extension://<id>` origins listed in `ALLOWED_EXTENSION_IDS`. **CORS is not
  authentication**: any non-browser client can send any Origin. The real protection is validation, rate limits and
  the pipeline's consensus rules.
- Limits (KV counters, approximate under concurrency): 200 cards per install per UTC day; 50 requests per IP per UTC
  hour (every POST counts, including invalid ones); 500 cards per IP per UTC day. IPv6 addresses are counted per /64.
  Over a limit: 429 with `Retry-After`. Override with `CARDS_PER_INSTALL_DAY`, `REQUESTS_PER_IP_HOUR`, `IP_CARDS_PER_DAY`.
- Global ceiling: when more than `MAX_PENDING` (5000) items wait in R2 the relay answers 503. The hourly cron
  keeps the counter honest.
- Accepted items wait in R2 as `pending/<server timestamp>-<uuid>.json`. The hourly cron commits up to
  `MAX_ITEMS_PER_FLUSH` (100) of them to the `inbox` branch, in chunks of at most `MAX_FLUSH_BYTES` (20 MB), one commit
  per chunk through the Git Data API, moving the branch with `force: false`. A chunk's items are deleted from R2 only
  after its ref update succeeded. New files are sent as tree entries with inline `content`, so GitHub creates the
  blobs in the same call (one request instead of one per file, which matters for Workers' subrequest limit).
- If GitHub rejects the content of a chunk (HTTP 400/413/422) the cron retries its items one by one; an item that
  is rejected in two separate runs moves to `quarantine/` in the R2 bucket instead of being retried forever.
  Server errors and conflicts are never blamed on an item.
- Missing `inbox` branch: created from the head of `main` on first use.

## Pipeline (nightly)

The workflow has two jobs. `build` has no secrets and a read-only token: it checks out code from `main` and the
contributions from `inbox` into `data/`, reads the live release (the latest non-pre-release) for the gate, runs the
tests and the pipeline, and uploads the bundle parts (`community-pack.bin`, `adapter.bin`, `check.bin`,
`catalog-extra.bin`; a part is absent when its trainer had too little data), `manifest.json` (unsigned),
`summary.json` and `STATUS.md` / `status.json` as an artifact. `sign` (environment `pack-signing`, the only place the key
exists) installs only the signing library (`requirements-sign.txt`), runs `scripts/sign_release.py` and publishes the
release. All actions are pinned to commit SHAs and Python packages to exact versions with hashes
(`pip install --require-hashes`). To refresh the pins: bump a version, then list that version's file hashes from PyPI.

1. Load every `contributions/**/*.json`, drop invalid ones (patterns are checked with `fullmatch`), drop answers
   whose card id is not in the catalog. Cards are merged by `(photoId, box rounded to 0.01)`; each install keeps its
   own latest vote, so the same photo is never counted twice but agreement between installs stays visible. The
   sample's crop is the one from whoever voted first by server time, so a later contribution cannot swap the picture
   under an existing vote.
2. Embed every crop as the extension does (`pipeline/embed.py`: centred square of the top 45% strip, 224 px,
   CLIP normalisation, int8 model). Images are bounded first: `Image.MAX_IMAGE_PIXELS = 1_000_000`, size read from
   the header and checked before conversion (sides 64..400), JPEG draft mode to 320 px. Bad images are skipped.
   `tests/test_drift.py` checks the constants against the extension source when this folder sits next to it.
3. TEACH consensus (`pipeline/consensus.py`). Crops are clustered at cosine 0.95. A cluster with any disagreement
   is held (a "not a card" vote disagrees with every card; "unknown" abstains). A clean cluster's card is promoted
   when at least two distinct installs voted for it **on at least two different server days**, or the card is also
   in the catalog's top-20 neighbours of that crop. Rows promoted by agreement are exempt from the install cap;
   rows promoted by the neighbour rule alone are capped so no install owns more than 10% of all promoted rows
   (`MAX_INSTALL_SHARE`). No card id may have more than 5 rows (`MAX_ROWS_PER_CARD`).
4. Only promoted TEACH crops become pack rows. EXAM contributions never enter the pack: `promote()` refuses
   anything that is not `teach`.
5. EXAM score. Exam contributions in which every card was answered (none `unknown`) are scored; `notACard`
   cards are skipped. A pack row wins as in `lib/references.js` (cosine at least 0.93 and ahead of the
   catalog's top score by 0.02) and, as in `background.js`, only when its card is already in the catalog's top
   20. Reported: top-1 accuracy before and after, and the wrong-at-high rate (a wrong answer carrying the
   "high" stripe by the `fusion.js` rule). Pack-promoted answers are tied with the top hit, so never high.
6. Regression set (`regression.npz`, fixed and owned by you, see step 10). The same scoring runs on it. `improved`
   additionally requires, on that set, a rise in top-1 and in accepted-right answers, no rise in accepted-wrong answers
   and an abstain rate up by at most 2 points (`pipeline/gate.py`; accepted/abstained follow the accept rule of
   `lib/fusion.js`, read on the raw cosine because the adapter only ranks); without the file it is never true.
7. `manifest.json` is signed by the `sign` job (ECDSA P-256, SHA-256) over its exact bytes. `manifest.sig` is the
   base64 of the 64-byte raw `r||s` signature, which is what WebCrypto `verify` takes with
   `{name: "ECDSA", hash: "SHA-256"}`. The public key is imported as `spki`. Before signing, `sign_release.py`
   re-checks: manifest fields, every part's sha256 / size / format (rank at most 64, finite check weights, at most
   20000 rows), each url is this run's release asset, `seq` equals the run number, `issuedAt` is within an hour, and
   `improved: true` is not contradicted by the numbers (regression present with no drop; no exam drop once it has 30
   cards; wrong "high" answers not up). The gate in `pipeline/gate.py` decides the rest. A bundle the gate did not
   approve is published as a pre-release, so "latest" is always the live bundle. `STATUS.md` and `status.json` go to
   the rolling `status` pre-release (the data repo's `main` is admin-only, so the nightly never commits there).

### manifest.json (v2)

```
{ "v": 2, "model": "<clip 16 hex>", "crop": 2,
  "assets": { "memory"?: A, "adapter"?: A, "check"?: A, "catalogExtra"?: A },   A = { "url", "sha256", "bytes" }
  "exam": { "before": 0.0-1.0, "after": 0.0-1.0, "n": N, "wrongAtHigh": { "before": .., "after": .. } },
  "regression": { "before": .., "after": .., "n": N } | null,
  "improved": <the gate's verdict>, "seq": <workflow run number>, "issuedAt": "YYYY-MM-DDTHH:MM:SSZ" }
```

Keyed to the CLIP model and crop only, never the catalog, so catalog changes do not invalidate learning. The
extension refuses a bundle whose `model` or `crop` differ from its own, whose signature fails, whose `seq` is lower
than the highest it has accepted, or whose `issuedAt` is older than 30 days; it downloads only parts whose `sha256`
it does not already hold, and any failing part leaves the previous live bundle in place. `improved` is the gate for
using the bundle at all. A bundle replaces all parts: a part missing from the new manifest is removed. To withdraw
a bad bundle, publish a new improved one without the part (an `improved: false` run keeps what clients have).

| Part | File | Format |
|---|---|---|
| memory | `community-pack.bin` | pack v1 below |
| adapter | `adapter.bin` | header `{v:1,dim:512,rank:r<=64,scale:s}`, then U and V (`512*r` int8 each, row-major); `e' = normalize(e + U(Vt e))` on queries and catalog rows |
| check | `check.bin` | header `{v:1,dim:512,bias,threshold}`, then 512 float32 weights; a detected box is dropped when `sigmoid(w.e+b) < threshold` on the raw embedding |
| catalogExtra | `catalog-extra.bin` | pack v1 plus `"cards":[{id,name,set,setName,number,image}]` parallel to `cardIds`; made by `pipeline/new_sets.py` from TCGdex, cumulative, at most 500 new cards per night |

### Pack byte format (v1)

```
bytes 0..3        little-endian uint32 H: length of the header
bytes 4..4+H      UTF-8 JSON {"v":1,"dim":512,"entries":N,"cardIds":[N strings]}
then              N * 512 int8, row-major; row i belongs to cardIds[i]
```

Each value is `round(127 * x)` of the L2-normalised component, clamped to -127..127; there is no scale field.
The file is exactly `4 + H + entries * 512` bytes. Same layout as `lib/bundle.js`;
`tests/shared-data-pack.test.js` parses a pipeline-written fixture with the extension's own `parsePack`.
`pipeline/pack.py` is the reference reader and writer.

## Privacy note (for the opt-in screen and privacy policy)

Shared data is the card crops only: JPEG images of the cards, the box inside the photo, what the user said the
card is, what the scanner guessed, and a pseudonymous install id. No page URLs, no listing text, no account or
device identifiers. The install id is a random UUID made on the user's device; the relay never commits it: it stores
the 16-hex HMAC described above, so the public repo does not contain the id a user could be asked for. The relay does
not keep IP addresses with contributions; an IP appears only inside a rate-limit counter key that expires after about
70 minutes (hourly) or a day (cards per day). Contributions sit in the public data repo's git history, so
deleting one means rewriting history.

**Deletion request.** The user finds their install id on the options page ("Your sharing id") and sends it to you
by GitHub issue or the contact email. Then:
1. Compute the hashed id the same way the relay does:
   `python -c "import hmac,hashlib,sys; print(hmac.new(sys.argv[1].encode(), sys.argv[2].encode(), hashlib.sha256).hexdigest()[:16])" "<RELAY_ID_SECRET>" "<install id>"`
2. Find their files on `inbox`: `grep -rl '"install":"<hashed id>"' contributions/` (file names carry its first 8 characters).
3. Remove them and purge them from history with `git filter-repo --invert-paths --paths-from-file list.txt`,
   then force-push `inbox` (temporarily lift the "block force pushes" rule on it). Re-run the nightly workflow so the next
   pack no longer contains rows made from those crops.

**Takedown / abuse.** If someone reports a contribution (for example a picture of a person or an offensive image), or
you find abuse: delete the files and rewrite `inbox` history exactly as above, rotate `RELAY_ID_SECRET` only if the
mapping itself leaked, and publish an abuse contact (the same contact email as the privacy policy) on the repo
README. A bad bundle is handled by publishing a newer improved bundle without the bad part; deleting a release does
not reach clients that already stored it.

## Tests

```
python -m unittest discover -s tests            # in the data repo
python -m unittest discover -s tools/shared-data/tests   # in the code repo
node --test tests/relay-*.test.js               # relay, in the code repo
```
