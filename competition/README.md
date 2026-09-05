# DocSem Shared Task HF Deployment

This folder contains source code and documentation for the DocInsights 2026 document-grounded quantitative reasoning shared task.

## Repositories

The current deployment targets:

- `amitbcp/docinsights-2026-shared-task-data`: public dataset repo for labelled train tasks, unlabelled validation tasks, and—after its audited release—held-out test tasks, PDFs, checksums, instructions, and a sample submission.
- `amitbcp/docinsights-2026-shared-task-submissions`: private dataset repo for hidden validation/test labels, immutable submissions, and derived leaderboard state.
- `amitbcp/docsem-docinsights`: public Gradio Space for split-aware submission scoring and the validation leaderboard.
- `amitbcp/docsem-docinsights-organizer`: private, read-only organizer Space for detailed held-out test attempts and projections.

## Source Data

Participant-visible tasks, labels, instructions, and PDFs are generated from the
canonical public release:

```text
/Users/aamita/Oracle/amitbcp/gsm-sem/docsem
```

Hidden validation labels are read separately from the organizer-only source at
`/Users/aamita/Oracle/amitbcp/docsem-workshop-final-public`. They are written
only to the ignored private HF payload.

Regenerate local HF payloads without changing leaderboard state:

```bash
/Users/aamita/miniconda3/bin/python scripts/prepare_docsem_hf_dataset.py
```

That no-argument command intentionally generates only train and validation and removes any stale generated `test/` output. It never discovers a test source by directory name.

No official held-out test release is present in this checkout, so the test portal remains disabled. After an explicitly selected official source has passed the separate public/private staging audit, add only its audited public half to the Hugging Face payload:

```bash
/Users/aamita/miniconda3/bin/python scripts/prepare_docsem_hf_dataset.py \
  --test-public-staging competition/hf-test-staging/public
```

The explicit staging root must contain exactly public test tasks, PDFs, checksums, and the sanitized release manifest. The generator re-audits counts and hashes, copies those public files byte-for-byte, and never reads or copies private staging. It also renders the test-ready dataset card from the tracked train/validation-only template in the same audited generation. Missing, malformed, linked, special, extra, or label-bearing inputs fail closed before generated output changes.

Do not upload a test-ready dataset card early or separately from the matching audited `test/` payload. The tracked card deliberately remains loadable with only train and validation. Publication tooling must consume the generated release-card bytes and matching test snapshot as one reconciled release.

To prepare an intentional leaderboard reset as part of a refreshed release:

```bash
/Users/aamita/miniconda3/bin/python scripts/prepare_docsem_hf_dataset.py --reset-leaderboard
```

The published dataset includes:

- 908 train tasks
- 908 train labels
- 217 validation tasks without public labels
- train and validation PDFs

Once the official held-out release is audited and published, it additionally includes:

- test tasks without labels
- test PDFs
- public checksums and sanitized release metadata

The private submissions dataset includes:

- 217 hidden validation labels
- held-out test labels and release policy only after the official private release is configured

Generated dataset payloads, private labels, submissions, and leaderboard state are intentionally ignored by this GitHub repository. Publish them only to the dedicated Hugging Face repositories.

## Participant Loading Example

```python
from datasets import load_dataset
from huggingface_hub import hf_hub_download

repo_id = "amitbcp/docinsights-2026-shared-task-data"

tasks = load_dataset(repo_id, "tasks")
train_tasks = tasks["train"]
val_tasks = tasks["validation"]
# After the official held-out release, reloading this config also provides
# tasks["test"].

train_labels = load_dataset(repo_id, "labels")["train"]

pdf_path = hf_hub_download(
    repo_id=repo_id,
    repo_type="dataset",
    filename=val_tasks[0]["document_pdf"],
)
```

## Submission Format

Participants submit JSONL for the selected split, one object per instance:

```json
{"instance_id":"task_000909","answer":"140","evidence":["b14"]}
```

Validation remains anonymous-compatible, unlimited, and visible on the existing public validation leaderboard. The held-out test workflow is different:

- Hugging Face OAuth is required for test; the authenticated account controls quota.
- Each Hugging Face account receives at most three valid test attempts.
- Attempt 1 returns answer accuracy and evidence F1 and can be revisited from that signed-in account's submission history.
- Attempts 2 and 3 are accepted with receipts but their scores are withheld.
- The account's best of three is selected for the final ranking using deterministic metric and timestamp tie-breaks.
- No public test score or rank is shown while the window is open. Organizers use the separate private, read-only Space for detailed live test results.

## Publishing

Upload the public dataset. Use the resumable large-folder uploader because the
release contains 1,125 PDFs:

```bash
/Users/aamita/miniconda3/bin/huggingface-cli upload-large-folder amitbcp/docinsights-2026-shared-task-data competition/hf-dataset --repo-type dataset --num-workers 4
```

Do not run that publication command with a generated `test/` directory until the label-free release manifest, task/PDF counts, and every checksum have passed the guarded publication dry run. Declaring the future `test` split in the dataset card does not mean the held-out data has been released or that the submission window is open.

Reset prior submissions and leaderboard state only when the public release has
changed and prior scores are no longer comparable. The command is a dry run
unless `--yes` is provided, and it preserves hidden validation labels:

```bash
/Users/aamita/miniconda3/bin/python scripts/reset_docsem_hf_leaderboard.py
/Users/aamita/miniconda3/bin/python scripts/reset_docsem_hf_leaderboard.py --yes
```

For an organizer-only validation-label correction, preserve the stored
submissions and recompute them instead of resetting the leaderboard. The
corrections file maps each private instance ID to `expected` and `replacement`
answers. Run the script once without `--yes`, then briefly deploy and verify the
Space submission-maintenance gate before committing the pinned snapshot:

```bash
/Users/aamita/miniconda3/bin/python scripts/recompute_docsem_hf_leaderboard.py \
  --corrections-file /secure/path/corrections.json
/Users/aamita/miniconda3/bin/python scripts/recompute_docsem_hf_leaderboard.py \
  --corrections-file /secure/path/corrections.json \
  --maintenance-confirmed --yes
```

The write is compare-and-swap protected against the audited private-repository
revision. If that revision moves, keep submissions paused, take a new snapshot,
and rerun rather than overwriting concurrent work. Verify the corrected gold,
every stored submission, and the rebuilt leaderboard before reopening scoring.

The deployed Gradio Space reads public tasks from the dataset repo and stores
scored submissions in the private repo. When republishing
`competition/hf-space`, configure:

```text
PUBLIC_DATASET_REPO=amitbcp/docinsights-2026-shared-task-data
GOLD_REPO_ID=amitbcp/docinsights-2026-shared-task-submissions
GOLD_FILE=private/val_labels.jsonl
SUBMISSIONS_REPO_ID=amitbcp/docinsights-2026-shared-task-submissions
HF_WRITE_TOKEN=<write token with access to private submissions repo>
```

## Held-out test deployment gate

No official test release is present in this checkout. Publish the Space with the test workflow disabled:

```text
TEST_SUBMISSIONS_ENABLED=false
TEST_PUBLIC_LEADERBOARD_ENABLED=false
```

Only after organizers have published and pinned the official public task manifest and private scoring release may they request test activation. The deployment must then supply a `TEST_RELEASE_ID` made of letters, digits, dot, underscore, or hyphen; lowercase 64-character SHA-256 values for `TEST_TASK_MANIFEST_SHA256` and `TEST_GOLD_SHA256`; RFC3339 UTC (`Z`) `TEST_OPEN_AT` and `TEST_CLOSE_AT` values with `open < close`; exactly `TEST_MAX_ATTEMPTS=3`; and server-secret `TEST_RELEASE_CONFIG_PATH` and `TEST_GOLD_CONFIG_PATH` values. Those server paths have no public defaults. Any omission, malformed release ID/digest/window/path, reversed window, or non-three attempt value keeps both requested test surfaces disabled. The private server release remains the authority and is verified again at submission time.

Validation remains anonymous and follows the established validation scoring and public-leaderboard behavior under every test-gate outcome. Do not place test gold paths, labels, answer/evidence values, OAuth subjects, email addresses, raw predictions, per-example metrics, or later-attempt scores in the public Space configuration, logs, site content, or deployment documentation.
