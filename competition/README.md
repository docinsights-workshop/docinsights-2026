# DocSem Shared Task HF Deployment

This folder contains source code and documentation for the DocInsights 2026 document-grounded quantitative reasoning shared task.

## Repositories

The current deployment targets:

- `amitbcp/docinsights-2026-shared-task-data`: public dataset repo for labelled train tasks, unlabelled validation tasks, PDFs, instructions, and a sample submission.
- `amitbcp/docinsights-2026-shared-task-submissions`: private dataset repo for hidden validation labels, raw submissions, and leaderboard state.
- `amitbcp/docsem-docinsights`: Gradio Space for submission scoring and the public leaderboard.

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

To prepare an intentional leaderboard reset as part of a refreshed release:

```bash
/Users/aamita/miniconda3/bin/python scripts/prepare_docsem_hf_dataset.py --reset-leaderboard
```

The published dataset includes:

- 908 train tasks
- 908 train labels
- 217 validation tasks without public labels
- train and validation PDFs

The private submissions dataset includes:

- 217 hidden validation labels

Generated dataset payloads, private labels, submissions, and leaderboard state are intentionally ignored by this GitHub repository. Publish them only to the dedicated Hugging Face repositories.

## Participant Loading Example

```python
from datasets import load_dataset
from huggingface_hub import hf_hub_download

repo_id = "amitbcp/docinsights-2026-shared-task-data"

tasks = load_dataset(repo_id, "tasks")
train_tasks = tasks["train"]
val_tasks = tasks["validation"]

train_labels = load_dataset(repo_id, "labels")["train"]

pdf_path = hf_hub_download(
    repo_id=repo_id,
    repo_type="dataset",
    filename=val_tasks[0]["document_pdf"],
)
```

## Submission Format

Participants submit JSONL for validation, one object per instance:

```json
{"instance_id":"task_000909","answer":"140","evidence":["b14"]}
```

## Publishing

Upload the public dataset. Use the resumable large-folder uploader because the
release contains 1,125 PDFs:

```bash
/Users/aamita/miniconda3/bin/huggingface-cli upload-large-folder amitbcp/docinsights-2026-shared-task-data competition/hf-dataset --repo-type dataset --num-workers 4
```

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
