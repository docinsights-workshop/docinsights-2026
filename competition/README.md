# DocSem Shared Task HF Deployment

This folder contains source code and documentation for the DocInsights 2026 document-grounded quantitative reasoning shared task.

## Repositories

The current deployment targets:

- `amitbcp/docinsights-2026-shared-task-data`: public dataset repo for labelled train tasks, unlabelled validation tasks, PDFs, instructions, and a sample submission.
- `amitbcp/docinsights-2026-shared-task-submissions`: private dataset repo for hidden validation labels, raw submissions, and leaderboard state.
- `amitbcp/docsem-docinsights`: Gradio Space for submission scoring and the public leaderboard.

## Source Data

The package is generated from:

```text
/Users/aamita/Oracle/amitbcp/docsem-workshop-final-public
```

Regenerate local HF payloads with:

```bash
/Users/aamita/miniconda3/bin/python scripts/prepare_docsem_hf_dataset.py
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

Upload the public dataset and private evaluation store:

```bash
/Users/aamita/miniconda3/bin/huggingface-cli upload amitbcp/docinsights-2026-shared-task-data competition/hf-dataset . --repo-type dataset --commit-message "Upload DocSem public data"
/Users/aamita/miniconda3/bin/huggingface-cli upload amitbcp/docinsights-2026-shared-task-submissions competition/hf-submissions . --repo-type dataset --private --commit-message "Upload DocSem private validation labels"
```

The dynamic Gradio Space currently requires HF Pro or an organization with dynamic Space support for this account. Once enabled, publish `competition/hf-space` and configure:

```text
PUBLIC_DATASET_REPO=amitbcp/docinsights-2026-shared-task-data
GOLD_REPO_ID=amitbcp/docinsights-2026-shared-task-submissions
GOLD_FILE=private/val_labels.jsonl
SUBMISSIONS_REPO_ID=amitbcp/docinsights-2026-shared-task-submissions
HF_WRITE_TOKEN=<write token with access to private submissions repo>
```
