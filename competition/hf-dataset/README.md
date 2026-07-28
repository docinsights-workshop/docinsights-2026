---
pretty_name: "DocInsights 2026 Shared Task: DocSem"
license: cc-by-4.0
language:
- en
task_categories:
- question-answering
tags:
- document-ai
- document-understanding
- quantitative-reasoning
- pdf
- docsem
- docinsights-2026
- shared-task
configs:
- config_name: tasks
  data_files:
  - split: train
    path: train/tasks.jsonl
  - split: validation
    path: val/tasks.jsonl
- config_name: labels
  data_files:
  - split: train
    path: train/labels.jsonl
---

# DocInsights 2026 Shared Task: DocSem

**Document-grounded quantitative reasoning with evidence attribution**

DocSem is the shared task of [DocInsights 2026](https://docinsights-workshop.github.io/docinsights-2026/), the Workshop on Document Intelligence and Understanding co-located with EMNLP 2026 in Budapest, Hungary. The workshop theme is **Beyond Plain Text: Bridging NLP and Document AI**.

[Workshop website](https://docinsights-workshop.github.io/docinsights-2026/) | [Submission portal](https://amitbcp-docsem-docinsights.hf.space/) | [Participant instructions](./INSTRUCTIONS.md)

Participants receive a PDF document and a paraphrased `user_query`. Systems must locate the relevant quantitative passage in the PDF and submit the final numerical answer plus supporting PDF block IDs.

## Shared Task At A Glance

- **Input:** a PDF document and a document-grounded quantitative question.
- **Output:** a final numerical answer and the supporting PDF block IDs.
- **Training data:** 908 tasks with labels and PDFs.
- **Validation data:** 217 tasks with PDFs; labels remain private for leaderboard evaluation.
- **Evaluation:** normalized answer exact match, evidence exact match, and evidence F1.

Submit validation predictions through the [official DocSem submission portal](https://amitbcp-docsem-docinsights.hf.space/).

## Splits

Use config `tasks` for public inputs:

- `train`: 908 labelled training task inputs with PDFs.
- `validation`: 217 validation task inputs with PDFs and no public labels.

Use config `labels` for public labels:

- `train`: 908 training labels with `answer` and `evidence`.

Validation labels are not included in this public dataset. They are stored in the private organizer evaluation repository and used only by the submission portal.

## Files

- `train/tasks.jsonl`: public training manifest.
- `train/labels.jsonl`: public training labels.
- `train/documents/*.pdf`: training PDFs.
- `val/tasks.jsonl`: public validation manifest.
- `val/documents/*.pdf`: validation PDFs.
- `examples/sample_val_submission.jsonl`: example validation submission shape.
- `INSTRUCTIONS.md`: participant instructions copied from the source release.

Each task row has:

```json
{
  "instance_id": "task_000001",
  "user_query": "According to the brief...",
  "document_pdf": "train/documents/task_000001.pdf"
}
```

Each label row has:

```json
{
  "instance_id": "task_000001",
  "answer": "10",
  "evidence": ["b10"]
}
```

## Loading With Python

```python
from datasets import load_dataset
from huggingface_hub import hf_hub_download

repo_id = "amitbcp/docinsights-2026-shared-task-data"

tasks = load_dataset(repo_id, "tasks")
train_tasks = tasks["train"]
val_tasks = tasks["validation"]

train_labels = load_dataset(repo_id, "labels")["train"]

first_pdf = hf_hub_download(
    repo_id=repo_id,
    repo_type="dataset",
    filename=train_tasks[0]["document_pdf"],
)
print(first_pdf)
```

## Submission Format

Submit one JSON object per validation instance in JSONL format:

```json
{"instance_id":"task_000909","answer":"140","evidence":["b14"]}
```

Requirements:

- `instance_id` must match a validation input row.
- `answer` must contain only the final answer.
- `evidence` must be a non-empty list of visible PDF block IDs such as `b01`.
- Include every validation instance exactly once.

See `examples/sample_val_submission.jsonl`.

## Evaluation

The primary metric is normalized exact-match accuracy on `answer`. Evidence is evaluated separately with exact block-set match and evidence F1 diagnostics.

## Workshop

DocInsights 2026 brings together researchers and practitioners working across NLP, Document AI, multimodal learning, information retrieval, knowledge representation, and human-centered systems.

- [DocInsights 2026 workshop website](https://docinsights-workshop.github.io/docinsights-2026/)
- [DocSem submission portal and leaderboard](https://amitbcp-docsem-docinsights.hf.space/)
- [EMNLP 2026 workshop listing](https://2026.emnlp.org/calls/workshops/)
