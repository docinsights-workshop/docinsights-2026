---
pretty_name: "DocInsights 2026 Shared Task: DocSem"
license: other
license_name: universal-permissive-license-1.0
license_link: https://github.com/oracle-samples/gsm-sem/blob/main/LICENSE.txt
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

DocSem is the shared task of [DocInsights 2026](https://docinsights-workshop.github.io/docinsights-2026/shared-task/), the Workshop on Document Intelligence and Understanding co-located with EMNLP 2026 in Budapest, Hungary. The workshop theme is **Beyond Plain Text: Bridging NLP and Document AI**.

[Workshop shared task](https://docinsights-workshop.github.io/docinsights-2026/shared-task/) | [Source repository](https://github.com/oracle-samples/gsm-sem) | [Submission portal](https://amitbcp-docsem-docinsights.hf.space/) | [Participant guide](https://github.com/oracle-samples/gsm-sem/blob/main/docsem/PARTICIPANT_INSTRUCTIONS.md)

Participants receive a PDF document and a paraphrased `user_query`. Systems must locate the relevant quantitative passage in the PDF and submit the final numerical answer plus supporting PDF block IDs.

## Shared Task At A Glance

- **Input:** a PDF document and a document-grounded quantitative question.
- **Output:** a final numerical answer and the supporting PDF block IDs.
- **Training data:** 908 tasks with labels and PDFs.
- **Validation data:** 217 tasks with PDFs; labels remain private for leaderboard evaluation.
- **Evaluation:** normalized answer exact match and exact evidence block-set match, with evidence F1 reported as a diagnostic.

Submit validation predictions through the [official DocSem submission portal](https://amitbcp-docsem-docinsights.hf.space/).

## Source And Provenance

The canonical participant release is the [`docsem/` directory in `oracle-samples/gsm-sem`](https://github.com/oracle-samples/gsm-sem/tree/main/docsem). This mirror tracks revision [`332158b`](https://github.com/oracle-samples/gsm-sem/commit/332158b2549e7e8a1186e2ae3a922669e9018808), including the refreshed task PDFs merged into the default `main` branch.

This Hugging Face package mirrors the source release's 908 training tasks, 908 training labels, 217 validation tasks, and all 1,125 PDFs. The PDF files are byte-identical. The Hugging Face task manifests only prefix `document_pdf` with `train/` or `val/` so files resolve directly from this repository's root.

Copyright (c) 2026 Oracle and/or its affiliates. The participant release and this mirror are provided under the [Universal Permissive License v1.0](./LICENSE.txt).

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
- `LICENSE.txt`: Universal Permissive License v1.0.

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

## Citation

If you use this shared-task release, cite the originating GSM-SEM paper:

```bibtex
@article{singh2026gsmsem,
  title={GSM-SEM: Benchmark and Framework for Generating Semantically Variant Augmentations},
  author={Jyotika Singh and Fang Tu and Aziza Mirsaidova and Amit Agarwal and Hitesh Laxmichand Patel and Sandip Ghoshal and Miguel Ballesteros and Karan Dua and Yassine Benajiba and Weiyi Sun and Tao Sheng and Graham Horwood and Sujith Ravi and Dan Roth},
  year={2026},
  eprint={2605.07053},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2605.07053}
}
```

Citation source: [GSM-SEM on arXiv](https://arxiv.org/abs/2605.07053) and the [canonical DocSem release](https://github.com/oracle-samples/gsm-sem/tree/main/docsem#citation).

## Workshop

DocInsights 2026 brings together researchers and practitioners working across NLP, Document AI, multimodal learning, information retrieval, knowledge representation, and human-centered systems.

- [DocInsights 2026 shared task page](https://docinsights-workshop.github.io/docinsights-2026/shared-task/)
- [DocSem submission portal and leaderboard](https://amitbcp-docsem-docinsights.hf.space/)
