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
- **Held-out test:** not released in the current public payload. After the organizers publish an audited release, the `test` task split will contain tasks and PDFs without labels.
- **Evaluation:** normalized answer exact match and exact evidence block-set match, with evidence F1 reported as a diagnostic.

Submit predictions through the [official DocSem submission portal](https://amitbcp-docsem-docinsights.hf.space/). Validation remains available through the established anonymous-compatible workflow. Test submission stays closed until the official public manifest and private scoring release have both been audited and activated.

## Source And Provenance

The canonical participant release is the [`docsem/` directory in `oracle-samples/gsm-sem`](https://github.com/oracle-samples/gsm-sem/tree/main/docsem). This mirror tracks revision [`feb256e`](https://github.com/oracle-samples/gsm-sem/commit/feb256e108afca98f24d58b5f9019b6c36ca31ea), including the seven training annotation corrections published on August 31, 2026 following community feedback. The task definition, split sizes, and data format are unchanged; see the [source changelog](https://github.com/oracle-samples/gsm-sem/blob/main/docsem/CHANGELOG.md).

This Hugging Face package mirrors the source release's 908 training tasks, 908 training labels, 217 validation tasks, and all 1,125 PDFs. The PDF files are byte-identical. The Hugging Face task manifests only prefix `document_pdf` with `train/` or `val/` so files resolve directly from this repository's root.

No official held-out test payload is present in this revision, so the tracked dataset-card configuration intentionally declares only the existing train and validation task files. The release generator adds the `test` configuration to a deterministic release-card payload only after an explicitly selected public staging tree passes the complete audit. It must not be populated from similarly named local directories or archives.

## Changelog

### 2026-08-31

- Corrected seven annotation inconsistencies in the training split following community feedback.

### 2026-09-03

- Three organizer-only validation ground-truth labels have now been corrected following additional data review. The public validation tasks and PDFs, task definition, and data format are unchanged; all existing submissions were rescored in the private evaluation repository.

### 2026-09-04

- Declared the held-out `test` task-split contract and participant policy. The official test files are not included in this revision and the test submission window is not open.

Copyright (c) 2026 Oracle and/or its affiliates. The participant release and this mirror are provided under the [Universal Permissive License v1.0](./LICENSE.txt).

## Splits

Use config `tasks` for public inputs:

- `train`: 908 labelled training task inputs with PDFs.
- `validation`: 217 validation task inputs with PDFs and no public labels.
- `test`: held-out test task inputs with PDFs and no public labels, available only after the audited release is published.

Use config `labels` for public labels:

- `train`: 908 training labels with `answer` and `evidence`.

Validation and test labels are not included in this public dataset. They remain in the access-restricted organizer evaluation repository and are used only by the submission services.

## Files

- `train/tasks.jsonl`: public training manifest.
- `train/labels.jsonl`: public training labels.
- `train/documents/*.pdf`: training PDFs.
- `val/tasks.jsonl`: public validation manifest.
- `val/documents/*.pdf`: validation PDFs.
- `test/tasks.jsonl`: public held-out test manifest, once released.
- `test/documents/*.pdf`: public held-out test PDFs, once released.
- `test/SHA256SUMS`: checksums for the released public test payload.
- `test/release.json`: sanitized release identifier, counts, and aggregate digests.
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
# After the official release, reloading this config will also provide tasks["test"].

train_labels = load_dataset(repo_id, "labels")["train"]

first_pdf = hf_hub_download(
    repo_id=repo_id,
    repo_type="dataset",
    filename=train_tasks[0]["document_pdf"],
)
print(first_pdf)
```

## Submission Format

Submit one JSON object per selected-split instance in JSONL format:

```json
{"instance_id":"task_000909","answer":"140","evidence":["b14"]}
```

Requirements:

- `instance_id` must match an input row from the selected split.
- `answer` must contain only the final answer.
- `evidence` must be a non-empty list of visible PDF block IDs such as `b01`.
- Include every instance from the selected split exactly once.

See `examples/sample_val_submission.jsonl`.

### Held-out test submission policy

The test workflow is available only after the organizers announce that the audited test release and submission window are open.

- Sign in through Hugging Face OAuth to submit test predictions.
- Each Hugging Face account may make at most three valid test submissions. The quota follows the authenticated account, not a typed email address or self-entered team name.
- Attempt 1 returns aggregate answer accuracy and evidence F1 and remains visible in that account's `My test submissions` history.
- Attempts 2 and 3 receive accepted receipts, but their scores are withheld until finalization.
- The best of the three attempts, ordered first by answer accuracy and then evidence F1 with deterministic tie-breaks, is used for the final test ranking.
- No public test scores or ranks are displayed while the test window is open. Detailed live test results are visible only in the private organizer Space.

The public release contains only tasks, PDFs, checksums, and sanitized release metadata. It never contains validation or test labels, correct answers, evidence gold, per-example correctness, or organizer notes.

## Evaluation

The primary metric is normalized exact-match accuracy on `answer`. Evidence is evaluated separately with exact block-set match and evidence F1 diagnostics. Only answer accuracy and evidence F1 are returned for the first held-out test attempt; later-attempt metrics and all per-example test results remain organizer-only until finalization.

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
