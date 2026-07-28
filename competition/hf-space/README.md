---
title: "DocInsights 2026 Shared Task: DocSem"
sdk: gradio
app_file: app.py
license: apache-2.0
---

# DocInsights 2026 Shared Task: DocSem

DocSem is the document-grounded quantitative reasoning shared task of [DocInsights 2026](https://docinsights-workshop.github.io/docinsights-2026/), the Workshop on Document Intelligence and Understanding at EMNLP 2026 in Budapest, Hungary.

[Workshop website](https://docinsights-workshop.github.io/docinsights-2026/) | [Public dataset](https://huggingface.co/datasets/amitbcp/docinsights-2026-shared-task-data) | [Submission portal](https://amitbcp-docsem-docinsights.hf.space/)

Upload a validation `submission.jsonl` file to validate the submission schema, compute the shared-task score, and appear on the leaderboard.

For live competition use, configure the Space secrets:

- `GOLD_REPO_ID`
- `GOLD_FILE`
- `SUBMISSIONS_REPO_ID`
- `HF_WRITE_TOKEN`

The public task data is expected in `PUBLIC_DATASET_REPO`.
