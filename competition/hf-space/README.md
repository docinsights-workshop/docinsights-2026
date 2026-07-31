---
title: "DocInsights 2026 Shared Task: DocSem"
sdk: gradio
app_file: app.py
license: apache-2.0
---

# DocInsights 2026 Shared Task: DocSem

DocSem is the document-grounded quantitative reasoning shared task of [DocInsights 2026](https://docinsights-workshop.github.io/docinsights-2026/shared-task/), the Workshop on Document Intelligence and Understanding at EMNLP 2026 in Budapest, Hungary.

[Workshop shared task](https://docinsights-workshop.github.io/docinsights-2026/shared-task/) | [Public dataset](https://huggingface.co/datasets/amitbcp/docinsights-2026-shared-task-data) | [GitHub source](https://github.com/oracle-samples/gsm-sem) | [Participant guide](https://github.com/oracle-samples/gsm-sem/blob/main/docsem/PARTICIPANT_INSTRUCTIONS.md)

Upload a validation `submission.jsonl` file to validate the submission schema, compute the shared-task score, and appear on the leaderboard.

The canonical release is maintained under [`docsem/` in `oracle-samples/gsm-sem`](https://github.com/oracle-samples/gsm-sem/tree/main/docsem). It provides 908 labelled training tasks and 217 unlabelled validation tasks.

The public leaderboard shows the best submission for each normalized team and contact-email identity, together with its total attempt count. Contact emails remain private. Rankings use answer accuracy first and evidence F1 as the tie-breaker. If two attempts have identical public scores, the more recent attempt is shown. Evidence exact match is retained in the detailed submission result as a strict diagnostic, but it is not a separate public leaderboard column.

For live competition use, configure the Space secrets:

- `GOLD_REPO_ID`
- `GOLD_FILE`
- `SUBMISSIONS_REPO_ID`
- `HF_WRITE_TOKEN`

The public task data is expected in `PUBLIC_DATASET_REPO`.
