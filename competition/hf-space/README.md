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

## Dataset freeze and final evaluation

If you downloaded the problem set before **August 5, 2026**, pull the latest version. The development data was updated on August 5 and is now frozen; there will be no further updates to it.

The current leaderboard contains provisional validation results. A held-out test set will be released five days before the September 10, 2026 final submission deadline. Participants will be notified when it is available and asked to submit their test-set results. Performance on that held-out test set will determine the final leaderboard.

The submission form requires participant name(s), team name, contact email, and submission name. Participant names and contact emails are stored only in the private submission repository and are never rendered on the public leaderboard.

The public leaderboard shows the latest submission for each normalized team and contact-email identity, together with its total attempt count. Rankings use the latest attempt's answer accuracy first and evidence F1 as the tie-breaker. A new valid attempt replaces that identity's previously displayed result even when its score is lower. Legacy submissions without participant names remain valid. Evidence exact match is retained in the detailed submission result as a strict diagnostic, but it is not a separate public leaderboard column.

For live competition use, configure the Space secrets:

- `GOLD_REPO_ID`
- `GOLD_FILE`
- `SUBMISSIONS_REPO_ID`
- `HF_WRITE_TOKEN`

The public task data is expected in `PUBLIC_DATASET_REPO`.
