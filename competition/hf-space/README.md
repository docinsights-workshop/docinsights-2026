---
title: "DocInsights 2026 Shared Task: DocSem"
sdk: gradio
sdk_version: 4.42.0
app_file: app.py
license: apache-2.0
hf_oauth: true
hf_oauth_scopes:
  - email
---

# DocInsights 2026 Shared Task: DocSem

DocSem is the document-grounded quantitative reasoning shared task of [DocInsights 2026](https://docinsights-workshop.github.io/docinsights-2026/shared-task/), the Workshop on Document Intelligence and Understanding at EMNLP 2026 in Budapest, Hungary.

[Workshop shared task](https://docinsights-workshop.github.io/docinsights-2026/shared-task/) | [Public dataset](https://huggingface.co/datasets/amitbcp/docinsights-2026-shared-task-data) | [GitHub source](https://github.com/oracle-samples/gsm-sem) | [Participant guide](https://github.com/oracle-samples/gsm-sem/blob/main/docsem/PARTICIPANT_INSTRUCTIONS.md)

Use the split selector to upload a validation `submission.jsonl` file or, when the official held-out release opens, submit final test predictions. Validation remains available without signing in. Test submission and `My test submissions` use Hugging Face OAuth so test receipts and attempt limits stay bound to the signed-in account.

The canonical release is maintained under [`docsem/` in `oracle-samples/gsm-sem`](https://github.com/oracle-samples/gsm-sem/tree/main/docsem). It provides 908 labelled training tasks and 217 unlabelled validation tasks.

## Dataset update and final evaluation

The training split was updated on **August 31, 2026** to correct seven annotation inconsistencies identified through community feedback. These changes affect only the training data; the task definition and data format are unchanged. Pull the latest version before training or comparing results.

Three organizer-only validation ground-truth labels have now been corrected, most recently on **September 3, 2026**, following additional data review. All existing submissions were rescored, and the leaderboard now reflects the updated results. Public validation inputs, the task definition, and the data format are unchanged.

The current leaderboard contains provisional validation results. The official held-out test release is not available yet, and test submissions remain closed while the organizers complete the release and integrity checks. Participants will be notified when it is available. Performance on the held-out test set will determine the final leaderboard.

The test workflow is deployed disabled until the organizers publish and pin the official test tasks and private scoring release. When it opens, each signed-in Hugging Face account can make up to three accepted test submissions. The first accepted attempt returns answer accuracy and evidence F1 and can be retrieved later with the same login; scores for attempts two and three are withheld until organizer finalization. The best eligible attempt per account is used for the final ranking. The quota is per Hugging Face account, not per team or person, so team members using separate authenticated accounts are treated as separate accounts.

The portal has separate **Validation leaderboard** and **Final test leaderboard** views. The validation view remains available throughout the competition. The final-test view is a notice only until organizer finalization; it exposes no test rows, ranks, per-example results, later-attempt scores, contact details, OAuth subjects, participant names, or predictions. After finalization and explicit operator activation, it reads one sanitized seven-field projection from the exact private repository head and publishes the selected best-of-three rows.

## Disabled test-release configuration

The checked-in deployment is safe to publish with both test controls disabled:

```text
TEST_SUBMISSIONS_ENABLED=false
TEST_PUBLIC_LEADERBOARD_ENABLED=false
```

Do not enable either flag for a candidate or partially prepared release. Activation is fail-closed: a requested test surface remains disabled unless all of the following are explicit and valid. These values are deployment secrets/configuration, never participant inputs or rendered Space configuration.

```text
TEST_RELEASE_ID=<official identifier using letters, digits, dot, underscore, or hyphen>
TEST_TASK_MANIFEST_SHA256=<64 lowercase hexadecimal SHA-256>
TEST_GOLD_SHA256=<64 lowercase hexadecimal SHA-256>
TEST_OPEN_AT=YYYY-MM-DDTHH:MM:SSZ
TEST_CLOSE_AT=YYYY-MM-DDTHH:MM:SSZ
TEST_MAX_ATTEMPTS=3
TEST_RELEASE_CONFIG_PATH=private/test_release.json
TEST_GOLD_CONFIG_PATH=private/test_labels.jsonl
TEST_TASKS_FILE=test/tasks.jsonl
```

The timestamps must be RFC3339 UTC values ending in `Z`, and the open instant must precede the close instant. The three paths are fixed, server-selected canonical paths; any alternate path fails closed and is never accepted from a participant request. The release identifier, digests, exact normalized UTC window, attempt limit, first-attempt-only feedback policy, and task path must agree exactly with the organizer-pinned server release. The server independently verifies the private scoring material before accepting an upload. `TEST_MAX_ATTEMPTS` is fixed at three. A malformed/missing value, a non-UTC or reversed window, noncanonical path, or any attempt-limit value other than `3` leaves test submission and the public-test flag disabled without changing anonymous validation or the validation leaderboard.

Keep `TEST_PUBLIC_LEADERBOARD_ENABLED=false` until organizer finalization. Enabling it is not a substitute for finalization: the server also requires a private exact-SHA repository snapshot, a finalized and disabled release, matching configured release/task/gold digests, and matching final-projection and audit hashes. It downloads only the fixed server-selected release, sanitized final projection, and finalization audit paths. It never accepts a client-provided split or path and never exposes private per-example results, participant emails, OAuth subjects, raw predictions, unselected attempts, or later-attempt scores.

The submission form requires participant name(s), team name, contact email, and submission name. Participant names and contact emails are stored only in the private submission repository and are never rendered on the public leaderboard.

The public leaderboard shows the latest submission for each normalized team and contact-email identity, together with its total attempt count. Rankings use the latest attempt's answer accuracy first and evidence F1 as the tie-breaker. A new valid attempt replaces that identity's previously displayed result even when its score is lower. Legacy submissions without participant names remain valid. Evidence exact match is retained in the detailed submission result as a strict diagnostic, but it is not a separate public leaderboard column.

For live competition use, configure the Space secrets:

- `GOLD_REPO_ID`
- `GOLD_FILE`
- `SUBMISSIONS_REPO_ID`
- `HF_WRITE_TOKEN`

The public task data is expected in `PUBLIC_DATASET_REPO`.
