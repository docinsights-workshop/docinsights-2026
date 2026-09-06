---
title: "DocInsights 2026 Shared Task: DocSem"
sdk: gradio
sdk_version: 4.42.0
python_version: "3.12"
app_file: app.py
license: apache-2.0
hf_oauth: true
hf_oauth_scopes:
  - email
---

# DocInsights 2026 Shared Task: DocSem

DocSem is the document-grounded quantitative reasoning shared task of [DocInsights 2026](https://docinsights-workshop.github.io/docinsights-2026/shared-task/), the Workshop on Document Intelligence and Understanding at EMNLP 2026 in Budapest, Hungary.

[Workshop shared task](https://docinsights-workshop.github.io/docinsights-2026/shared-task/) | [Public dataset](https://huggingface.co/datasets/amitbcp/docinsights-2026-shared-task-data) | [GitHub source](https://github.com/oracle-samples/gsm-sem) | [Participant guide](https://github.com/oracle-samples/gsm-sem/blob/main/docsem/PARTICIPANT_INSTRUCTIONS.md)

Use the split selector to upload a validation `submission.jsonl` file or, when the portal's authoritative status notice says the test window is open, submit final test predictions. Validation remains available without signing in. Hugging Face sign-in is optional and strongly recommended for test submissions and `My test submissions`: signed-in attempts are keyed to the immutable HF account subject and use the verified profile email privately. Signed-out attempts require a valid contact email and are keyed to its normalized form.

The canonical release is maintained under [`docsem/` in `oracle-samples/gsm-sem`](https://github.com/oracle-samples/gsm-sem/tree/main/docsem). It provides 908 labelled training tasks and 217 unlabelled validation tasks.

## Dataset update and final evaluation

The training split was updated on **August 31, 2026** to correct seven annotation inconsistencies identified through community feedback. These changes affect only the training data; the task definition and data format are unchanged. Pull the latest version before training or comparing results.

Three organizer-only validation ground-truth labels have now been corrected, most recently on **September 3, 2026**, following additional data review. All existing submissions were rescored, and the leaderboard now reflects the updated results. Public validation inputs, the task definition, and the data format are unchanged.

The current leaderboard contains provisional validation results. **The held-out test inputs have been released:** public test tasks and PDFs are available in the [public dataset](https://huggingface.co/datasets/amitbcp/docinsights-2026-shared-task-data). The portal derives its live open, scheduled, or closed notice and countdown from the validated server deployment policy. Performance on the held-out test set will determine the final leaderboard.

When the authoritative portal notice says the submission window is open, it permits up to **3 accepted test submissions per identity**—a Hugging Face account when signed in, or a normalized contact email when signed out. Attempt 1 reports aggregate **Joint Exact Accuracy, Answer Exact Accuracy, and Evidence F1 (macro)** that are private to that submitting identity; attempts 2–3 are accepted with their metrics withheld. During the open window, provisional public ranks use only attempt 1 and display no metrics. After the window closes, the final ranking uses the best of all 3 eligible attempts. The quota is not per team or person: separate HF accounts receive separate quotas, and alternate anonymous emails cannot be prevented from receiving separate quotas.

The portal has separate **Validation leaderboard** and **Final test leaderboard** views. The validation view remains available throughout the competition. Before finalization, the test view exposes only provisional ranks derived from each identity's first accepted attempt—never metric values, per-example results, later-attempt scores, contact details, identity subjects, participant names, or predictions. Anonymous rows display `Not signed in` in the Hugging Face account column and never expose the contact email. After finalization and explicit operator activation, the portal reads a sanitized projection from the exact private repository head and publishes the selected best-of-three rows.

The collapsed **How metrics are computed** section in the portal defines the evaluation precisely. Answer Exact Accuracy is the mean exact normalized answer match. Evidence precision and recall are computed from set overlap for each task; their harmonic mean is macro-averaged across tasks as Evidence F1 (macro), so partial evidence receives partial credit. Joint Exact Accuracy is the mean of answer exact **and** evidence-set exact on the same task. It neither combines aggregate percentages nor uses Evidence F1. Rankings use Joint Exact Accuracy, Answer Exact Accuracy, Evidence F1, accepted time, and stable submission ID, in that order. A legacy row with no Joint value is displayed as `Not yet computed`, never as zero.

## Disabled test-release configuration

The checked-in deployment is safe to publish with both test controls disabled:

```text
VALIDATION_SUBMISSIONS_ENABLED=true
TEST_SUBMISSIONS_ENABLED=false
TEST_PROVISIONAL_LEADERBOARD_ENABLED=false
TEST_PUBLIC_LEADERBOARD_ENABLED=false
```

`VALIDATION_SUBMISSIONS_ENABLED` defaults to `true` for compatibility with the
existing validation workflow. Set it to `false` only for a bounded organizer
maintenance window: the server then rejects validation submissions before
reading the uploaded file or invoking scoring/persistence, while the existing
validation leaderboard remains readable.

Do not enable either flag for a candidate or partially prepared release. Activation is fail-closed: a requested test surface remains disabled unless all of the following are explicit and valid. These values are deployment secrets/configuration, never participant inputs or rendered Space configuration.

```text
TEST_RELEASE_ID=<official identifier using letters, digits, dot, underscore, or hyphen>
TEST_TASK_MANIFEST_SHA256=<64 lowercase hexadecimal SHA-256>
TEST_GOLD_SHA256=<64 lowercase hexadecimal SHA-256>
TEST_OPEN_AT=YYYY-MM-DDTHH:MM:SSZ
TEST_CLOSE_AT=2026-09-11T12:00:00Z
TEST_MAX_ATTEMPTS=3
TEST_RELEASE_CONFIG_PATH=private/test_release.json
TEST_GOLD_CONFIG_PATH=private/test_labels.jsonl
TEST_TASKS_FILE=test/tasks.jsonl
```

The timestamps must be RFC3339 UTC values ending in `Z`, and the open instant must precede the close instant. The official close value is the exclusive instant immediately after September 10, 2026 at 23:59:59 Anywhere on Earth. The three paths are fixed, server-selected canonical paths; any alternate path fails closed and is never accepted from a participant request. The release identifier, digests, exact normalized UTC window, attempt limit, first-attempt-only feedback policy, and task path must agree exactly with the organizer-pinned server release. The server independently verifies the private scoring material before accepting an upload. `TEST_MAX_ATTEMPTS` is fixed at three. A malformed/missing value, a non-UTC or reversed window, noncanonical path, or any attempt-limit value other than `3` leaves test submission and the public-test flag disabled without changing anonymous validation or the validation leaderboard.

`TEST_PROVISIONAL_LEADERBOARD_ENABLED` and `TEST_PUBLIC_LEADERBOARD_ENABLED` are independent. During the open window, enable only the provisional flag; the Space reads the active release and rank-only provisional projection from one exact private repository head and accepts only the exact public row fields `rank`, `hf_username`, and `team`. Keep `TEST_PUBLIC_LEADERBOARD_ENABLED=false` until organizer finalization. Enabling it is not a substitute for finalization: the server also requires a private exact-SHA repository snapshot, a finalized and disabled release, matching configured release/task/gold digests, and matching final-projection and audit hashes. It downloads only the fixed server-selected release, sanitized final projection, and finalization audit paths. It never accepts a client-provided split or path and never exposes private per-example results, participant emails, identity subjects, raw predictions, unselected attempts, or later-attempt scores.

The submission form requires participant name(s), team name, contact email, and submission name. Participant names and contact emails are stored only in the private submission repository and are never rendered on the public leaderboard.

The validation leaderboard shows the latest submission for each normalized team and contact-email identity, together with its total attempt count. Rankings use Joint Exact Accuracy first, Answer Exact Accuracy second, and Evidence F1 (macro) as the tie-breaker. A new valid attempt replaces that identity's previously displayed result even when its score is lower. Legacy submissions without participant names remain valid. Evidence exact match is retained in the detailed submission result as a strict diagnostic, but it is not a separate public leaderboard column.

For live competition use, configure the Space secrets:

- `GOLD_REPO_ID`
- `GOLD_FILE`
- `SUBMISSIONS_REPO_ID`
- `HF_WRITE_TOKEN`

The public task data is expected in `PUBLIC_DATASET_REPO`.

## Safari post-deployment smoke check

After every public Space deployment, open the running app directly in desktop Safari and verify that the page's single vertical scrollbar stays at the browser edge. Select each split, upload a disposable local JSONL file without submitting it, and confirm the uploader has no nested vertical scrollbar. Finally, narrow the Safari window to a mobile-sized viewport and confirm that links wrap and there is no page-level horizontal scrollbar. This visual check supplements the pinned Gradio DOM and configuration regression tests; it must not submit or mutate competition data.
