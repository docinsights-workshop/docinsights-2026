# DocSem dual-split disabled deployment receipt — 2026-09-05

## Outcome

The reviewed dual-split participant workflow is deployed in disabled-test mode.
Anonymous validation remains the default and the existing validation leaderboard
is live. The held-out test submission button is disabled, and the final-test
leaderboard view is notice-only with no score table.

No held-out test data, test labels, test attempts, or test leaderboard rows were
published by this deployment.

## Revisions

- Workshop GitHub `main`: `73525f758ad425b3ca1f9f847757bd060f59f0d2`
- Public participant Space: `f4f157b77d74eac667f1a368a43083fec9520bb8`
- Private submissions dataset inspected at:
  `6762afb4808ad569c776bbfc98fb8c8e814f96f9`
- Canonical `oracle-samples/gsm-sem` `main` inspected at:
  `feb256e108afca98f24d58b5f9019b6c36ca31ea`

## Public participant verification

Verified at `2026-09-05T13:05:16Z`:

- Space visibility: public
- Space SDK/runtime: Gradio `4.42.0`, Python `3.12`
- Space runtime stage: `RUNNING`
- Deployed source inventory: exactly eight production files
- Root, `/config`, and `/info`: HTTP 200
- Evaluation-split default: `Validation (development)`
- Leaderboard-view default: `Validation leaderboard`
- Validation leaderboard: 56 rows observed at verification time
- Test split: `Test submissions are not open yet`
- Test submit control: disabled
- Final test leaderboard: unpublished notice visible; zero tables rendered
- Test configuration flags: absent and therefore false by the checked-in
  fail-closed defaults

The workshop shared-task and FAQ pages both returned HTTP 200 and contained the
new held-out-test policy, three-attempt description, and separate validation and
final-test leaderboard descriptions.

## Data and privacy verification

- The freshly fetched canonical source contains no `docsem/test` tree.
- The private submissions dataset reports private visibility.
- No `private/test_*`, `attempts/test/`, `projections/test/`,
  `exclusions/test/`, or `adjudications/test/` paths were present.
- The public Hugging Face dataset was not modified because no authoritative test
  payload exists; its train/validation configuration therefore remains loadable.
- Validation/test labels, OAuth subjects, verified emails, raw predictions, and
  per-example test metrics are absent from this receipt.

## Deployment recovery

Two intermediate public Space revisions failed safely and were superseded:

1. `031e6bb15e8bdd4f9a3ff46019c50635dc8ea34d` exposed a build-time Gradio
   version conflict. The Space metadata now pins `sdk_version: 4.42.0`.
2. `3709dc07bfcce3a05bdb927b7e7a9f374f8a774c` reached Python 3.13 startup,
   where Gradio 4.42's audio dependency lacked `audioop`. The metadata now pins
   `python_version: "3.12"`.

The final revision above built and passed live UI verification.

## Remaining gates

The private organizer Space source and guarded publisher are complete and
tested, but the live private Space was not created. The operator's secure Hugging
Face store currently provides only one owner write token. Full access-boundary
verification additionally requires:

- a distinct classic owner read-only token for the organizer runtime; and
- a distinct authenticated non-owner token used only to prove access denial.

Neither token may be stored in Git or printed in logs. Until both are available,
the publisher correctly refuses to claim a verified private organizer deployment.

Test activation is independently blocked until an authoritative test release is
present and the admission close-fence is resolved. Keep
`TEST_SUBMISSIONS_ENABLED=false` and
`TEST_PUBLIC_LEADERBOARD_ENABLED=false`.
