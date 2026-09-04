# DocSem Test Release and Dual-Split Scoring Design

**Date:** September 4, 2026
**Status:** Approved for implementation
**Scope:** Public test-data release, split-aware participant submission, protected test feedback, final ranking, and organizer-only monitoring

## 1. Objectives

This design extends the existing DocSem deployment without replacing or breaking the live validation workflow.

The system will:

1. Publish the held-out test inputs on the canonical GitHub repository and public Hugging Face dataset without publishing test labels.
2. Keep the current public Space as the single participant portal for both validation and test submissions.
3. Preserve all existing validation submissions, validation scoring behavior, and the public validation leaderboard.
4. Require Hugging Face OAuth only for test submissions.
5. Allow at most three valid, unique test attempts per authenticated Hugging Face account.
6. Return aggregate test metrics only for the first attempt and make that first score retrievable after login.
7. Withhold scores for attempts two and three while still recording them for final selection.
8. Select each authenticated account's best test attempt for the final ranking.
9. Keep live test rankings private until the test window closes.
10. Provide a separate private organizer Space for detailed test monitoring.

## 2. Non-Goals and Accepted Limitations

- There is no organizer-issued team identifier or team account.
- The three-attempt cap is per authenticated Hugging Face account, not per team.
- Multiple members of one team may use different Hugging Face accounts and receive separate quotas. This is an explicitly accepted limitation.
- The system will not attempt to infer or enforce team membership from names or email domains.
- Self-entered team names remain display metadata and are not an authorization boundary.
- No persistent SQL database or external service will be introduced.
- Post-hoc erasure of personal fields from immutable private Git history is outside this design.
- Existing validation records will not be rewritten merely to adopt the new test schema.
- The pre-existing local `test/`, `test_hard/`, `test_hard_1/`, and ZIP artifacts are not implicitly considered the official incoming test release. The release source must be explicitly selected when the final data arrives.

## 3. Chosen Architecture

The existing public Gradio Space remains the participant portal. The existing private Hugging Face dataset remains the durable scoring store. Test attempts are stored as immutable JSON records, and derived account/leaderboard projections are updated in the same compare-and-swap commit.

A separate private Hugging Face Space provides organizer-only views. It reads the private repository but is not a participant submission channel.

SQLite is not the source of truth. A Space-local SQLite database would be ephemeral and unsafe to upload as a shared binary after every submission. The organizer Space may build a temporary in-memory or local SQLite index from immutable JSON records to speed rendering, but that index is disposable and fully reconstructible.

## 4. Public and Private Data Boundaries

### 4.1 Canonical public GitHub release

The official GitHub release will add only:

```text
docsem/test/tasks.jsonl
docsem/test/documents/*.pdf
docsem/test/SHA256SUMS
```

Public test task rows contain exactly:

```json
{
  "instance_id": "opaque-id",
  "user_query": "document-grounded question",
  "document_pdf": "documents/opaque-id.pdf"
}
```

No public test file may contain `answer`, `evidence`, per-example correctness, source mappings, organizer notes, or labels.

### 4.2 Public Hugging Face dataset

The public dataset adds:

```text
test/tasks.jsonl
test/documents/*.pdf
test/SHA256SUMS
```

The dataset card's `tasks` configuration gains a `test` split. The `labels` configuration continues to expose only training labels. Validation and test labels must never appear in the public tree or reachable release history.

### 4.3 Private organizer repository

The private repository retains the current validation paths and adds test-specific paths:

```text
private/val_labels.jsonl
private/test_labels.jsonl
private/test_release.json
attempts/test/<account_key>/<submission_id>.json
projections/test/accounts/<account_key>.json
projections/test/organizer_leaderboard.json
projections/test/public_final.json
```

`private/test_release.json` records the release identifier, task-manifest digest, gold-label digest, attempt limit, feedback policy, open/close configuration, and whether finalization has occurred.

The existing validation paths remain unchanged:

```text
submissions/*.json
leaderboard/leaderboard.json
```

## 5. Authentication and Identity

The public Space enables Hugging Face OAuth with `openid`, `profile`, and `email` scopes.

- Validation submissions remain compatible with the existing anonymous/self-entered workflow. OAuth is optional for validation.
- Test submissions require a signed-in Hugging Face account.
- The immutable OpenID `sub` claim is the quota identity.
- A deterministic SHA-256 digest of `sub` is used as `account_key` in repository paths.
- The private attempt record retains the authenticated Hugging Face username and verified email for organizer audit.
- The participant never types an email to retrieve a test score.
- OAuth access tokens, refresh tokens, session cookies, and client secrets are never written to submission records or logs.

Changing the verified email on the same Hugging Face account does not reset the quota because `sub`, not email, controls the attempt count. Different Hugging Face accounts receive separate quotas even when their users claim the same team.

Account renames also leave the quota unchanged. The final public row uses the authenticated username captured with the selected attempt, while the private record retains the stable subject. Deleted accounts do not erase accepted attempts during the challenge audit period. Organizer smoke-test accounts are marked by a private exclusion record and never enter the final projection. Any disqualification or technical appeal creates an append-only private adjudication record; accepted attempts are never silently edited or deleted.

Every test handler, including direct Gradio API calls, requires a server-injected OAuth profile. Split names, identity values, email addresses, and attempt numbers supplied by a browser or API caller are treated as untrusted input and cannot select gold files, persistence paths, or quota identities.

## 6. Participant Portal Behavior

### 6.1 Split selection

The existing public Space gains a required split selector:

- `Validation (development)`
- `Test (final)`

Validation remains the default to preserve the current workflow. The selected split controls instructions, authentication requirements, expected IDs, gold file, feedback, persistence path, and visible leaderboard.

### 6.2 Validation flow

Validation behavior stays unchanged:

- No OAuth requirement.
- Existing team, participant-name, contact-email, and submission-name fields remain.
- Valid submissions receive the current aggregate metrics.
- Submissions remain unlimited.
- The existing public validation leaderboard remains visible and continues to rank the latest attempt per existing validation identity.
- Existing validation records remain readable without a schema migration.

### 6.3 Test flow

When `Test (final)` is selected:

- The user must sign in with Hugging Face.
- The verified email replaces the self-entered contact email for persistence and is displayed only in a masked form to that user.
- Team name, participant names, and submission name remain self-entered metadata.
- The UI shows the number of accepted attempts remaining.
- The backend, not the dropdown or browser, enforces every policy.

Attempt feedback is:

| Attempt | Participant response |
|---|---|
| 1 | Accepted receipt, answer accuracy, evidence F1, attempt number, timestamp |
| 2 | Accepted receipt, attempt number, timestamp; score withheld |
| 3 | Accepted receipt, attempt number, timestamp; score withheld |
| 4+ | Rejected before persistence; no score returned |

No test response exposes per-example metrics, correct IDs, evidence differences, missing-answer hints, rank changes, or whether a later attempt became the private best.

Attempt-one metrics use the existing scorer: answer accuracy and evidence F1 are persisted at six decimal places and displayed as percentages with two decimal places. Evidence exact match and per-example metrics remain organizer-only for every test attempt.

### 6.4 My test submissions

A signed-in participant can reopen the Space and view `My test submissions` for their OAuth account.

It displays:

- attempt number;
- submission name;
- accepted timestamp;
- receipt/submission identifier;
- the saved answer accuracy and evidence F1 for attempt one;
- `Score withheld until finalization` for attempts two and three.

Lookup is always bound to the OAuth `sub` claim. There is no typed-email score lookup.

## 7. Attempt Accounting and Idempotency

A test attempt is consumed only when all of the following succeed:

1. OAuth identity and verified email are present.
2. The test window is open and test submissions are enabled.
3. The submission parses successfully.
4. Its prediction IDs exactly match the pinned release's test task IDs.
5. Every row passes schema validation.
6. The private repository atomically accepts the attempt commit.

The server computes a canonical SHA-256 hash over the normalized prediction payload, split, release identifier, and OAuth `sub`.

- Repeating the same request or retrying after a timeout returns the existing receipt and does not consume another attempt.
- A different valid payload consumes the next attempt even if the submission name is unchanged.
- A fourth distinct valid payload is rejected.
- Filenames use UUID submission identifiers, not timestamps and slugs alone.

## 8. Atomic Persistence

Test persistence replaces the current two-write pattern with one compare-and-swap transaction implemented as a single Hugging Face commit.

For each request, the server:

1. Resolves one private-repository base SHA.
2. Loads the test release policy, pinned gold, existing account attempts, and projections at that SHA.
3. Checks idempotency and the three-attempt limit.
4. Scores against the pinned test gold.
5. Selects the account's best attempt.
6. Prepares the immutable attempt record, account projection, and organizer leaderboard projection.
7. Creates one commit with `parent_commit=<base SHA>`.
8. On a conflict, discards the plan, reloads the new SHA, and retries a bounded number of times.

A participant receives an accepted receipt only after the commit succeeds. Repository outages, exhausted conflicts, or scoring failures fail closed and do not consume an attempt.

The limit check is repeated after every compare-and-swap conflict. Therefore, four simultaneous valid submissions for one subject can produce at most attempts one through three; the losing request is rejected without receiving or persisting a score. Scoring may be computed before admission, but no result leaves the process until the atomic commit succeeds.

Each attempt record includes:

```json
{
  "schema_version": 2,
  "submission_id": "uuid",
  "release_id": "configured-test-release",
  "split": "test",
  "account_key": "sha256-of-oauth-sub",
  "hf_subject": "private-stable-subject",
  "hf_username": "authenticated-user",
  "verified_email": "private@example.org",
  "team": "self-entered display name",
  "participant_names": "private participant names",
  "submission_name": "run-name",
  "submitted_at": "server UTC timestamp",
  "submission_hash": "canonical-payload-sha256",
  "attempt_number": 1,
  "task_manifest_sha256": "public-manifest-digest",
  "gold_sha256": "private-gold-digest",
  "metrics": {},
  "predictions": []
}
```

All fields above remain private. Public projections are separately constructed from an explicit allowlist.

## 9. Best-of-Three and Final Ranking

The best attempt for one authenticated account is selected by:

1. answer accuracy, descending;
2. evidence F1, descending;
3. accepted timestamp, ascending;
4. submission ID, ascending.

During the test window:

- no public test scores or ranks are rendered;
- the public API does not return a test leaderboard;
- only the private organizer projection is updated.

After the test window closes, an organizer finalization command:

1. pins the private repository and release configuration;
2. verifies every accepted attempt against the pinned gold;
3. recomputes each account's selected best attempt independently;
4. creates `projections/test/public_final.json` from an allowlist;
5. marks the release finalized in the same compare-and-swap commit.

The public final test leaderboard contains only:

- rank;
- Hugging Face username;
- self-entered team name;
- selected submission name;
- selected attempt number;
- answer accuracy;
- evidence F1.

It does not contain email, OAuth subject, participant names, raw predictions, per-example metrics, or unselected-attempt scores. Multiple authenticated accounts claiming the same team remain separate rows and are distinguishable by Hugging Face username.

The close decision uses the server's configured UTC deadline. A test attempt is eligible only if the server rechecks that the window is open immediately before its compare-and-swap commit. Client timestamps are never authoritative. The selected attempt must belong to the frozen release and evaluator digests recorded in the private release manifest.

Test gold and evaluator behavior are frozen for the open window. A necessary label or evaluator correction triggers maintenance mode, a versioned incident record, deterministic recomputation of every accepted attempt, and a participant notice before reopening. No partial correction is permitted.

## 10. Organizer-Only Space

A separate private Space, `amitbcp/docsem-docinsights-organizer`, reads the private repository with a read-only token. Hugging Face repository visibility and an explicit organizer collaborator list provide the access boundary.

It shows:

- authenticated account and verified email;
- self-entered team and participant metadata;
- all accepted attempts and timestamps;
- aggregate and per-example metrics;
- the currently selected best attempt;
- submission and gold/task digests;
- malformed, duplicate, conflict, and audit status where available.

The organizer Space is read-only. Finalization and repair remain explicit CLI operations with dry-run, pinned-revision, maintenance, and compare-and-swap safeguards. A hidden tab in the public Space is not an organizer security boundary and will not be used.

Only the participant submission bot and designated organizers have write access to the private repository. Normal operations prohibit deletion, history rewrites, and force-pushes. Finalization records the private repository commit and creates a hash manifest that can reconstruct every derived view from immutable attempts and adjudication records.

## 11. Test Data Ingestion and Release Gates

The incoming test source is accepted only through an explicit source path. Nothing under the current organizer directories is selected implicitly.

Before release, the ingestion tool must prove:

- task IDs are unique;
- task IDs and PDF stems are a bijection;
- private label IDs exactly equal task IDs;
- test IDs do not overlap train or validation IDs;
- task rows contain only the public schema;
- labels contain only the private schema;
- all evidence values are valid non-empty block IDs;
- all PDFs are readable and contain visible evidence block identifiers;
- the public payload contains no label/gold/answer file or label fields;
- GitHub and Hugging Face public file hashes match;
- the private label digest matches the enabled test release policy;
- no ZIP containing labels is uploaded publicly.

The release process is:

1. Build and test the dual-split feature with `TEST_SUBMISSIONS_ENABLED=false`.
2. Receive the explicitly selected official test source.
3. Generate separate public and private payloads in permission-restricted staging directories.
4. Run structural, schema, PDF, hash, and leakage audits.
5. Publish tasks, PDFs, checksums, and documentation to the canonical GitHub release.
6. Publish the byte-identical public test payload to the public Hugging Face dataset.
7. Publish test labels and the release policy only to the private repository.
8. Deploy and verify the private organizer Space.
9. Smoke-test OAuth, attempt accounting, score withholding, retrieval, and organizer visibility against a staging release.
10. Set the official open/close configuration and enable test submissions.

The exact test deadline is a required deployment configuration derived from the official competition schedule. The application refuses to enable test submissions if the release ID, task digest, gold digest, open time, or close time is absent.

## 12. Feature Flags and Configuration

The public Space uses server-side configuration:

```text
TEST_SUBMISSIONS_ENABLED=false
TEST_PUBLIC_LEADERBOARD_ENABLED=false
TEST_MAX_ATTEMPTS=3
TEST_RELEASE_ID=required-release-manifest-id
TEST_TASKS_FILE=test/tasks.jsonl
TEST_GOLD_FILE=private/test_labels.jsonl
TEST_OPEN_AT=required-RFC3339-UTC-timestamp
TEST_CLOSE_AT=required-RFC3339-UTC-timestamp
```

OAuth metadata adds:

```yaml
hf_oauth: true
hf_oauth_scopes:
  - email
```

`openid` and `profile` remain included by Hugging Face. Test activation is fail-closed unless every required setting and digest matches.

## 13. Backward Compatibility

- Enabling OAuth on the Space does not require validation users to sign in.
- The validation form, current gold path, stored records, scoring response, and public leaderboard remain supported.
- Split routing defaults to validation and rejects cross-split IDs before persistence.
- Legacy validation records without `schema_version` or `split` are interpreted as validation records only.
- Test code never scans, ranks, resets, or rewrites legacy validation paths.
- Validation recomputation/reset tools must continue to require an explicit validation target and must never match test attempt paths.
- Existing public URLs remain unchanged.

## 14. Error and Abuse Handling

- Missing OAuth or verified email: reject test submission without reading or scoring the file.
- Closed/disabled test window: reject before reading or scoring.
- Invalid schema or ID set: return a generic validation error; consume no attempt.
- Fourth distinct valid attempt: return the existing three receipts and reject the new payload.
- Duplicate request: return the existing receipt without rescoring or consuming an attempt.
- Persistence conflict: retry from a fresh pinned snapshot; never merge stale projections.
- Repository failure: fail closed and return no score or accepted receipt.
- Attempts two and three: never return their score through UI state, API output, exception text, logs, or public projections.
- Public rendering uses explicit field allowlists and output-level leakage tests.
- Request size, line count, parse time, and scoring concurrency are bounded before private scoring. Operational throttling is defense in depth and never substitutes for the durable quota ledger.
- Private verified emails, OAuth subjects, predictions, and detailed metrics remain in the access-restricted immutable audit repository for the challenge history lifetime. The private repository is never made public, and access is limited to designated organizers and the submission service. This design makes no false promise that embedded personal fields can later be removed without changing Git history and provenance hashes. A future legal or policy deletion requirement requires a separately approved encrypted-envelope or history-purge design.

## 15. Testing Strategy

Implementation follows test-driven development.

### Policy and identity tests

- OAuth is optional for validation and required for test.
- OAuth `sub` controls quota even if verified email changes.
- Different OAuth subjects receive independent quotas.
- Missing email scope fails closed for test.

### Attempt tests

- Valid attempts one through three are accepted.
- A fourth distinct valid attempt is rejected.
- Invalid submissions consume no attempt.
- Exact retries are idempotent.
- Concurrent requests cannot exceed three accepted attempts.
- Same-second and same-name submissions cannot collide.

### Feedback and retrieval tests

- Attempt one returns only answer accuracy and evidence F1.
- Attempts two and three return no metrics or best-attempt signal.
- The authenticated owner can retrieve the saved first score.
- Another OAuth identity and a typed email cannot retrieve it.
- Public Gradio API responses cannot expose withheld metrics.

### Ranking tests

- Best-of-three selection follows the documented deterministic ordering.
- Multiple accounts with the same team remain distinct.
- The public final projection contains only allowlisted fields.
- No public test leaderboard exists before finalization.

### Compatibility tests

- Existing validation submission fixtures produce unchanged responses.
- Existing validation leaderboard rows render unchanged.
- Legacy validation records remain readable.
- Validation and test persistence paths cannot overlap.

### Release and privacy tests

- Public GitHub and Hugging Face test payloads contain tasks, PDFs, and checksums only.
- Validation and test labels are absent from current public trees and release history.
- Task/PDF/label ID sets and digests reconcile exactly.
- Public source, UI, API schemas, logs, and projections contain no private test labels or per-example test scores.

### Deployment tests

- Disabled test mode leaves live validation behavior unchanged.
- OAuth login/logout and verified-email scope work in a staging Space.
- CAS conflict retries neither lose nor duplicate attempts.
- Private organizer Space sees all test attempts; public Space sees none of the withheld fields.
- Closing and finalizing the release is idempotent.
- Four simultaneous valid submissions for one OAuth subject result in exactly three immutable accepted attempts and one compare-and-swap rejection with no score returned.
- Direct unauthenticated or crafted test API calls cannot select test gold, create records, retrieve attempt-one scores, or obtain withheld metrics.

## 16. Rollout and Rollback

### Phase 1: Safe preparation

- Implement split-aware components and private ledger behind disabled flags.
- Add OAuth while keeping validation anonymous-compatible.
- Deploy to a staging Space/private staging repository.
- Verify all compatibility and privacy gates.

### Phase 2: Data arrival

- Select the official incoming test source explicitly.
- Generate public/private payloads and release manifest.
- Complete leakage, PDF, schema, and hash audits.

### Phase 3: Publication

- Publish public test inputs to GitHub and Hugging Face.
- Publish private test labels and release policy privately.
- Deploy the organizer Space and verify its access boundary.
- Keep participant test submission disabled until all repositories reconcile.

### Phase 4: Activation

- Enable the official test window.
- Verify one controlled first attempt and one idempotent retry.
- Monitor private ledger integrity without exposing scores.

### Phase 5: Finalization

- Close test submissions at the configured time.
- Recompute and verify all private attempts.
- Atomically create the sanitized final projection.
- Publish the final test leaderboard and preserve the validation leaderboard as a separate view.

Rollback before activation disables test submissions and leaves validation untouched. After attempts exist, rollback disables new test submissions but never deletes accepted immutable attempts. Repairs create new audited projections from the ledger rather than editing attempt records.

## 17. Acceptance Criteria

The feature is ready to activate only when:

1. Existing validation submission and leaderboard behavior passes unchanged compatibility tests.
2. Public validation labels remain unexposed.
3. Public test artifacts contain only tasks, PDFs, checksums, and documentation.
4. Test labels exist only in the private repository.
5. Test OAuth identity and verified-email capture work without storing OAuth tokens.
6. Three valid unique test attempts are enforced per OAuth subject under concurrency.
7. Attempt one score is privately retrievable after login.
8. Attempts two and three disclose no score or best-attempt signal.
9. No public test ranking exists before finalization.
10. Organizer Space access is private and shows a complete reconstructible audit view.
11. Best-of-three and final ranking are deterministic and independently recomputed.
12. GitHub, public Hugging Face, private Hugging Face, public Space, and organizer Space revisions are recorded and verified.
