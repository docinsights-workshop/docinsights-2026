# DocSem Dual-Split Participant Space Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing public DocSem Space with backward-compatible validation submission plus OAuth-protected, three-attempt test submission and first-score retrieval.

**Architecture:** Keep the current validation path intact and route test requests into a new immutable JSON/CAS ledger in the existing private Hugging Face repository. Enable Hugging Face OAuth globally but require it only on server-side test handlers; public test scores remain hidden until finalization.

**Tech Stack:** Python 3.12, Gradio 4.42.0, `huggingface_hub==0.29.3`, JSON/JSONL, `unittest`, Hugging Face Spaces OAuth.

**Spec:** `docs/superpowers/specs/2026-09-04-docsem-test-release-and-dual-split-scoring-design.md`

## Global Constraints

- Existing anonymous validation submission and the current validation leaderboard must remain behaviorally unchanged.
- Test submission requires server-injected Hugging Face OAuth `sub`, username, and verified email.
- Test quota is exactly three valid unique attempts per OAuth subject, not per team.
- Attempt one returns answer accuracy and evidence F1; attempts two and three return no metrics or best-attempt signal.
- Public API/UI/log/error output must not expose test labels, per-example results, later-attempt scores, private identities, or live test ranks.
- Test persistence uses one exact-parent compare-and-swap commit; no blind `upload_file` write is allowed on test paths.
- `TEST_SUBMISSIONS_ENABLED` defaults to `false` and activation fails closed without a complete pinned release configuration.
- OAuth tokens and cookies are never persisted.

---

### Task 1: Pure test policy and identity model

**Files:**
- Create: `competition/hf-space/test_policy.py`
- Create: `competition/hf-space/test_test_policy.py`

**Interfaces:**
- Produces: `OAuthIdentity`, `TestReleasePolicy`, `canonical_submission_hash`, `account_key`, `select_best_attempt`, `participant_test_response`.
- Consumes: parsed prediction rows and aggregate metrics from `scoring.py`.

- [ ] **Step 1: Write failing identity and policy tests**

```python
class TestPolicyTests(unittest.TestCase):
    def test_account_key_uses_stable_subject_not_email(self):
        first = OAuthIdentity(sub="stable-1", username="u", email="a@example.org")
        changed = OAuthIdentity(sub="stable-1", username="u2", email="b@example.org")
        self.assertEqual(account_key(first), account_key(changed))

    def test_missing_verified_email_is_rejected(self):
        with self.assertRaisesRegex(TestPolicyError, "verified email"):
            OAuthIdentity.from_profile({"sub": "s", "preferred_username": "u"})

    def test_disabled_or_closed_policy_rejects_before_scoring(self):
        policy = TestReleasePolicy.disabled()
        with self.assertRaisesRegex(TestPolicyError, "not open"):
            policy.require_open(now=dt.datetime(2026, 9, 5, tzinfo=dt.timezone.utc))
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 /Users/aamita/miniconda3/bin/python competition/hf-space/test_test_policy.py`

Expected: FAIL because `test_policy.py` and its interfaces do not exist.

- [ ] **Step 3: Implement the minimal pure model**

```python
@dataclass(frozen=True)
class OAuthIdentity:
    sub: str
    username: str
    email: str

    @classmethod
    def from_profile(cls, profile):
        data = dict(profile or {})
        sub = str(data.get("sub") or "").strip()
        username = str(data.get("preferred_username") or "").strip()
        email = str(data.get("email") or "").strip().casefold()
        if not sub or not username or not email:
            raise TestPolicyError("Test submission requires a verified email and HF identity.")
        return cls(sub=sub, username=username, email=email)


def account_key(identity):
    return hashlib.sha256(identity.sub.encode("utf-8")).hexdigest()
```

Define `TestReleasePolicy` with `release_id`, task/gold digests, UTC open/close datetimes, `enabled`, and `max_attempts=3`. Reject missing configuration and non-UTC or closed windows.

- [ ] **Step 4: Add deterministic hashing, best-selection, and feedback tests**

```python
def test_attempt_one_feedback_has_only_public_aggregates(self):
    response = participant_test_response(1, METRICS, "receipt-1")
    self.assertEqual(set(response), {"accepted", "attempt", "receipt", "answer_accuracy", "evidence_f1"})

def test_attempt_two_feedback_withholds_every_metric(self):
    response = participant_test_response(2, METRICS, "receipt-2")
    self.assertEqual(response, {"accepted": True, "attempt": 2, "receipt": "receipt-2", "score": "withheld"})

def test_best_attempt_uses_accuracy_f1_time_and_id(self):
    self.assertEqual(select_best_attempt(FIXTURE_ATTEMPTS)["submission_id"], "expected-id")
```

- [ ] **Step 5: Implement and verify GREEN**

Run the focused test file and confirm all policy, identity, hashing, feedback, and tie-order tests pass.

- [ ] **Step 6: Commit**

```bash
git add competition/hf-space/test_policy.py competition/hf-space/test_test_policy.py
git commit -m "Add DocSem test submission policy model"
```

### Task 2: Atomic private test-attempt store

**Files:**
- Create: `competition/hf-space/test_store.py`
- Create: `competition/hf-space/test_test_store.py`

**Interfaces:**
- Consumes: `OAuthIdentity`, `TestReleasePolicy`, `account_key`, `canonical_submission_hash`, `select_best_attempt` from `test_policy.py`.
- Produces: `HubTestStore.submit(identity, metadata, predictions, metrics, now) -> TestReceipt` and `HubTestStore.account_history(identity) -> list[dict]`.

- [ ] **Step 1: Write failing store tests using an in-memory exact-parent Hub fake**

The fake must model repository SHA changes and reject a stale `parent_commit`; it must not merely assert a mock call.

```python
def test_three_concurrent_attempts_commit_and_fourth_is_rejected(self):
    store = HubTestStore(InMemoryHub(), repo_id="private/repo")
    receipts = run_four_concurrent_submissions(store, IDENTITY, POLICY)
    self.assertEqual(sorted(r.attempt for r in receipts if r.accepted), [1, 2, 3])
    self.assertEqual(sum(not r.accepted for r in receipts), 1)
    self.assertEqual(len(store.account_history(IDENTITY)), 3)

def test_exact_retry_returns_existing_receipt(self):
    first = store.submit(IDENTITY, META, PREDICTIONS, METRICS, NOW)
    replay = store.submit(IDENTITY, META, PREDICTIONS, METRICS, NOW)
    self.assertEqual(first.submission_id, replay.submission_id)
    self.assertEqual(len(store.account_history(IDENTITY)), 1)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 /Users/aamita/miniconda3/bin/python competition/hf-space/test_test_store.py`

Expected: FAIL because `HubTestStore` does not exist.

- [ ] **Step 3: Implement snapshot reads and immutable record planning**

Use paths:

```python
attempt_path = f"attempts/test/{account_key(identity)}/{submission_id}.json"
account_path = f"projections/test/accounts/{account_key(identity)}.json"
organizer_path = "projections/test/organizer_leaderboard.json"
```

Read `private/test_release.json`, `private/test_labels.jsonl`, account attempts, and organizer projection at one SHA. Validate release/task/gold digests before scoring is accepted.

- [ ] **Step 4: Implement one-commit compare-and-swap persistence**

```python
api.create_commit(
    repo_id=repo_id,
    repo_type="dataset",
    revision="main",
    parent_commit=base_sha,
    operations=[attempt_add, account_projection_add, organizer_projection_add],
    commit_message=f"Accept DocSem test attempt {attempt_number}",
)
```

On parent mismatch, reload and rederive. Bound retries at five. Never retry using a stale attempt number or projection.

- [ ] **Step 5: Test failure and privacy branches**

Cover disabled/closed release, missing gold, digest mismatch, invalid account, fourth attempt, repo outage, exhausted conflicts, duplicate hash, and generic value-free exceptions. Assert commit messages and raised errors contain no email, OAuth subject, answer, score, or prediction content.

- [ ] **Step 6: Run focused and existing scoring tests**

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/aamita/miniconda3/bin/python competition/hf-space/test_test_store.py
PYTHONDONTWRITEBYTECODE=1 /Users/aamita/miniconda3/bin/python competition/hf-space/test_scoring.py
```

Expected: atomic-store and existing scoring tests pass.

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/aamita/miniconda3/bin/python competition/hf-space/test_test_store.py
PYTHONDONTWRITEBYTECODE=1 /Users/aamita/miniconda3/bin/python competition/hf-space/test_scoring.py
```

- [ ] **Step 7: Commit**

```bash
git add competition/hf-space/test_store.py competition/hf-space/test_test_store.py
git commit -m "Add atomic DocSem test attempt ledger"
```

### Task 3: Split-aware service boundary

**Files:**
- Create: `competition/hf-space/submission_service.py`
- Create: `competition/hf-space/test_submission_service.py`
- Modify: `competition/hf-space/app.py`

**Interfaces:**
- Consumes: existing validation functions from `app.py`/`scoring.py`, `HubTestStore`, and `OAuthIdentity`.
- Produces: `submit_for_split(split, file_obj, metadata, oauth_profile) -> dict` and `history_for_oauth(oauth_profile) -> list[dict]`.

- [ ] **Step 1: Characterize the legacy validation behavior before refactoring**

Add a fixture test that calls the existing validation path and asserts the same metrics, persistence payload keys, identity semantics, and public response shape.

- [ ] **Step 2: Run characterization test and verify it passes against current code**

This is a preservation test, not the RED step for new behavior.

- [ ] **Step 3: Write failing split-routing tests**

```python
def test_validation_does_not_require_oauth(self):
    result = service.submit_for_split("validation", FILE, VALIDATION_META, None)
    self.assertEqual(result["answer_accuracy"], 1.0)

def test_test_rejects_missing_oauth_before_reading_file(self):
    unreadable = FileProbe()
    with self.assertRaisesRegex(SubmissionError, "Sign in"):
        service.submit_for_split("test", unreadable, TEST_META, None)
    self.assertFalse(unreadable.was_read)

def test_client_cannot_supply_gold_path_or_attempt_identity(self):
    with self.assertRaises(SubmissionError):
        service.submit_for_split("../../private", FILE, TEST_META, PROFILE)
```

- [ ] **Step 4: Implement explicit server-side routing**

Use a closed enum/allowlist for `validation` and `test`. Validation calls the unchanged legacy path. Test constructs identity only from the injected OAuth profile and loads file/config paths only from server policy.

- [ ] **Step 5: Test feedback suppression at the service boundary**

Assert direct Python and Gradio API calls receive attempt-one aggregates only and receive no metrics for attempts two/three. Assert exception wrapping never interpolates private exception details.

- [ ] **Step 6: Run service, policy, store, and legacy suites**

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/aamita/miniconda3/bin/python competition/hf-space/test_submission_service.py
PYTHONDONTWRITEBYTECODE=1 /Users/aamita/miniconda3/bin/python competition/hf-space/test_test_policy.py
PYTHONDONTWRITEBYTECODE=1 /Users/aamita/miniconda3/bin/python competition/hf-space/test_test_store.py
PYTHONDONTWRITEBYTECODE=1 /Users/aamita/miniconda3/bin/python competition/hf-space/test_scoring.py
```

Expected: split routing, OAuth trust boundary, test ledger, and legacy validation suites pass.

- [ ] **Step 7: Commit**

```bash
git add competition/hf-space/submission_service.py competition/hf-space/test_submission_service.py competition/hf-space/app.py
git commit -m "Route DocSem validation and test submissions safely"
```

### Task 4: OAuth-enabled participant UI and score retrieval

**Files:**
- Modify: `competition/hf-space/README.md`
- Modify: `competition/hf-space/requirements.txt`
- Modify: `competition/hf-space/app.py`
- Modify: `competition/hf-space/test_portal_layout.py`
- Create: `competition/hf-space/test_portal_behavior.py`

**Interfaces:**
- Consumes: `submit_for_split`, `history_for_oauth`.
- Produces: one public Space UI containing split selector, optional OAuth login, adaptive instructions, submission response, `My test submissions`, validation leaderboard, and post-finalization test leaderboard placeholder.

- [ ] **Step 1: Write failing rendered-config tests**

Assert the Gradio config contains:

```text
Validation (development)
Test (final)
Sign in with Hugging Face
My test submissions
```

Also assert validation remains the selector default and the public initial config contains no test score, rank, email, OAuth subject, label path, or private repository file listing.

- [ ] **Step 2: Add OAuth Space metadata**

```yaml
hf_oauth: true
hf_oauth_scopes:
  - email
```

Pin the deployment runtime to the version already exercised by the live Space and add the OAuth extra:

```text
gradio[oauth]==4.42.0
huggingface_hub==0.29.3
```

- [ ] **Step 3: Implement adaptive UI**

Add `gr.LoginButton`, `gr.LogoutButton`, a validation-default dropdown, and split-specific instructions. Inject `gr.OAuthProfile | None` into test submission/history handlers. Do not accept identity fields from component inputs.

- [ ] **Step 4: Implement authenticated history view**

Render only the current account's attempt receipts. Attempt one shows answer accuracy/evidence F1; attempts two/three show `Score withheld until finalization`. Mask the verified email. Do not add an email lookup input.

- [ ] **Step 5: Test direct endpoint behavior**

Exercise the generated Gradio API functions directly: unauthenticated test submit/history fail closed, authenticated subject A cannot access subject B, and later-attempt metrics never appear in serialized component updates.

- [ ] **Step 6: Run all Space tests**

```bash
cd competition/hf-space
PYTHONDONTWRITEBYTECODE=1 /Users/aamita/miniconda3/bin/python -m unittest discover -p 'test_*.py'
```

- [ ] **Step 7: Commit**

```bash
git add competition/hf-space/README.md competition/hf-space/requirements.txt competition/hf-space/app.py competition/hf-space/test_portal_layout.py competition/hf-space/test_portal_behavior.py
git commit -m "Add OAuth test workflow to DocSem participant Space"
```

### Task 5: Feature flags, compatibility audit, and disabled deployment package

**Files:**
- Modify: `competition/hf-space/README.md`
- Modify: `competition/README.md`
- Modify: `scripts/test_competition_scoring.py`
- Modify: `scripts/audit_site.py`

**Interfaces:**
- Produces: fail-closed configuration documentation and a deployment artifact safe to publish with test submission disabled.

- [ ] **Step 1: Add failing configuration tests**

Test missing release ID, task digest, gold digest, open time, or close time with `TEST_SUBMISSIONS_ENABLED=true`; each must disable test submission and leave validation operational.

- [ ] **Step 2: Implement configuration parsing with safe defaults**

`TEST_SUBMISSIONS_ENABLED` and `TEST_PUBLIC_LEADERBOARD_ENABLED` default false. `TEST_MAX_ATTEMPTS` is fixed at three. Reject non-RFC3339 UTC windows and `open >= close`.

- [ ] **Step 3: Add full backward-compatibility fixtures**

Run stored validation fixtures through the new app/service and compare their exact public metrics and rendered leaderboard rows with the pre-change behavior.

- [ ] **Step 4: Add public leakage scans**

Scan Space source, rendered config, API schema, participant responses, and logs produced by failure fixtures for test gold paths, answer/evidence values, later scores, emails, OAuth subjects, and private per-example metrics. Include positive fixtures proving the scanner fails on each forbidden class.

- [ ] **Step 5: Run the complete disabled-mode verification**

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/aamita/miniconda3/bin/python -m unittest discover -s competition/hf-space -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 /Users/aamita/miniconda3/bin/python scripts/test_competition_scoring.py
```

Expected: all new test controls are present but disabled; validation behavior and its public leaderboard remain unchanged.

- [ ] **Step 6: Commit**

```bash
git add competition/hf-space competition/README.md scripts/test_competition_scoring.py scripts/audit_site.py
git commit -m "Gate DocSem test workflow behind disabled release config"
```
