# DocSem Organizer Space and Finalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide a private organizer-only view of all held-out test attempts and a deterministic, audited best-of-three finalization workflow.

**Architecture:** A separate private Gradio Space builds read-only views from immutable test attempt JSON in the private Hugging Face dataset. A CLI finalizer independently recomputes every score from pinned release/evaluator revisions and atomically writes one sanitized public-final projection; the organizer Space never mutates competition state.

**Tech Stack:** Python 3.12, Gradio, `huggingface_hub==0.29.3`, JSON/JSONL, optional disposable SQLite/read index, `unittest`.

**Spec:** `docs/superpowers/specs/2026-09-04-docsem-test-release-and-dual-split-scoring-design.md`

## Global Constraints

- The organizer Space must be platform-private and use a read-only private-repository token.
- A hidden tab or client-side flag in the public Space is not acceptable.
- The immutable JSON attempt ledger is authoritative; SQLite is disposable only.
- No organizer view, logs, tests, or commit messages may print raw labels or OAuth tokens.
- Finalization is unavailable before the configured close time and is idempotent.
- Public final output uses an explicit field allowlist and remains absent until finalization succeeds.

---

### Task 1: Organizer projection reader

**Files:**
- Create: `competition/hf-organizer-space/organizer_data.py`
- Create: `competition/hf-organizer-space/test_organizer_data.py`

**Interfaces:**
- Produces: `load_snapshot(repo_id, revision, token) -> OrganizerSnapshot`, `verify_snapshot(snapshot) -> AuditReport`, and `organizer_rows(snapshot) -> list[dict]`.

- [ ] **Step 1: Write failing snapshot tests**

Use fixtures with multiple accounts/attempts. Assert complete reconstruction, digest checks, attempt numbering, best marker, excluded accounts, and adjudication records. Reject mutable/missing/duplicate attempt IDs and projection mismatches.

- [ ] **Step 2: Run tests and verify RED**

- [ ] **Step 3: Implement pinned read-only reconstruction**

Read one exact private repo SHA into a permission-restricted temporary cache. Derive rows from attempts rather than trusting projections; compare derived and stored projections and surface mismatch status.

- [ ] **Step 4: Add optional disposable index**

If row volume requires it, build an in-memory SQLite index from `OrganizerSnapshot`. Do not write or upload the database; deleting it must not lose information.

- [ ] **Step 5: Run reconstruction tests**

Run: `PYTHONDONTWRITEBYTECODE=1 /Users/aamita/miniconda3/bin/python competition/hf-organizer-space/test_organizer_data.py`

Expected: reconstruction, digest, projection, exclusion, and adjudication tests pass.

- [ ] **Step 6: Commit**

```bash
git add competition/hf-organizer-space/organizer_data.py competition/hf-organizer-space/test_organizer_data.py
git commit -m "Reconstruct DocSem organizer test views"
```

### Task 2: Private organizer Gradio Space

**Files:**
- Create: `competition/hf-organizer-space/README.md`
- Create: `competition/hf-organizer-space/app.py`
- Create: `competition/hf-organizer-space/requirements.txt`
- Create: `competition/hf-organizer-space/test_app.py`

**Interfaces:**
- Consumes: `OrganizerSnapshot`, `AuditReport`, `organizer_rows`.
- Produces: private read-only filters, detailed attempts table, per-example view, selected-best markers, and integrity summary.

- [ ] **Step 1: Write failing rendered-config and access tests**

Assert the app contains no submission/finalization controls, never embeds secrets/labels in config, and refuses to start without `ORGANIZER_READ_TOKEN` and `PRIVATE_REPO_ID`.

- [ ] **Step 2: Implement the private UI**

Show revision, release/digest status, account/verified email, team/participants, attempt metadata, aggregate and per-example metrics, selected best, exclusions, and adjudications. Fetch sensitive detail only in authenticated server callbacks; do not serialize all private data into initial public component config.

- [ ] **Step 3: Add filters and CSV export safely**

CSV export is organizer-only, generated server-side from the pinned snapshot, and includes an audit header with repository/release/evaluator revisions. It must never contain OAuth tokens or gold answers.

- [ ] **Step 4: Run organizer Space tests**

Run: `PYTHONDONTWRITEBYTECODE=1 /Users/aamita/miniconda3/bin/python -m unittest discover -s competition/hf-organizer-space -p 'test_*.py'`

Expected: app/config/access and organizer-data tests pass without network access.

- [ ] **Step 5: Commit**

```bash
git add competition/hf-organizer-space
git commit -m "Add private DocSem organizer leaderboard Space"
```

### Task 3: Deterministic finalizer

**Files:**
- Create: `scripts/finalize_docsem_test_leaderboard.py`
- Create: `scripts/test_finalize_docsem_test_leaderboard.py`

**Interfaces:**
- Consumes: pinned private release, labels, attempts, exclusions/adjudications, and scorer revision.
- Produces: `build_finalization(snapshot, now) -> FinalizationPlan` and optional CAS commit containing private audit manifest plus sanitized `projections/test/public_final.json`.

- [ ] **Step 1: Write failing best-of-three and cutoff tests**

Test answer accuracy, evidence F1, accepted timestamp, and submission-ID ordering; exclude post-cutoff, wrong-release, wrong-gold, excluded, malformed, duplicate, and uncommitted attempts.

- [ ] **Step 2: Write failing public allowlist tests**

Assert each public row has exactly:

```text
rank, hf_username, team, submission_name, selected_attempt, answer_accuracy, evidence_f1
```

Inject email, subject, participant names, predictions, per-example metrics, unselected scores, and private paths into positive-control fixtures and verify the audit rejects them.

- [ ] **Step 3: Implement dry-run plan and independent rescoring**

Recompute all accepted attempts using pinned labels and scorer. Compare stored metrics, record mismatches privately, and refuse finalization until resolved. Output only counts, hashes, and revisions.

- [ ] **Step 4: Implement idempotent CAS finalization**

Require `--yes`, `--maintenance-confirmed`, exact expected private SHA, and a closed window. Write the final projection, private audit manifest, and `finalized=true` policy in one exact-parent commit. Re-running against the finalized revision must reproduce the same hashes and perform no write.

- [ ] **Step 5: Run finalization tests**

Run: `PYTHONDONTWRITEBYTECODE=1 /Users/aamita/miniconda3/bin/python scripts/test_finalize_docsem_test_leaderboard.py`

Expected: cutoff, independent rescoring, allowlist, idempotency, and CAS tests pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/finalize_docsem_test_leaderboard.py scripts/test_finalize_docsem_test_leaderboard.py
git commit -m "Finalize DocSem best-of-three test leaderboard"
```

### Task 4: Public final leaderboard integration

**Files:**
- Modify: `competition/hf-space/app.py`
- Modify: `competition/hf-space/test_portal_behavior.py`
- Modify: `competition/hf-space/README.md`
- Modify: `shared-task.md`
- Modify: `faq.md`
- Modify: `scripts/audit_site.py`

**Interfaces:**
- Consumes: sanitized `projections/test/public_final.json` only when `finalized=true` and `TEST_PUBLIC_LEADERBOARD_ENABLED=true`.
- Produces: separate validation and final-test leaderboard views in the existing public Space.

- [ ] **Step 1: Write failing pre/post-finalization visibility tests**

Before finalization, assert there is no public test score/rank endpoint or table. After finalization fixture activation, assert the test table renders only allowlisted fields while validation remains unchanged.

- [ ] **Step 2: Implement final-only public rendering**

Use two leaderboard selectors/sections: `Validation leaderboard` and `Final test leaderboard`. The test view reads only the sanitized projection and refuses unfinalized/mismatched release data.

- [ ] **Step 3: Update public documentation**

Document first-attempt feedback, later withholding, per-account quota, accepted multi-account limitation, final best-of-three policy, and separate validation/final-test views.

- [ ] **Step 4: Run Space and rendered-site audits**

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/aamita/miniconda3/bin/python -m unittest discover -s competition/hf-space -p 'test_*.py'
mkdir -p /private/tmp/docsem-site-audit
env JEKYLL_NO_BUNDLER_REQUIRE=true JEKYLL_NO_DISK_CACHE=true arch -x86_64 /usr/local/bin/jekyll build --destination /private/tmp/docsem-site-audit/docinsights-2026
/Users/aamita/miniconda3/bin/python scripts/audit_site.py /private/tmp/docsem-site-audit
```

Expected: all Space tests and the rendered public-site audit pass.

- [ ] **Step 5: Commit**

```bash
git add competition/hf-space shared-task.md faq.md scripts/audit_site.py
git commit -m "Publish final DocSem test leaderboard view"
```

### Task 5: Private deployment and operational verification

**Files:**
- Create: `scripts/publish_docsem_organizer_space.py`
- Create: `scripts/test_publish_docsem_organizer_space.py`
- Modify: `competition/README.md`

**Interfaces:**
- Produces: dry-run/private-Space creation and update workflow for `amitbcp/docsem-docinsights-organizer`.

- [ ] **Step 1: Write failing deployment-plan tests**

Assert the Space must be private, uses a read-only token, has an explicit collaborator allowlist, and rejects public visibility or a write-scoped participant token.

- [ ] **Step 2: Implement dry-run-only publication default**

Inspect current Space visibility/config without printing secrets. Require `--publish`, exact expected source revision, and explicit private visibility for any write.

- [ ] **Step 3: Deploy and verify access boundary**

Create/update the private Space, set only required secrets, verify authorized organizer access, verify unauthenticated/non-collaborator denial, and confirm no submission/finalization API exists.

- [ ] **Step 4: Reconcile organizer and participant views**

At the same private repository revision, confirm organizer reconstruction matches every immutable attempt while the public participant Space exposes only attempt-one owner feedback and no live test leaderboard.

- [ ] **Step 5: Commit deployment receipt without secrets or private rows**
