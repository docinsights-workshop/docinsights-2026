# DocSem Test Data Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fail-closed pipeline that accepts one explicitly selected official test source and publishes only tasks, PDFs, checksums, and documentation to GitHub and public Hugging Face while keeping labels private.

**Architecture:** A new preparation module validates the selected source, creates separate permission-restricted public/private staging payloads, and emits a signed-by-hash release manifest. Publication is a later explicit operation gated on reconciliation; no existing local test directory or ZIP is selected implicitly.

**Tech Stack:** Python 3.12, JSON/JSONL, SHA-256, PDF inspection, Git/Git LFS, `huggingface_hub==0.29.3`, `unittest`.

**Spec:** `docs/superpowers/specs/2026-09-04-docsem-test-release-and-dual-split-scoring-design.md`

## Global Constraints

- Test source path must be explicitly supplied.
- Public output contains no labels, answers, evidence, source mappings, organizer notes, or ZIPs.
- Private labels never enter the public Git index, Hugging Face dataset tree, logs, or command output.
- Existing train/validation files are not rewritten by test preparation.
- Test activation remains disabled until GitHub, public Hugging Face, private Hugging Face, and release-manifest hashes reconcile.
- No publication occurs from fixtures or from `test`, `test_hard`, `test_hard_1`, or ZIPs by name alone.

---

### Task 1: Source validator and release manifest

**Files:**
- Create: `scripts/prepare_docsem_test_release.py`
- Create: `scripts/test_prepare_docsem_test_release.py`

**Interfaces:**
- Produces: `validate_source(source_root, train_ids, val_ids) -> ValidatedTestSource` and `build_release_manifest(validated, release_id) -> dict`.

- [ ] **Step 1: Write failing fixture-based validation tests**

Create temporary fixtures inside tests with three tasks/PDFs/labels. Test exact task schema, exact label schema, unique IDs, task/PDF bijection, label/task equality, non-empty evidence, readable PDFs, visible block IDs, and disjoint train/validation IDs.

- [ ] **Step 2: Add negative leakage fixtures**

Test duplicate IDs, missing/extra PDF, missing/extra label, overlapping split ID, public answer/evidence field, empty evidence, unreadable PDF, PDF without a visible block ID, and source supplied as a ZIP.

- [ ] **Step 3: Run tests and verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 /Users/aamita/miniconda3/bin/python scripts/test_prepare_docsem_test_release.py`

- [ ] **Step 4: Implement validation and deterministic manifest**

The manifest contains release ID, counts, sorted IDs digest, tasks SHA-256, aggregate PDF inventory digest, private labels SHA-256, and schema version. It never contains labels or per-instance answers.

- [ ] **Step 5: Run tests and verify GREEN**

Run: `PYTHONDONTWRITEBYTECODE=1 /Users/aamita/miniconda3/bin/python scripts/test_prepare_docsem_test_release.py`

Expected: all source-validation tests pass without printing private rows.

- [ ] **Step 6: Commit**

```bash
git add scripts/prepare_docsem_test_release.py scripts/test_prepare_docsem_test_release.py
git commit -m "Validate DocSem held-out test releases"
```

### Task 1A: Canonical image-backed PDF compatibility

**Why this task exists:** Final Task 1 review established that the canonical
DocSem PDFs are raster/image-backed and expose no PDF text traces. The original
vector-text proof therefore cannot validate the actual release format.

**Files:**
- Modify: `scripts/prepare_docsem_test_release.py`
- Modify: `scripts/test_prepare_docsem_test_release.py`
- Modify: `docs/superpowers/plans/2026-09-04-docsem-test-data-release.md`

**Interfaces:**
- Produces: a renderer-backed OCR visibility audit that operates on both
  image-backed and vector PDFs without trusting PDF extraction text.

- [x] **Step 1: Add failing canonical-format tests**

Add an image-only PDF positive fixture whose evidence header is visually
readable after rendering. Add negatives for a clipped two-pixel fragment,
hidden/extraction-only text, missing OCR backend, oversized files/pages/raster
allocations, OCR timeout, and malformed OCR output.

- [x] **Step 2: Replace private renderer bindings with bounded raster OCR**

Use only public PyMuPDF rendering APIs and an explicit Tesseract CLI contract.
Require line-anchored `bNN:` evidence headers in OCR output. Bound input bytes,
page count, page dimensions, rendered pixels, OCR output, and per-page runtime.
Fail closed if the exact tested OCR runtime contract is unavailable. Never log
OCR text or private label rows.

The exact tested release-host contract is PyMuPDF 1.26.3 and Tesseract 5.5.1
with the English language data. The validator checks both versions at runtime
before inspecting any release PDF.

- [x] **Step 3: Record the visibility-audit contract in the release manifest**

Record only the method/version and bounded render settings. Do not include OCR
text, evidence IDs, private paths, or labels.

- [x] **Step 4: Run focused and canonical-format compatibility checks**

Run the full source-validator suite and a sanitized compatibility audit against
the current public validation PDF format. Report only counts and pass/fail.

- [x] **Step 5: Commit**

```bash
git add scripts/prepare_docsem_test_release.py scripts/test_prepare_docsem_test_release.py docs/superpowers/plans/2026-09-04-docsem-test-data-release.md
git commit -m "Validate image-backed DocSem evidence safely"
```

### Task 2: Permission-separated public/private staging

**Files:**
- Modify: `scripts/prepare_docsem_test_release.py`
- Modify: `scripts/test_prepare_docsem_test_release.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `stage_release(validated, public_root, private_root, release_id) -> manifest`.

- [x] **Step 1: Write failing staging tests**

Assert public staging contains exactly `test/tasks.jsonl`, `test/documents/*.pdf`, `test/SHA256SUMS`, and a public manifest without private fields. Assert private staging contains `private/test_labels.jsonl` and `private/test_release.json` with mode `0600`; private directories use mode `0700`.

- [x] **Step 2: Implement deterministic staging**

Normalize public `document_pdf` to `test/documents/<filename>`, copy PDFs byte-for-byte, emit sorted checksums, and write private files without printing rows or answers.

- [x] **Step 3: Add a positive-control public leakage scanner**

Implement `audit_public_payload(path)` and prove it rejects fixtures containing label filenames, `answer`/`evidence` fields, private path names, source mappings, or ZIP archives.

- [x] **Step 4: Verify staging does not alter existing public splits**

Hash current train/validation manifests and PDF inventories before and after staging; assert equality.

- [x] **Step 5: Run staging and leakage tests**

Run: `PYTHONDONTWRITEBYTECODE=1 /Users/aamita/miniconda3/bin/python scripts/test_prepare_docsem_test_release.py`

Expected: public/private separation, file-mode, hash, and positive-control leakage tests pass.

- [x] **Step 6: Commit**

```bash
git add .gitignore scripts/prepare_docsem_test_release.py scripts/test_prepare_docsem_test_release.py
git commit -m "Stage DocSem test data with hard privacy boundaries"
```

### Task 3: Public dataset configuration and documentation

**Files:**
- Modify: `competition/hf-dataset/README.md`
- Modify: `competition/README.md`
- Modify: `scripts/prepare_docsem_hf_dataset.py`
- Modify: `scripts/test_prepare_docsem_test_release.py`

**Interfaces:**
- Consumes: staged public test payload and release manifest.
- Produces: Hugging Face `tasks` config with train, validation, and test; `labels` config remains train-only.

- [ ] **Step 1: Write a failing metadata test**

Parse the dataset card front matter and assert `tasks` has `test/tasks.jsonl`, while `labels` contains only `train/labels.jsonl` and contains no validation/test label path.

- [ ] **Step 2: Implement metadata and participant instructions**

Document the held-out test schema, checksums, OAuth submission portal, three attempts per HF account, first-attempt feedback, later score withholding, and finalization behavior. Do not include test IDs or private repository paths beyond the generic label contract.

- [ ] **Step 3: Update the generator safely**

The existing no-argument generation path remains train/validation-only. Test inclusion requires explicit `--test-public-staging` and validates the manifest before copying only public files.

- [ ] **Step 4: Run metadata, generator, and scoring tests**

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/aamita/miniconda3/bin/python scripts/test_prepare_docsem_test_release.py
PYTHONDONTWRITEBYTECODE=1 /Users/aamita/miniconda3/bin/python scripts/test_competition_scoring.py
```

Expected: metadata exposes test tasks but no validation/test labels; existing scoring checks remain green.

- [ ] **Step 5: Commit**

```bash
git add competition/hf-dataset/README.md competition/README.md scripts/prepare_docsem_hf_dataset.py scripts/test_prepare_docsem_test_release.py
git commit -m "Document DocSem held-out test release"
```

### Task 4: Publication and reconciliation tooling

**Files:**
- Create: `scripts/publish_docsem_test_release.py`
- Create: `scripts/test_publish_docsem_test_release.py`
- Modify: `competition/README.md`

**Interfaces:**
- Produces: dry-run publication plan and explicit `--publish` workflow for canonical GitHub, public HF, and private HF targets.

- [ ] **Step 1: Write failing dry-run tests**

Assert the plan names exact repositories, base revisions, allowed public paths, private paths, counts, and hashes. Assert publication refuses a dirty/behind/diverged source checkout, missing manifest, nonmatching hashes, public labels, or unspecified source branch.

- [ ] **Step 2: Implement dry-run-only default**

No remote write occurs without `--publish`, an exact expected source revision, and a permission-restricted private staging root. Print counts and hashes only, never private rows or answers.

- [ ] **Step 3: Implement non-force publication gates**

Publish the canonical GitHub test directory and public HF test directory with explicit paths. Publish only private labels/policy to the private repository using exact-parent CAS. Abort on any remote movement and regenerate the plan.

- [ ] **Step 4: Implement post-publication reconciliation**

Download public tasks/checksums, compare all public PDF SHA-256 values, verify private label/release digests, and scan current public trees and reachable history for forbidden paths/fields. Emit a sanitized revision receipt.

- [ ] **Step 5: Run publication dry-run tests**

Run: `PYTHONDONTWRITEBYTECODE=1 /Users/aamita/miniconda3/bin/python scripts/test_publish_docsem_test_release.py`

Expected: remote-movement, dirty-tree, allowed-path, hash, and privacy gates pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/publish_docsem_test_release.py scripts/test_publish_docsem_test_release.py competition/README.md
git commit -m "Add guarded DocSem test release publication"
```

### Task 5: Data-arrival execution gate

**Files:**
- No production file changes unless the official source is present.
- Generated public/private staging remains ignored.

**Interfaces:**
- Consumes: organizer-selected official test source path and release ID.
- Produces: verified public/private staged payloads and a sanitized release receipt.

- [ ] **Step 1: Recheck authoritative source and branch**

Fetch without switching or cleaning. Require a clean named source and explicit organizer selection; do not infer from directory names.

- [ ] **Step 2: Run preparation and private/public audits**

```bash
umask 077
/Users/aamita/miniconda3/bin/python scripts/prepare_docsem_test_release.py \
  --source "$DOCSEM_OFFICIAL_TEST_SOURCE" \
  --release-id "$DOCSEM_TEST_RELEASE_ID" \
  --public-output competition/hf-test-staging/public \
  --private-output competition/hf-test-staging/private
```

Expected: a sanitized count/hash receipt with no label rows or answers printed.

Record only counts, schemas, hashes, and pass/fail results in visible output.

- [ ] **Step 3: Stop safely if data is absent or ambiguous**

Leave `TEST_SUBMISSIONS_ENABLED=false`, do not publish any candidate, and report the exact missing authority/data selection.

- [ ] **Step 4: If data is authoritative, execute guarded publication and verify all revisions**

Run the publication tool first without `--publish`; require every gate to pass, then rerun with `--publish`, the exact expected GitHub/HF base revisions printed by the dry run, and the same staged release digest.

- [ ] **Step 5: Record the release receipt in a tracked, label-free Markdown document and commit it**
