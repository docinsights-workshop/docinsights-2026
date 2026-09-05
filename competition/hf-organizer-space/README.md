---
title: "DocSem Organizer Test Dashboard"
sdk: gradio
sdk_version: 4.42.0
app_file: app.py
license: apache-2.0
---

# DocSem organizer test dashboard

This is the source for the organizer-only DocSem held-out test dashboard. It
must be deployed only to a **platform-private Hugging Face Space** with an
explicit organizer collaborator allowlist. It is not a hidden route or tab in
the participant Space.

The Space is read-only. It resolves the private test-ledger dataset HEAD only
after an organizer requests a refresh, pins all reconstruction to that exact
commit SHA, and refuses data whose release, digest, immutable-attempt, account
projection, organizer projection, exclusion, or adjudication checks fail.
The directory is a self-contained Space bundle: `app.py`, `organizer_data.py`,
and `organizer_contract.py` are deployed together; no sibling participant-Space
source is imported at runtime.

Required Space secrets:

- `ORGANIZER_READ_TOKEN`: read-only access to the private ledger dataset.
- `PRIVATE_REPO_ID`: the `owner/name` of that private dataset.

Do not configure a write-scoped participant token. This application has no
submission, finalization, repository-commit, or upload control. Initial Gradio
configuration is empty; private attempts and per-example metrics are loaded
only by authenticated server callbacks inside the private Space.

CSV exports are produced server-side from the pinned verified snapshot in a
mode-0700 temporary directory with mode-0600 files. They include revision and
digest audit metadata plus allowlisted attempt fields, but never OAuth tokens,
gold/correct answers, or raw predictions.
