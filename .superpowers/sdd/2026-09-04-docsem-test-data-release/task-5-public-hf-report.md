# Task 5P implementation receipt

Status: DONE_WITH_CONCERNS

Implemented a dedicated `scripts/publish_docsem_public_hf_test_release.py`
tool and focused behavioral suite. The tool is public-only and dry-run by
default: it has no private backend, repository, token option, label reader, or
publication invocation in this task.

## TDD record

RED was recorded with `python3 scripts/test_publish_docsem_public_hf_test_release.py`:
eight focused behavioral tests failed because the dedicated publisher module
was absent. The failure was the expected missing-feature failure.

GREEN was recorded with the same command after the minimal implementation:
`Ran 9 tests ... OK`. The focused tests cover default no-write dry runs,
confirmation/base refusal, source identity and unsafe-source refusal,
normalised exact staging with same-device PDF hardlinks, public-only ordering,
documentation-last CAS, remote mismatch refusal, resumable partial state,
idempotency, sanitized output, and public-history/visibility refusal.

## Verification

- Focused suite: 9 passed.
- Existing publisher suite: 19 passed.
- Python compilation of the new and existing release modules: passed.
- `git diff --check`: passed.
- No network write, deployment, private repository operation, credential
  access, private-label read, or source publication was performed.

## Implementation notes

The publisher requires the approved clean source identity and manifest digest,
constructs only the audited `test/**` allowlist, invokes
`audit_public_payload`, and uses hardlinks rather than PDF copies. It uses the
Hub resumable large-folder API for `test/**`, verifies the exact test
inventory/metadata/PDF size-LFS SHA data, scans public current/history paths
while allowing only canonical train labels, and updates `README.md` plus
`INSTRUCTIONS.md` together only after the test namespace reconciles. Release
text records the release/upstream identity, byte-identical collision remaps,
closed submissions, and opaque pre-colon evidence tokens without imposing a
token grammar.

## Concern

The full existing preparation suite requires PyYAML/PyMuPDF. System Python
lacks those modules; its initial run stopped at missing `yaml`. The repository
Miniconda interpreter was used for a subsequent run, but this execution host
returned after its 30-second command window before an end-of-suite result was
available. The existing publisher suite and all focused tests did complete.

The final local commit SHA is recorded in the task handoff.
