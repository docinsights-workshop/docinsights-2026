# DocSem public test release receipt — 2026-09-05

## Release identity

- Source remote branch: `docsem-test-release` at
  `a4205880bfdd47aa3683050cd4a6ddf923fadffb`.
- Collision-free local preparation commit:
  `41c675bda8fa93662675f3dcd90fb7af4000cc22`. This does not claim that the
  preparation commit is on GitHub `main`.
- Release ID: `docsem-test-a4205880-r1`.

## Public dataset

- Dataset: `amitbcp/docinsights-2026-shared-task-data`.
- Final revision: `d9e1a394b46d2ac0a4dd87e12dd4a917a69f46e2`.
- Contents: 1,730 tasks, 1,730 PDFs, and 1,733 exact `test/**` paths.
- Sorted-ID digest:
  `e30896a0540726d0dafab507d0c4d6408030ef86a702bcbd127526e49c06b3a9`.
- Public HF task-manifest digest:
  `5fe8fbb8169b0c2b396fe155d263db36f4fa34b02a0cedd9075423b0bd3fc40d`.
- PDF inventory digest:
  `3fce062d7485c44c3986df8945eac069703adfd8c52571578763458e11748238`.

## Independent public readback

- 1,730/1,730 LFS SHA-256 values matched.
- The dataset is public; train and validation are unchanged.
- The tasks configuration has train, validation, and test splits.
- The labels configuration has train only.
- No public validation/test-label or private path is present.

## Public Space

- Space: `amitbcp/docsem-docinsights`.
- Revision: `282fb9d37d30b18497dc2b90648f7a9740ca2bf2`.
- Runtime: `RUNNING`.
- Root, `/config`, and `/info` returned HTTP 200.
- Validation defaults are preserved.
- The public-input notice is visible.
- Test submissions are closed.
- No final-test table is serialized.

## Verification and publication safety

- Local verification passed: 24 public-publisher tests, 19 existing publisher
  tests, 68 preparation tests, and 120 pinned-runtime Space tests.
- The first nine-worker upload hit HF's 2,500-request/five-minute rate limit.
  It was safely checkpointed with docs withheld and resumed with the reviewed
  two-worker bound.
- The only allowed non-test upload side effect was the exact 1,730-line
  test-PDF LFS extension to `.gitattributes`.
- No private test label, answer/evidence row, private repository, Space secret,
  scoring flag, or final-test row was published or changed.

## Pending activation gate

Authoritative private labels will be separately validated and installed later.
Keep `TEST_SUBMISSIONS_ENABLED=false` and
`TEST_PUBLIC_LEADERBOARD_ENABLED=false` until private hash reconciliation and
activation smoke tests.
