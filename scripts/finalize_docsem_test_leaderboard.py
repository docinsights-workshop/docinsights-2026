#!/usr/bin/env python3
"""Deterministically finalize the private DocSem held-out test ledger.

The command is a dry-run by default.  It reads one exact private Hugging Face
dataset revision, independently re-scores committed attempts with the pinned
repository scorer, and builds a seven-field public projection.  A write needs
both explicit confirmations and uses one exact-parent commit.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SPACE_ROOT = REPOSITORY_ROOT / "competition" / "hf-space"
SCORING_PATH = SPACE_ROOT / "scoring.py"
if str(SPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(SPACE_ROOT))

from scoring import score_predictions  # noqa: E402
from test_contract import (  # noqa: E402
    MAX_LEDGER_FILE_BYTES,
    bounded_private_text,
    repository_id,
    revision_digest,
    sha256_digest,
    validate_test_predictions,
)
from test_policy import (  # noqa: E402
    OAuthIdentity,
    canonical_submission_hash,
    select_best_attempt,
)


RELEASE_PATH = "private/test_release.json"
GOLD_PATH = "private/test_labels.jsonl"
PUBLIC_FINAL_PATH = "projections/test/public_final.json"
AUDIT_PATH = "private/test_finalization_audit.json"
ORGANIZER_PROJECTION_PATH = "projections/test/organizer_leaderboard.json"

MAX_FILES = 4096
MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_LABEL_BYTES = 64 * 1024 * 1024
MAX_AUDIT_RECORDS = 4096
MAX_ATTEMPTS = 30_000
METRIC_ABSOLUTE_TOLERANCE = 1e-12

PUBLIC_ROW_FIELDS = frozenset(
    {
        "rank",
        "hf_username",
        "team",
        "submission_name",
        "selected_attempt",
        "answer_accuracy",
        "evidence_f1",
    }
)
PUBLIC_PROJECTION_FIELDS = frozenset(
    {"schema_version", "split", "release_id", "task_manifest_sha256", "rows"}
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_ATTEMPT_PATH = re.compile(
    r"attempts/test/(?P<account>[0-9a-f]{64})/"
    r"(?P<submission>[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12})\.json\Z"
)
_ACCOUNT_PROJECTION_PATH = re.compile(
    r"projections/test/accounts/(?P<account>[0-9a-f]{64})\.json\Z"
)
_EXCLUSION_PATH = re.compile(
    r"exclusions/test/(?P<record>[A-Za-z0-9][A-Za-z0-9._-]{0,127})\.json\Z"
)
_ADJUDICATION_PATH = re.compile(
    r"adjudications/test/(?P<record>[A-Za-z0-9][A-Za-z0-9._-]{0,127})\.json\Z"
)
_FORBIDDEN_PUBLIC_PATH = re.compile(
    r"(?:^|/)(?:private|attempts/test|projections/test/accounts|"
    r"exclusions/test|adjudications/test)(?:/|$)",
    re.IGNORECASE,
)
_FINALIZATION_RELEASE_FIELDS = frozenset(
    {
        "finalized_at",
        "finalization_source_revision",
        "finalization_scorer_revision",
        "finalization_scorer_sha256",
        "final_projection_sha256",
        "finalization_audit_sha256",
    }
)
_ADJUDICATION_ACTIONS = frozenset(
    {
        "note",
        "exclude_account",
        "reinstate_account",
        "exclude_attempt",
        "reinstate_attempt",
    }
)


class FinalizationError(RuntimeError):
    """Sanitized refusal raised when a final leaderboard cannot be proven."""


@dataclass(frozen=True, repr=False)
class SnapshotRecord:
    """One bounded private-repository record with aggregate-only repr."""

    path: str
    sha256: str
    committed: bool
    value: object = field(repr=False, compare=False)

    def __post_init__(self):
        if (
            not _safe_path(self.path)
            or _SHA256.fullmatch(self.sha256) is None
            or not isinstance(self.committed, bool)
        ):
            raise FinalizationError("A private finalization record is malformed.")

    @classmethod
    def from_value(cls, path: str, value: object, committed: bool = True):
        raw = _canonical_json(value)
        return cls(path, _sha256(raw), committed, copy.deepcopy(value))

    def __repr__(self) -> str:
        return (
            "SnapshotRecord("
            f"path_kind={_path_kind(self.path)!r}, committed={self.committed!r}, "
            f"sha256={self.sha256!r})"
        )


@dataclass(frozen=True, repr=False)
class FinalizationSnapshot:
    """Exact private ledger and scorer inputs; sensitive values stay out of repr."""

    repo_id: str
    revision: str
    private: bool
    ancestor_revisions: tuple[str, ...]
    release: Mapping = field(repr=False, compare=False)
    release_bytes: bytes = field(repr=False, compare=False)
    labels: tuple[Mapping, ...] = field(repr=False, compare=False)
    gold_bytes: bytes = field(repr=False, compare=False)
    attempts: tuple[SnapshotRecord, ...] = field(repr=False, compare=False)
    exclusions: tuple[SnapshotRecord, ...] = field(
        default=(), repr=False, compare=False
    )
    adjudications: tuple[SnapshotRecord, ...] = field(
        default=(), repr=False, compare=False
    )
    projection_issue_codes: tuple[str, ...] = ()
    scorer_revision: str = ""
    scorer_code_sha256: str = ""
    scorer: Callable = field(default=score_predictions, repr=False, compare=False)
    existing_public_final: Mapping | None = field(
        default=None, repr=False, compare=False
    )
    existing_audit_manifest: Mapping | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self):
        try:
            repository_id(self.repo_id)
            revision_digest(self.revision)
            revision_digest(self.scorer_revision)
            sha256_digest(self.scorer_code_sha256)
            if self.private is not True:
                raise ValueError()
            if (
                not isinstance(self.ancestor_revisions, tuple)
                or not self.ancestor_revisions
                or len(self.ancestor_revisions) > 10_000
                or any(
                    _REVISION.fullmatch(item) is None
                    for item in self.ancestor_revisions
                )
                or len(set(self.ancestor_revisions)) != len(self.ancestor_revisions)
                or self.revision not in self.ancestor_revisions
                or not isinstance(self.release, Mapping)
                or not isinstance(self.release_bytes, bytes)
                or not isinstance(self.gold_bytes, bytes)
                or not isinstance(self.labels, tuple)
                or len(self.labels) > MAX_ATTEMPTS
                or len(self.attempts) > MAX_ATTEMPTS
                or len(self.exclusions) + len(self.adjudications) > MAX_AUDIT_RECORDS
                or not callable(self.scorer)
            ):
                raise ValueError()
        except (TypeError, ValueError):
            raise FinalizationError("The finalization snapshot is malformed.") from None

    def __repr__(self) -> str:
        return (
            "FinalizationSnapshot("
            f"revision={self.revision!r}, attempt_count={len(self.attempts)}, "
            f"exclusion_count={len(self.exclusions)}, "
            f"adjudication_count={len(self.adjudications)}, "
            f"projection_issue_count={len(self.projection_issue_codes)})"
        )

    def constructor_fields(self) -> dict:
        """Return a defensive constructor copy for audited transformations."""

        return {
            item.name: copy.deepcopy(getattr(self, item.name))
            if item.name != "scorer"
            else self.scorer
            for item in fields(self)
        }


@dataclass(frozen=True, repr=False)
class FinalizationPlan:
    """Bounded deterministic artifacts for one optional exact-parent commit."""

    source_revision: str
    scorer_revision: str
    public_projection: Mapping = field(repr=False, compare=False)
    audit_manifest: Mapping = field(repr=False, compare=False)
    finalized_release: Mapping = field(repr=False, compare=False)
    public_bytes: bytes = field(repr=False, compare=False)
    audit_bytes: bytes = field(repr=False, compare=False)
    release_bytes: bytes = field(repr=False, compare=False)
    projection_sha256: str
    audit_sha256: str
    release_sha256: str
    eligible_attempt_count: int
    excluded_attempt_count: int
    selected_account_count: int
    already_finalized: bool

    def __post_init__(self):
        try:
            revision_digest(self.source_revision)
            revision_digest(self.scorer_revision)
            for value in (
                self.projection_sha256,
                self.audit_sha256,
                self.release_sha256,
            ):
                sha256_digest(value)
            if any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in (
                    self.eligible_attempt_count,
                    self.excluded_attempt_count,
                    self.selected_account_count,
                )
            ) or not isinstance(self.already_finalized, bool):
                raise ValueError()
            if any(
                not isinstance(payload, bytes) or len(payload) > MAX_LEDGER_FILE_BYTES
                for payload in (self.public_bytes, self.audit_bytes, self.release_bytes)
            ):
                raise ValueError()
        except (TypeError, ValueError):
            raise FinalizationError("The finalization plan is malformed.") from None

    def dry_run_summary(self) -> dict:
        """Return only non-sensitive counts, hashes, and exact revisions."""

        return {
            "source_revision": self.source_revision,
            "scorer_revision": self.scorer_revision,
            "eligible_attempt_count": self.eligible_attempt_count,
            "excluded_attempt_count": self.excluded_attempt_count,
            "selected_account_count": self.selected_account_count,
            "projection_sha256": self.projection_sha256,
            "audit_sha256": self.audit_sha256,
            "release_sha256": self.release_sha256,
            "already_finalized": self.already_finalized,
        }

    def __repr__(self) -> str:
        return f"FinalizationPlan({self.dry_run_summary()!r})"


def load_finalization_snapshot(
    repo_id,
    revision,
    token,
    *,
    scorer_revision,
    scorer_code_sha256,
    scorer_bytes_at_revision=None,
    api=None,
) -> FinalizationSnapshot:
    """Load one exact private repository SHA and the exact local scorer bytes."""

    try:
        repository = repository_id(repo_id)
        pinned = revision_digest(revision)
        scorer_rev = revision_digest(scorer_revision)
        expected_scorer_sha = sha256_digest(scorer_code_sha256)
        if not isinstance(token, str) or not token.strip() or len(token) > 4096:
            raise ValueError()
        actual_scorer_sha = _sha256(
            _bounded_regular_file(SCORING_PATH, 2 * 1024 * 1024)
        )
        revision_reader = scorer_bytes_at_revision or _git_scorer_bytes
        pinned_scorer_bytes = revision_reader(scorer_rev)
        if (
            not isinstance(pinned_scorer_bytes, bytes)
            or not pinned_scorer_bytes
            or len(pinned_scorer_bytes) > 2 * 1024 * 1024
            or _sha256(pinned_scorer_bytes) != expected_scorer_sha
            or actual_scorer_sha != expected_scorer_sha
        ):
            raise ValueError()
        hub = api if api is not None else HfApi(token=token)
        info = hub.repo_info(
            repository, repo_type="dataset", revision=pinned, token=token
        )
        if (
            getattr(info, "sha", None) != pinned
            or getattr(info, "private", None) is not True
        ):
            raise ValueError()
        commits = tuple(
            getattr(item, "commit_id", None)
            for item in hub.list_repo_commits(
                repository, repo_type="dataset", revision=pinned, token=token
            )
        )
        if (
            not commits
            or pinned not in commits
            or len(commits) > 10_000
            or len(set(commits)) != len(commits)
            or any(
                not isinstance(item, str) or _REVISION.fullmatch(item) is None
                for item in commits
            )
        ):
            raise ValueError()
        paths = list(
            hub.list_repo_files(
                repository, repo_type="dataset", revision=pinned, token=token
            )
        )
        selected = _select_snapshot_paths(paths)
        if RELEASE_PATH not in selected or GOLD_PATH not in selected:
            raise ValueError()
        raw_files: dict[str, bytes] = {}
        total = 0
        with tempfile.TemporaryDirectory(prefix="docsem-finalize-") as cache_name:
            cache = Path(cache_name)
            cache.chmod(0o700)
            for path in sorted(selected):
                local = hub.hf_hub_download(
                    repository,
                    path,
                    repo_type="dataset",
                    revision=pinned,
                    token=token,
                    cache_dir=str(cache),
                )
                limit = MAX_LABEL_BYTES if path == GOLD_PATH else MAX_LEDGER_FILE_BYTES
                payload = _bounded_regular_file(Path(local), limit)
                total += len(payload)
                if total > MAX_TOTAL_BYTES:
                    raise ValueError()
                raw_files[path] = payload

        release_value = _decode_json(raw_files[RELEASE_PATH])
        labels = tuple(_decode_jsonl(raw_files[GOLD_PATH]))
        if _sha256(raw_files[GOLD_PATH]) != release_value.get("gold_sha256"):
            raise ValueError()
        _validate_label_rows(labels)

        projection_refs, projection_issues = _projection_references(
            raw_files, release_value
        )
        attempts = []
        exclusions = []
        adjudications = []
        for path, payload in sorted(raw_files.items()):
            if _ATTEMPT_PATH.fullmatch(path):
                value = _decode_json(payload)
                match = _ATTEMPT_PATH.fullmatch(path)
                reference = projection_refs.get(
                    (match.group("account"), match.group("submission"))
                )
                digest = _sha256(payload)
                committed = bool(
                    reference
                    and reference.get("record_sha256") == digest
                    and reference.get("attempt_number") == value.get("attempt_number")
                )
                attempts.append(SnapshotRecord(path, digest, committed, value))
            elif _EXCLUSION_PATH.fullmatch(path):
                exclusions.append(
                    SnapshotRecord(path, _sha256(payload), True, _decode_json(payload))
                )
            elif _ADJUDICATION_PATH.fullmatch(path):
                adjudications.append(
                    SnapshotRecord(path, _sha256(payload), True, _decode_json(payload))
                )

        existing_public = (
            _decode_json(raw_files[PUBLIC_FINAL_PATH])
            if PUBLIC_FINAL_PATH in raw_files
            else None
        )
        existing_audit = (
            _decode_json(raw_files[AUDIT_PATH]) if AUDIT_PATH in raw_files else None
        )
        return FinalizationSnapshot(
            repo_id=repository,
            revision=pinned,
            private=True,
            ancestor_revisions=commits,
            release=release_value,
            release_bytes=raw_files[RELEASE_PATH],
            labels=labels,
            gold_bytes=raw_files[GOLD_PATH],
            attempts=tuple(attempts),
            exclusions=tuple(exclusions),
            adjudications=tuple(adjudications),
            projection_issue_codes=tuple(sorted(projection_issues)),
            scorer_revision=scorer_rev,
            scorer_code_sha256=actual_scorer_sha,
            scorer=score_predictions,
            existing_public_final=existing_public,
            existing_audit_manifest=existing_audit,
        )
    except FinalizationError:
        raise
    except Exception:
        raise FinalizationError(
            "The private finalization snapshot is unavailable."
        ) from None


def build_finalization(snapshot, now) -> FinalizationPlan:
    """Build and verify one deterministic best-of-three finalization plan."""

    if not isinstance(snapshot, FinalizationSnapshot):
        raise FinalizationError("The finalization snapshot is malformed.")
    if snapshot.projection_issue_codes:
        raise FinalizationError(
            "Private test projections failed integrity verification."
        )
    current = _require_utc(now)
    release_value, close_at = _validate_release(snapshot, current)
    base_release = _base_release(release_value)
    finalized = release_value.get("finalized") is True

    _validate_scorer(snapshot)
    _validate_label_rows(snapshot.labels)
    if _sha256(snapshot.gold_bytes) != base_release.get("gold_sha256"):
        raise FinalizationError("The pinned test gold digest does not match.")

    source_revision = snapshot.revision
    finalized_at = _format_utc(current)
    if finalized:
        if not isinstance(snapshot.existing_audit_manifest, Mapping):
            raise FinalizationError("Finalized test artifacts are incomplete.")
        source_revision = snapshot.existing_audit_manifest.get("source_revision")
        finalized_at = snapshot.existing_audit_manifest.get("finalized_at")
        if (
            not isinstance(source_revision, str)
            or source_revision not in snapshot.ancestor_revisions
            or _REVISION.fullmatch(source_revision) is None
            or _parse_utc(finalized_at) is None
        ):
            raise FinalizationError("Finalized test artifacts are inconsistent.")

    candidates, excluded = _eligible_attempts(snapshot, base_release, close_at)
    decisions, applied_records = _audit_decisions(snapshot, base_release, current)
    candidates, audit_excluded = _apply_audit_decisions(candidates, decisions)
    excluded.extend(audit_excluded)

    rescored = []
    metric_mismatches = []
    for item in candidates:
        record = item.value
        try:
            metrics = snapshot.scorer(record["predictions"], list(snapshot.labels))
        except Exception:
            raise FinalizationError("Independent test rescoring failed.") from None
        if not _metrics_equal(record.get("metrics"), metrics):
            metric_mismatches.append(
                {
                    "path_sha256": item.sha256,
                    "submission_id": record.get("submission_id"),
                    "reason_code": "stored_metric_mismatch",
                }
            )
            continue
        rescored.append((item, metrics))
    if metric_mismatches:
        raise FinalizationError(
            "Stored test scores do not match independent rescoring; finalization refused."
        )

    selected = _select_accounts(rescored)
    public_projection = _public_projection(base_release, selected)
    audit_public_projection(public_projection)
    public_bytes = _bounded_json(public_projection)
    projection_sha = _sha256(public_bytes)

    input_manifest = _input_manifest(snapshot, base_release)
    audit_manifest = {
        "schema_version": 1,
        "split": "test",
        "release_id": base_release["release_id"],
        "source_revision": source_revision,
        "finalized_at": finalized_at,
        "close_at": _format_utc(close_at),
        "task_manifest_sha256": base_release["task_manifest_sha256"],
        "gold_sha256": base_release["gold_sha256"],
        "scorer_revision": snapshot.scorer_revision,
        "scorer_code_sha256": snapshot.scorer_code_sha256,
        "metric_absolute_tolerance": METRIC_ABSOLUTE_TOLERANCE,
        "input_manifest_sha256": _sha256(_canonical_json(input_manifest)),
        "input_records": input_manifest["records"],
        "projection_issue_codes": list(snapshot.projection_issue_codes),
        "eligible_attempt_count": len(rescored),
        "excluded_attempt_count": len(excluded),
        "selected_account_count": len(selected),
        "eligible_attempts": [
            {
                "account_key": item.value["account_key"],
                "submission_id": item.value["submission_id"],
                "attempt_number": item.value["attempt_number"],
                "record_sha256": item.sha256,
                "selected": item.value["submission_id"]
                in {chosen.value["submission_id"] for chosen, _ in selected},
                "answer_accuracy": metrics["answer_accuracy"],
                "evidence_f1": metrics["evidence_f1"],
                "rescored_metrics_sha256": _sha256(_canonical_json(metrics)),
            }
            for item, metrics in rescored
        ],
        "excluded_attempts": sorted(
            excluded,
            key=lambda item: (
                str(item.get("submission_id", "")),
                str(item.get("path_sha256", "")),
            ),
        ),
        "applied_audit_records": applied_records,
        "public_projection_sha256": projection_sha,
    }
    audit_bytes = _bounded_json(audit_manifest)
    audit_sha = _sha256(audit_bytes)
    finalized_release = {
        **base_release,
        "enabled": False,
        "finalized": True,
        "finalized_at": finalized_at,
        "finalization_source_revision": source_revision,
        "finalization_scorer_revision": snapshot.scorer_revision,
        "finalization_scorer_sha256": snapshot.scorer_code_sha256,
        "final_projection_sha256": projection_sha,
        "finalization_audit_sha256": audit_sha,
    }
    final_release_bytes = _bounded_json(finalized_release)
    release_sha = _sha256(final_release_bytes)

    if finalized:
        if (
            snapshot.existing_public_final != public_projection
            or snapshot.existing_audit_manifest != audit_manifest
            or release_value != finalized_release
        ):
            raise FinalizationError("Finalized test artifacts are inconsistent.")

    return FinalizationPlan(
        source_revision=source_revision,
        scorer_revision=snapshot.scorer_revision,
        public_projection=public_projection,
        audit_manifest=audit_manifest,
        finalized_release=finalized_release,
        public_bytes=public_bytes,
        audit_bytes=audit_bytes,
        release_bytes=final_release_bytes,
        projection_sha256=projection_sha,
        audit_sha256=audit_sha,
        release_sha256=release_sha,
        eligible_attempt_count=len(rescored),
        excluded_attempt_count=len(excluded),
        selected_account_count=len(selected),
        already_finalized=finalized,
    )


def audit_public_projection(value) -> bool:
    """Fail closed unless the public final artifact has only allowed fields."""

    if not isinstance(value, Mapping) or set(value) != PUBLIC_PROJECTION_FIELDS:
        raise FinalizationError("The public final projection failed its privacy audit.")
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or value.get("split") != "test"
        or not isinstance(value.get("release_id"), str)
        or not value["release_id"].strip()
        or _SHA256.fullmatch(str(value.get("task_manifest_sha256", ""))) is None
        or not isinstance(value.get("rows"), list)
    ):
        raise FinalizationError("The public final projection failed its privacy audit.")
    for expected_rank, row in enumerate(value["rows"], start=1):
        if not isinstance(row, Mapping) or set(row) != PUBLIC_ROW_FIELDS:
            raise FinalizationError(
                "The public final projection failed its privacy audit."
            )
        if (
            type(row.get("rank")) is not int
            or row["rank"] != expected_rank
            or type(row.get("selected_attempt")) is not int
            or not 1 <= row["selected_attempt"] <= 3
        ):
            raise FinalizationError(
                "The public final projection failed its privacy audit."
            )
        for field_name in ("hf_username", "team", "submission_name"):
            text = row.get(field_name)
            if (
                not isinstance(text, str)
                or not text.strip()
                or len(text) > 4096
                or _FORBIDDEN_PUBLIC_PATH.search(text.replace("\\", "/"))
            ):
                raise FinalizationError(
                    "The public final projection failed its privacy audit."
                )
        for field_name in ("answer_accuracy", "evidence_f1"):
            metric = row.get(field_name)
            if (
                type(metric) is not float
                or not math.isfinite(metric)
                or not 0.0 <= metric <= 1.0
            ):
                raise FinalizationError(
                    "The public final projection failed its privacy audit."
                )
    return True


def commit_finalization(
    api,
    snapshot,
    plan,
    *,
    token,
    expected_private_sha,
    now,
    yes,
    maintenance_confirmed,
):
    """Optionally persist all three finalization artifacts in one CAS commit."""

    if not isinstance(snapshot, FinalizationSnapshot) or not isinstance(
        plan, FinalizationPlan
    ):
        raise FinalizationError("The finalization commit request is malformed.")
    try:
        expected = revision_digest(expected_private_sha)
    except ValueError:
        raise FinalizationError("The expected private revision is invalid.") from None
    if expected != snapshot.revision:
        raise FinalizationError(
            "The expected private revision does not match the plan."
        )

    current = _require_utc(now)
    planned_at = _parse_utc(plan.audit_manifest.get("finalized_at"))
    close_at = _parse_utc(plan.audit_manifest.get("close_at"))
    if (
        planned_at is None
        or close_at is None
        or planned_at < close_at
        or current < close_at
        or current < planned_at
        or plan.finalized_release.get("finalized_at") != _format_utc(planned_at)
    ):
        raise FinalizationError("The finalization plan time is invalid.")

    _require_current_private_head(api, snapshot.repo_id, expected, token)
    if plan.already_finalized:
        return None
    if yes is not True or maintenance_confirmed is not True:
        raise FinalizationError(
            "Finalization requires --yes and --maintenance-confirmed."
        )
    fresh_release_bytes = _download_exact(
        api, snapshot.repo_id, expected, token, RELEASE_PATH
    )
    if _sha256(fresh_release_bytes) != _sha256(snapshot.release_bytes):
        raise FinalizationError("The private release changed before finalization.")
    fresh_release = _decode_json(fresh_release_bytes)
    fresh_snapshot = FinalizationSnapshot(
        **{
            **snapshot.constructor_fields(),
            "release": fresh_release,
            "release_bytes": fresh_release_bytes,
        }
    )
    # ``finalized_at`` is part of every plan hash.  The precommit rebuild must
    # therefore reuse that immutable planned instant while ``current`` above
    # independently proves that the actual commit attempt is still after both
    # the close and the plan time.
    fresh_plan = build_finalization(fresh_snapshot, planned_at)
    if (
        fresh_plan.projection_sha256 != plan.projection_sha256
        or fresh_plan.audit_sha256 != plan.audit_sha256
        or fresh_plan.release_sha256 != plan.release_sha256
    ):
        raise FinalizationError("The finalization plan changed before commit.")
    artifacts = (
        (PUBLIC_FINAL_PATH, plan.public_bytes, plan.projection_sha256),
        (AUDIT_PATH, plan.audit_bytes, plan.audit_sha256),
        (RELEASE_PATH, plan.release_bytes, plan.release_sha256),
    )
    if any(_sha256(payload) != digest for _, payload, digest in artifacts):
        raise FinalizationError("The finalization plan artifacts are invalid.")
    operations = [
        CommitOperationAdd(PUBLIC_FINAL_PATH, plan.public_bytes),
        CommitOperationAdd(AUDIT_PATH, plan.audit_bytes),
        CommitOperationAdd(RELEASE_PATH, plan.release_bytes),
    ]
    try:
        result = api.create_commit(
            repo_id=snapshot.repo_id,
            repo_type="dataset",
            revision="main",
            parent_commit=expected,
            operations=operations,
            commit_message="Finalize DocSem held-out test leaderboard",
            token=token,
        )
        returned = getattr(result, "oid", None) or getattr(result, "commit_id", None)
    except Exception:
        raise FinalizationError("The finalization commit did not succeed.") from None

    # A successful API acknowledgement is not proof that the exact atomic
    # write became the private main head.  Verify the returned revision and all
    # three immutable artifacts before reporting success.  A failure in this
    # phase may follow a real write, so the refusal explicitly directs a rerun
    # from the newly observed head instead of inviting a blind retry.
    try:
        revision = revision_digest(returned)
        if revision == expected:
            raise ValueError()
        _require_current_private_head(api, snapshot.repo_id, revision, token)
        for path, expected_bytes, expected_digest in artifacts:
            observed = _download_exact(api, snapshot.repo_id, revision, token, path)
            if observed != expected_bytes or _sha256(observed) != expected_digest:
                raise ValueError()
    except Exception:
        raise FinalizationError(
            "The finalization commit may have succeeded but could not be verified; "
            "rerun from the current private revision."
        ) from None
    return revision


def _validate_release(snapshot: FinalizationSnapshot, now: dt.datetime):
    release = snapshot.release
    required = (
        "schema_version",
        "release_id",
        "task_manifest_sha256",
        "gold_sha256",
        "enabled",
        "finalized",
        "max_attempts",
        "feedback_policy",
        "open_at",
        "close_at",
        "public_revision",
        "public_repo_id",
        "task_manifest_path",
    )
    try:
        if not isinstance(release, Mapping) or any(
            name not in release for name in required
        ):
            raise ValueError()
        if (
            type(release["schema_version"]) is not int
            or release["schema_version"] != 1
            or not isinstance(release["release_id"], str)
            or not release["release_id"].strip()
            or release["enabled"] is not False
            or not isinstance(release["finalized"], bool)
            or type(release["max_attempts"]) is not int
            or release["max_attempts"] != 3
            or release["feedback_policy"] != "first-attempt-only"
            or release["task_manifest_path"] != "test/tasks.jsonl"
        ):
            raise ValueError()
        sha256_digest(release["task_manifest_sha256"])
        sha256_digest(release["gold_sha256"])
        revision_digest(release["public_revision"])
        repository_id(release["public_repo_id"])
        opened = _parse_utc(release["open_at"])
        closed = _parse_utc(release["close_at"])
        if opened is None or closed is None or closed <= opened or now < closed:
            raise ValueError()
        if _sha256(snapshot.release_bytes) != _sha256(_canonical_json(release)):
            # Whitespace differences are permitted for loaded source bytes, but
            # semantic content must have decoded canonically without duplicates.
            _decode_json(snapshot.release_bytes)
        return release, closed
    except (TypeError, ValueError):
        raise FinalizationError(
            "The test release is not closed and ready for finalization."
        ) from None


def _base_release(release: Mapping) -> dict:
    result = {
        key: copy.deepcopy(value)
        for key, value in release.items()
        if key not in _FINALIZATION_RELEASE_FIELDS
    }
    result["enabled"] = False
    result["finalized"] = False
    return result


def _validate_scorer(snapshot: FinalizationSnapshot) -> None:
    try:
        revision_digest(snapshot.scorer_revision)
        actual = _sha256(_bounded_regular_file(SCORING_PATH, 2 * 1024 * 1024))
        if actual != snapshot.scorer_code_sha256:
            raise ValueError()
    except (TypeError, ValueError):
        raise FinalizationError("The pinned scorer contract does not match.") from None


def _eligible_attempts(snapshot, release, close_at):
    records = sorted(snapshot.attempts, key=lambda item: item.path)
    ids = Counter(
        item.value.get("submission_id")
        for item in records
        if isinstance(item.value, Mapping)
        and isinstance(item.value.get("submission_id"), str)
    )
    hashes = Counter(
        item.value.get("submission_hash")
        for item in records
        if isinstance(item.value, Mapping)
        and isinstance(item.value.get("submission_hash"), str)
    )
    account_numbers = Counter(
        (item.value.get("account_key"), item.value.get("attempt_number"))
        for item in records
        if isinstance(item.value, Mapping)
        and isinstance(item.value.get("account_key"), str)
        and type(item.value.get("attempt_number")) is int
    )
    accepted = []
    excluded = []
    for item in records:
        reasons = _attempt_reasons(
            item, snapshot, release, close_at, ids, hashes, account_numbers
        )
        if reasons:
            excluded.append(_excluded_attempt(item, reasons))
        else:
            accepted.append(item)
    return accepted, excluded


def _attempt_reasons(item, snapshot, release, close_at, ids, hashes, account_numbers):
    if not item.committed:
        return {"uncommitted"}
    record = item.value
    match = _ATTEMPT_PATH.fullmatch(item.path)
    if not isinstance(record, Mapping) or match is None:
        return {"malformed"}
    reasons = set()
    state = {
        "schema_version": 2,
        "split": "test",
        "release_id": release["release_id"],
        "task_manifest_sha256": release["task_manifest_sha256"],
        "gold_sha256": release["gold_sha256"],
    }
    if record.get("release_id") != state["release_id"] or record.get("split") != "test":
        reasons.add("wrong_release")
    if record.get("task_manifest_sha256") != state["task_manifest_sha256"]:
        reasons.add("wrong_task")
    if (
        record.get("gold_sha256") != state["gold_sha256"]
        or record.get("scoring_gold_sha256") != state["gold_sha256"]
    ):
        reasons.add("wrong_gold")
    if (
        record.get("schema_version") != 2
        or type(record.get("schema_version")) is not int
    ):
        reasons.add("malformed")
    submission_id = record.get("submission_id")
    account_key = record.get("account_key")
    number = record.get("attempt_number")
    if (
        submission_id != match.group("submission")
        or account_key != match.group("account")
        or not _uuid4(submission_id)
        or type(number) is not int
        or not 1 <= number <= 3
    ):
        reasons.add("malformed")
    if ids.get(submission_id, 0) > 1:
        reasons.add("duplicate_submission_id")
    submission_hash = record.get("submission_hash")
    if (
        not isinstance(submission_hash, str)
        or _SHA256.fullmatch(submission_hash) is None
    ):
        reasons.add("malformed")
    elif hashes.get(submission_hash, 0) > 1:
        reasons.add("duplicate_submission_hash")
    try:
        if account_numbers.get((account_key, number), 0) > 1:
            reasons.add("duplicate_attempt_number")
    except TypeError:
        reasons.add("malformed")
    submitted = _parse_utc(record.get("submitted_at"))
    opened = _parse_utc(release.get("open_at"))
    if submitted is None:
        reasons.add("malformed")
    elif opened is None or submitted < opened:
        reasons.add("pre_window")
    elif submitted >= close_at:
        reasons.add("post_cutoff")
    if (
        record.get("scoring_public_revision") != release["public_revision"]
        or record.get("scoring_public_repo_id") != release["public_repo_id"]
        or record.get("scoring_task_manifest_path") != release["task_manifest_path"]
        or record.get("scoring_private_revision") not in snapshot.ancestor_revisions
    ):
        reasons.add("wrong_provenance")
    try:
        for name in (
            "hf_subject",
            "hf_username",
            "verified_email",
            "team",
            "participant_names",
            "submission_name",
        ):
            bounded_private_text(record.get(name), name)
        if hashlib.sha256(record["hf_subject"].encode()).hexdigest() != account_key:
            raise ValueError()
        validate_test_predictions(record.get("predictions"))
        identity = OAuthIdentity(
            record["hf_subject"], record["hf_username"], record["verified_email"]
        )
        expected_hash = canonical_submission_hash(
            record["predictions"], "test", release["release_id"], identity
        )
        if record.get("submission_hash") != expected_hash:
            raise ValueError()
        if not _valid_metrics_shape(record.get("metrics"), len(snapshot.labels)):
            raise ValueError()
    except (KeyError, TypeError, ValueError):
        reasons.add("malformed")
    return reasons


def _valid_metrics_shape(metrics, expected_examples):
    expected = {
        "answer_accuracy",
        "evidence_exact_match",
        "evidence_f1",
        "examples",
        "per_example",
    }
    if not isinstance(metrics, Mapping) or set(metrics) != expected:
        return False
    if (
        type(metrics.get("examples")) is not int
        or metrics["examples"] != expected_examples
    ):
        return False
    for name in ("answer_accuracy", "evidence_exact_match", "evidence_f1"):
        value = metrics.get(name)
        if type(value) is not float or not math.isfinite(value) or not 0 <= value <= 1:
            return False
    per_example = metrics.get("per_example")
    if not isinstance(per_example, list) or len(per_example) != expected_examples:
        return False
    for row in per_example:
        if not isinstance(row, Mapping) or set(row) != {
            "instance_id",
            "answer_exact_match",
            "evidence_exact_match",
            "evidence_f1",
        }:
            return False
        for name in ("answer_exact_match", "evidence_exact_match", "evidence_f1"):
            value = row.get(name)
            if (
                type(value) is not float
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                return False
    return True


def _audit_decisions(snapshot, release, now):
    events = []
    seen_ids = set()
    known_accounts = {
        item.value.get("account_key")
        for item in snapshot.attempts
        if isinstance(item.value, Mapping)
        and isinstance(item.value.get("account_key"), str)
    }
    known_attempts = {
        (item.value.get("account_key"), item.value.get("submission_id"))
        for item in snapshot.attempts
        if isinstance(item.value, Mapping)
        and isinstance(item.value.get("account_key"), str)
        and isinstance(item.value.get("submission_id"), str)
    }
    for item in snapshot.exclusions:
        record_id, created = _validate_audit_record(item, release, now, "exclusion")
        if item.value.get("account_key") not in known_accounts:
            raise FinalizationError("A private exclusion target is invalid.")
        if record_id in seen_ids:
            raise FinalizationError("Private audit records contain a duplicate ID.")
        seen_ids.add(record_id)
        events.append(
            (
                created,
                record_id,
                "exclude_account",
                str(item.value["account_key"]),
                None,
                str(item.value["reason_code"]),
                item.sha256,
            )
        )
    for item in snapshot.adjudications:
        record_id, created = _validate_audit_record(item, release, now, "adjudication")
        if record_id in seen_ids:
            raise FinalizationError("Private audit records contain a duplicate ID.")
        seen_ids.add(record_id)
        action = item.value.get("action")
        submission_id = item.value.get("submission_id")
        if action not in _ADJUDICATION_ACTIONS:
            raise FinalizationError("A private adjudication action is invalid.")
        if action.endswith("attempt") and not _uuid4(submission_id):
            raise FinalizationError("A private adjudication target is invalid.")
        if action == "note" and submission_id is not None and not _uuid4(submission_id):
            raise FinalizationError("A private adjudication target is invalid.")
        if (
            action not in {"note", "exclude_attempt", "reinstate_attempt"}
            and submission_id is not None
        ):
            raise FinalizationError("A private adjudication target is invalid.")
        account_key = item.value.get("account_key")
        if account_key not in known_accounts or (
            submission_id is not None
            and (account_key, submission_id) not in known_attempts
        ):
            raise FinalizationError("A private adjudication target is invalid.")
        events.append(
            (
                created,
                record_id,
                action,
                str(item.value["account_key"]),
                submission_id,
                str(item.value["reason_code"]),
                item.sha256,
            )
        )
    events.sort(key=lambda item: (item[0], item[1]))
    account_excluded = set()
    attempt_excluded = set()
    applied = []
    for (
        created,
        record_id,
        action,
        account,
        submission_id,
        reason,
        record_sha,
    ) in events:
        if action == "exclude_account":
            account_excluded.add(account)
        elif action == "reinstate_account":
            account_excluded.discard(account)
        elif action == "exclude_attempt":
            attempt_excluded.add((account, submission_id))
        elif action == "reinstate_attempt":
            attempt_excluded.discard((account, submission_id))
        applied.append(
            {
                "record_id": record_id,
                "record_sha256": record_sha,
                "created_at": _format_utc(created),
                "action": action,
                "account_key": account,
                "submission_id": submission_id,
                "reason_code": reason,
            }
        )
    return (account_excluded, attempt_excluded), applied


def _validate_audit_record(item, release, now, kind):
    pattern = _EXCLUSION_PATH if kind == "exclusion" else _ADJUDICATION_PATH
    match = pattern.fullmatch(item.path)
    value = item.value
    if not item.committed or match is None or not isinstance(value, Mapping):
        raise FinalizationError("A private audit record is malformed.")
    record_id = value.get("record_id")
    created = _parse_utc(value.get("created_at"))
    if (
        record_id != match.group("record")
        or _IDENTIFIER.fullmatch(str(record_id or "")) is None
        or value.get("schema_version") != 2
        or type(value.get("schema_version")) is not int
        or value.get("split") != "test"
        or value.get("release_id") != release["release_id"]
        or value.get("task_manifest_sha256") != release["task_manifest_sha256"]
        or value.get("gold_sha256") != release["gold_sha256"]
        or _SHA256.fullmatch(str(value.get("account_key", ""))) is None
        or not isinstance(value.get("reason_code"), str)
        or not value["reason_code"].strip()
        or len(value["reason_code"]) > 4096
        or created is None
        or created > now
    ):
        raise FinalizationError("A private audit record is malformed.")
    return record_id, created


def _apply_audit_decisions(candidates, decisions):
    excluded_accounts, excluded_attempts = decisions
    included = []
    excluded = []
    for item in candidates:
        record = item.value
        if record["account_key"] in excluded_accounts:
            excluded.append(_excluded_attempt(item, {"excluded_account"}))
        elif (record["account_key"], record["submission_id"]) in excluded_attempts:
            excluded.append(_excluded_attempt(item, {"excluded_attempt"}))
        else:
            included.append(item)
    return included, excluded


def _select_accounts(rescored):
    grouped = defaultdict(list)
    for item, metrics in rescored:
        grouped[item.value["account_key"]].append((item, metrics))
    selected = []
    for account in sorted(grouped):
        selected.append(min(grouped[account], key=_attempt_rank_key))
    selected.sort(key=_attempt_rank_key)
    return selected


def _attempt_rank_key(entry):
    item, metrics = entry
    return (
        -float(metrics["answer_accuracy"]),
        -float(metrics["evidence_f1"]),
        _parse_utc(item.value["submitted_at"]),
        item.value["submission_id"],
    )


def _public_projection(release, selected):
    rows = []
    for rank, (item, metrics) in enumerate(selected, start=1):
        record = item.value
        rows.append(
            {
                "rank": rank,
                "hf_username": str(record["hf_username"]),
                "team": str(record["team"]),
                "submission_name": str(record["submission_name"]),
                "selected_attempt": int(record["attempt_number"]),
                "answer_accuracy": float(metrics["answer_accuracy"]),
                "evidence_f1": float(metrics["evidence_f1"]),
            }
        )
    return {
        "schema_version": 1,
        "split": "test",
        "release_id": release["release_id"],
        "task_manifest_sha256": release["task_manifest_sha256"],
        "rows": rows,
    }


def _input_manifest(snapshot, release):
    records = [
        {"kind": "attempt", "path": item.path, "sha256": item.sha256}
        for item in snapshot.attempts
    ]
    records.extend(
        {"kind": "exclusion", "path": item.path, "sha256": item.sha256}
        for item in snapshot.exclusions
    )
    records.extend(
        {"kind": "adjudication", "path": item.path, "sha256": item.sha256}
        for item in snapshot.adjudications
    )
    records.sort(key=lambda item: (item["kind"], item["path"]))
    return {
        "release_sha256": _sha256(_canonical_json(release)),
        "gold_sha256": _sha256(snapshot.gold_bytes),
        "scorer_revision": snapshot.scorer_revision,
        "scorer_code_sha256": snapshot.scorer_code_sha256,
        "records": records,
    }


def _metrics_equal(stored, recomputed):
    if not isinstance(stored, Mapping) or not isinstance(recomputed, Mapping):
        return False
    if set(stored) != set(recomputed):
        return False
    for key in stored:
        left = stored[key]
        right = recomputed[key]
        if isinstance(left, list) or isinstance(right, list):
            if (
                not isinstance(left, list)
                or not isinstance(right, list)
                or len(left) != len(right)
            ):
                return False
            if any(not _metrics_equal(a, b) for a, b in zip(left, right)):
                return False
        elif isinstance(left, Mapping) or isinstance(right, Mapping):
            if not _metrics_equal(left, right):
                return False
        elif type(left) is float or type(right) is float:
            if (
                not isinstance(left, (int, float))
                or isinstance(left, bool)
                or not isinstance(right, (int, float))
                or isinstance(right, bool)
                or not math.isclose(
                    float(left),
                    float(right),
                    rel_tol=0.0,
                    abs_tol=METRIC_ABSOLUTE_TOLERANCE,
                )
            ):
                return False
        elif left != right or type(left) is not type(right):
            return False
    return True


def _excluded_attempt(item, reasons):
    ordered = sorted(reasons)
    record = item.value if isinstance(item.value, Mapping) else {}
    return {
        "path_sha256": item.sha256,
        "submission_id": record.get("submission_id"),
        "account_key": record.get("account_key"),
        "reason_code": ordered[0],
        "reason_codes": ordered,
    }


def _validate_label_rows(labels):
    if (
        not isinstance(labels, (list, tuple))
        or not labels
        or len(labels) > MAX_ATTEMPTS
    ):
        raise FinalizationError("The pinned test labels are malformed.")
    seen = set()
    for row in labels:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"instance_id", "answer", "evidence"}
            or not isinstance(row.get("instance_id"), str)
            or not row["instance_id"].strip()
            or row["instance_id"] in seen
            or not isinstance(row.get("answer"), str)
            or not row["answer"].strip()
            or not isinstance(row.get("evidence"), list)
            or not row["evidence"]
            or any(
                not isinstance(value, str) or not value.strip()
                for value in row["evidence"]
            )
        ):
            raise FinalizationError("The pinned test labels are malformed.")
        seen.add(row["instance_id"])


def _projection_references(raw_files, release):
    references = {}
    issues = set()
    expected_state = {
        "schema_version": 2,
        "split": "test",
        "release_id": release.get("release_id"),
        "task_manifest_sha256": release.get("task_manifest_sha256"),
        "gold_sha256": release.get("gold_sha256"),
    }
    state_fields = set(expected_state)
    reference_fields = state_fields | {
        "submission_id",
        "attempt_number",
        "record_sha256",
    }
    account_fields = state_fields | {
        "account_key",
        "attempts",
        "best_submission_id",
    }
    organizer_row_fields = state_fields | {
        "account_key",
        "attempt_count",
        "best_submission_id",
        "hf_subject",
        "hf_username",
        "verified_email",
        "team",
        "participant_names",
        "submission_name",
        "submitted_at",
        "attempt_number",
        "metrics",
    }

    def matches_state(value):
        return isinstance(value, Mapping) and all(
            value.get(name) == expected
            and (name != "schema_version" or type(value.get(name)) is int)
            for name, expected in expected_state.items()
        )

    attempt_files = {}
    for path, payload in sorted(raw_files.items()):
        match = _ATTEMPT_PATH.fullmatch(path)
        if match is None:
            continue
        key = (match.group("account"), match.group("submission"))
        try:
            record = _decode_json(payload)
        except Exception:
            issues.add("account_projection_invalid")
            continue
        attempt_files[key] = {
            "record": record,
            "record_sha256": _sha256(payload),
        }

    account_records = {}
    account_best = {}
    account_paths = sorted(
        path for path in raw_files if _ACCOUNT_PROJECTION_PATH.fullmatch(path)
    )
    for path in account_paths:
        match = _ACCOUNT_PROJECTION_PATH.fullmatch(path)
        account = match.group("account")
        try:
            projection = _decode_json(raw_files[path])
            if (
                not matches_state(projection)
                or set(projection) != account_fields
                or projection.get("account_key") != account
            ):
                raise ValueError()
            entries = projection.get("attempts")
            if not isinstance(entries, list) or not 1 <= len(entries) <= 3:
                raise ValueError()
            if [
                entry.get("attempt_number")
                for entry in entries
                if isinstance(entry, Mapping)
            ] != list(range(1, len(entries) + 1)):
                raise ValueError()

            local_references = {}
            records = []
            for expected_number, entry in enumerate(entries, start=1):
                if (
                    not matches_state(entry)
                    or set(entry) != reference_fields
                    or type(entry.get("attempt_number")) is not int
                    or entry.get("attempt_number") != expected_number
                    or not _uuid4(entry.get("submission_id"))
                    or _SHA256.fullmatch(str(entry.get("record_sha256", ""))) is None
                ):
                    raise ValueError()
                submission = entry["submission_id"]
                key = (account, submission)
                if key in local_references:
                    raise ValueError()
                loaded = attempt_files.get(key)
                if loaded is None or loaded["record_sha256"] != entry["record_sha256"]:
                    raise ValueError()
                record = loaded["record"]
                if (
                    not isinstance(record, Mapping)
                    or record.get("account_key") != account
                    or record.get("submission_id") != submission
                    or type(record.get("attempt_number")) is not int
                    or record.get("attempt_number") != expected_number
                ):
                    raise ValueError()
                local_references[key] = dict(entry)
                records.append(record)

            account_attempt_keys = {key for key in attempt_files if key[0] == account}
            if account_attempt_keys != set(local_references):
                raise ValueError()
            best = select_best_attempt(records)
            if not isinstance(best, Mapping) or projection.get(
                "best_submission_id"
            ) != best.get("submission_id"):
                raise ValueError()

            references.update(local_references)
            account_records[account] = tuple(records)
            account_best[account] = best
        except Exception:
            issues.add("account_projection_invalid")

    # Every immutable attempt file must appear once in its account projection,
    # and every projection reference must resolve to an immutable file.  This
    # equality also catches accounts with attempts but no projection file.
    if set(attempt_files) != set(references):
        issues.add("account_projection_invalid")

    try:
        organizer = _decode_json(raw_files[ORGANIZER_PROJECTION_PATH])
        if not matches_state(organizer) or set(organizer) != state_fields | {
            "accounts"
        }:
            raise ValueError()
        accounts = organizer.get("accounts")
        if not isinstance(accounts, list) or len(accounts) != len(account_records):
            raise ValueError()
        organizer_accounts = set()
        for entry in accounts:
            if (
                not matches_state(entry)
                or set(entry) != organizer_row_fields
                or _SHA256.fullmatch(str(entry.get("account_key", ""))) is None
            ):
                raise ValueError()
            account = entry["account_key"]
            if account in organizer_accounts or account not in account_records:
                raise ValueError()
            organizer_accounts.add(account)
            best = account_best[account]
            expected = {
                "attempt_count": len(account_records[account]),
                "best_submission_id": best.get("submission_id"),
                "hf_subject": best.get("hf_subject"),
                "hf_username": best.get("hf_username"),
                "verified_email": best.get("verified_email"),
                "team": best.get("team"),
                "participant_names": best.get("participant_names"),
                "submission_name": best.get("submission_name"),
                "submitted_at": best.get("submitted_at"),
                "attempt_number": best.get("attempt_number"),
                "metrics": best.get("metrics"),
            }
            if (
                type(entry.get("attempt_count")) is not int
                or type(entry.get("attempt_number")) is not int
                or any(entry.get(name) != value for name, value in expected.items())
            ):
                raise ValueError()
        if organizer_accounts != set(account_records):
            raise ValueError()
    except Exception:
        issues.add("organizer_projection_mismatch")
    return references, issues


def _select_snapshot_paths(paths):
    if (
        not isinstance(paths, list)
        or len(paths) > MAX_FILES
        or len(set(paths)) != len(paths)
    ):
        raise ValueError()
    selected = set()
    governed_prefixes = (
        "attempts/test/",
        "projections/test/accounts/",
        "exclusions/test/",
        "adjudications/test/",
    )
    exact = {
        RELEASE_PATH,
        GOLD_PATH,
        PUBLIC_FINAL_PATH,
        AUDIT_PATH,
        ORGANIZER_PROJECTION_PATH,
    }
    for path in paths:
        if not _safe_path(path):
            raise ValueError()
        if path in exact:
            selected.add(path)
        elif any(
            pattern.fullmatch(path)
            for pattern in (
                _ATTEMPT_PATH,
                _ACCOUNT_PROJECTION_PATH,
                _EXCLUSION_PATH,
                _ADJUDICATION_PATH,
            )
        ):
            selected.add(path)
        elif path.startswith(governed_prefixes):
            raise ValueError()
    return selected


def _require_current_private_head(api, repo_id, expected, token):
    try:
        info = api.repo_info(repo_id, repo_type="dataset", revision="main", token=token)
        if (
            getattr(info, "sha", None) != expected
            or getattr(info, "private", None) is not True
        ):
            raise ValueError()
    except Exception:
        raise FinalizationError(
            "The private repository moved or is not private."
        ) from None


def _download_exact(api, repo_id, revision, token, path):
    try:
        with tempfile.TemporaryDirectory(
            prefix="docsem-finalize-recheck-"
        ) as cache_name:
            cache = Path(cache_name)
            cache.chmod(0o700)
            local = api.hf_hub_download(
                repo_id,
                path,
                repo_type="dataset",
                revision=revision,
                token=token,
                cache_dir=str(cache),
            )
            return _bounded_regular_file(Path(local), MAX_LEDGER_FILE_BYTES)
    except Exception:
        raise FinalizationError("The private release could not be rechecked.") from None


def _bounded_regular_file(path, maximum):
    if not path.is_file() or path.stat().st_size > maximum:
        raise ValueError()
    payload = path.read_bytes()
    if len(payload) > maximum:
        raise ValueError()
    return payload


def _git_scorer_bytes(revision):
    """Read the scorer exactly as committed at one local Git revision."""

    result = subprocess.run(
        [
            "git",
            "show",
            f"{revision}:competition/hf-space/scoring.py",
        ],
        cwd=REPOSITORY_ROOT,
        env={
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        },
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
    )
    if (
        result.returncode != 0
        or not result.stdout
        or len(result.stdout) > 2 * 1024 * 1024
        or len(result.stderr) > 64 * 1024
    ):
        raise ValueError()
    return result.stdout


def _decode_json(payload):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError()
            value[key] = item
        return value

    def invalid_constant(_):
        raise ValueError()

    value = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=pairs,
        parse_constant=invalid_constant,
    )
    if not isinstance(value, dict):
        raise ValueError()
    return value


def _decode_jsonl(payload):
    rows = []
    for line in payload.decode("utf-8").splitlines():
        if not line.strip():
            raise ValueError()
        rows.append(_decode_json(line.encode()))
    if not rows:
        raise ValueError()
    return rows


def _canonical_json(value):
    try:
        payload = (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError):
        raise FinalizationError("A finalization artifact is malformed.") from None
    return payload


def _bounded_json(value):
    payload = _canonical_json(value)
    if len(payload) > MAX_LEDGER_FILE_BYTES:
        raise FinalizationError("A finalization artifact exceeds its size limit.")
    return payload


def _safe_path(path):
    return (
        isinstance(path, str)
        and 0 < len(path) <= 512
        and not path.startswith("/")
        and "\\" not in path
        and all(part not in ("", ".", "..") for part in path.split("/"))
    )


def _path_kind(path):
    if _ATTEMPT_PATH.fullmatch(path):
        return "attempt"
    if _EXCLUSION_PATH.fullmatch(path):
        return "exclusion"
    if _ADJUDICATION_PATH.fullmatch(path):
        return "adjudication"
    return "other"


def _uuid4(value):
    try:
        parsed = uuid.UUID(str(value))
    except (TypeError, ValueError, AttributeError):
        return False
    return (
        str(parsed) == value and parsed.version == 4 and parsed.variant == uuid.RFC_4122
    )


def _parse_utc(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 64:
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        return None
    return parsed.astimezone(dt.timezone.utc)


def _require_utc(value):
    if (
        not isinstance(value, dt.datetime)
        or value.tzinfo is None
        or value.utcoffset() != dt.timedelta(0)
    ):
        raise FinalizationError("Finalization requires a timezone-aware UTC time.")
    return value.astimezone(dt.timezone.utc)


def _format_utc(value):
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--scorer-revision", required=True)
    parser.add_argument("--scorer-sha256", required=True)
    parser.add_argument("--token-env", default="ORGANIZER_WRITE_TOKEN")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--maintenance-confirmed", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    token = os.environ.get(args.token_env, "")
    snapshot = load_finalization_snapshot(
        args.repo_id,
        args.revision,
        token,
        scorer_revision=args.scorer_revision,
        scorer_code_sha256=args.scorer_sha256,
    )
    now = dt.datetime.now(dt.timezone.utc)
    plan = build_finalization(snapshot, now)
    print(json.dumps(plan.dry_run_summary(), sort_keys=True))
    if args.yes or args.maintenance_confirmed:
        api = HfApi(token=token)
        revision = commit_finalization(
            api,
            snapshot,
            plan,
            token=token,
            expected_private_sha=args.revision,
            now=dt.datetime.now(dt.timezone.utc),
            yes=args.yes,
            maintenance_confirmed=args.maintenance_confirmed,
        )
        if revision is not None:
            print(json.dumps({"committed_revision": revision}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
