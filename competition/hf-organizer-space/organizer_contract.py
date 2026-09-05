"""Package-local copy of the DocSem immutable test-ledger contract.

The organizer Space is deployed independently from the participant Space, so
its reader cannot import sibling source files.  Repository parity tests compare
this deliberately small copy with the participant producer's contract.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass


class TestContractError(ValueError):
    """Raised when participant-controlled data violates the ledger contract."""


class TestPolicyError(ValueError):
    """Raised when immutable submission or ranking policy is invalid."""


TEST_PREDICTION_KEYS = frozenset({"instance_id", "answer", "evidence"})
MAX_TEST_ROWS = 10_000
MAX_INSTANCE_ID_CHARACTERS = 256
MAX_ANSWER_CHARACTERS = 4_096
MAX_EVIDENCE_IDS = 128
MAX_EVIDENCE_ID_CHARACTERS = 256
MAX_PRIVATE_TEXT_CHARACTERS = 4_096
MAX_PARTICIPANT_NAMES_CHARACTERS = 500
MAX_REPOSITORY_ID_CHARACTERS = 256
MAX_LEDGER_FILE_BYTES = 16 * 1024 * 1024

PRIVATE_TEXT_LIMITS = {
    "release_id": MAX_PRIVATE_TEXT_CHARACTERS,
    "hf_subject": MAX_PRIVATE_TEXT_CHARACTERS,
    "hf_username": MAX_PRIVATE_TEXT_CHARACTERS,
    "verified_email": MAX_PRIVATE_TEXT_CHARACTERS,
    "team": MAX_PRIVATE_TEXT_CHARACTERS,
    "participant_names": MAX_PARTICIPANT_NAMES_CHARACTERS,
    "submission_name": MAX_PRIVATE_TEXT_CHARACTERS,
}

_FINAL_MARKER = re.compile(r"^\s*(final\s*answer\s*:|answer\s*:)\s*", re.IGNORECASE)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_REPOSITORY_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def normalize_answer(value):
    """Apply the exact answer normalization used by the participant scorer."""

    text = _FINAL_MARKER.sub("", str(value)).strip().lower()
    return re.sub(r"\s+", " ", text)


def bounded_private_text(value, field: str) -> str:
    """Return trimmed private text after applying its persisted field limit."""

    limit = PRIVATE_TEXT_LIMITS.get(field)
    if (
        limit is None
        or not isinstance(value, str)
        or not value.strip()
        or len(value) > limit
    ):
        raise TestContractError("Private test metadata is invalid.")
    return value.strip()


def is_bounded_private_text(value, field: str) -> bool:
    try:
        bounded_private_text(value, field)
    except TestContractError:
        return False
    return True


def sha256_digest(value) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TestContractError("Private test provenance is invalid.")
    return value


def revision_digest(value) -> str:
    if not isinstance(value, str) or _REVISION.fullmatch(value) is None:
        raise TestContractError("Private test provenance is invalid.")
    return value


def repository_id(value) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > MAX_REPOSITORY_ID_CHARACTERS
    ):
        raise TestContractError("Private test provenance is invalid.")
    normalized = value.strip()
    parts = normalized.split("/")
    if len(parts) != 2 or any(
        _REPOSITORY_PART.fullmatch(part) is None for part in parts
    ):
        raise TestContractError("Private test provenance is invalid.")
    return normalized


def validate_test_predictions(rows) -> None:
    """Require the exact immutable prediction schema and bounded values."""

    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_TEST_ROWS:
        raise TestContractError("Test predictions are invalid.")
    identifiers = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != TEST_PREDICTION_KEYS:
            raise TestContractError("Test predictions are invalid.")
        identifier = row.get("instance_id")
        answer = row.get("answer")
        evidence = row.get("evidence")
        if (
            not isinstance(identifier, str)
            or not identifier.strip()
            or len(identifier) > MAX_INSTANCE_ID_CHARACTERS
            or identifier in identifiers
            or not isinstance(answer, str)
            or len(answer) > MAX_ANSWER_CHARACTERS
            or not isinstance(evidence, list)
            or not 1 <= len(evidence) <= MAX_EVIDENCE_IDS
            or any(
                not isinstance(item, str)
                or not item.strip()
                or len(item) > MAX_EVIDENCE_ID_CHARACTERS
                for item in evidence
            )
        ):
            raise TestContractError("Test predictions are invalid.")
        identifiers.add(identifier)


@dataclass(frozen=True)
class OAuthIdentity:
    """Stored Hugging Face identity envelope used for submission hashing."""

    sub: str
    username: str
    email: str


def _canonical_value(value):
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(value[key]) for key in sorted(value, key=str)
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value.strip())
    return value


def _canonical_predictions(predictions):
    if not isinstance(predictions, (list, tuple)):
        raise TestPolicyError("Predictions must be a sequence of parsed rows.")
    rows = []
    for row in predictions:
        if not isinstance(row, Mapping):
            raise TestPolicyError("Predictions must contain parsed row objects.")
        normalized = _canonical_value(row)
        if "instance_id" in normalized:
            normalized["instance_id"] = str(normalized["instance_id"]).strip()
        if "answer" in normalized:
            normalized["answer"] = normalize_answer(normalized["answer"])
        if isinstance(normalized.get("evidence"), list):
            normalized["evidence"] = sorted(
                {str(value).strip().casefold() for value in normalized["evidence"]}
            )
        rows.append(normalized)
    if all("instance_id" in row for row in rows):
        rows.sort(key=lambda row: row["instance_id"])
    return rows


def canonical_submission_hash(
    predictions,
    split: str,
    release_id: str,
    identity: OAuthIdentity,
) -> str:
    """Hash normalized predictions with split, release, and OAuth subject."""

    if not isinstance(split, str) or not split.strip():
        raise TestPolicyError("Submission split is required.")
    if not isinstance(release_id, str) or not release_id.strip():
        raise TestPolicyError("Test release ID is required.")
    if not isinstance(identity, OAuthIdentity) or not identity.sub:
        raise TestPolicyError(
            "A valid HF OAuth identity is required for test submissions."
        )
    envelope = {
        "oauth_sub": identity.sub,
        "payload": _canonical_predictions(predictions),
        "release_id": release_id.strip(),
        "split": split.strip().casefold(),
    }
    serialized = json.dumps(
        envelope,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _metric(attempt: Mapping, name: str) -> float:
    metrics = attempt.get("metrics")
    source = metrics if isinstance(metrics, Mapping) else attempt
    try:
        value = float(source.get(name, 0.0))
    except (TypeError, ValueError) as exc:
        raise TestPolicyError(f"Attempt metric {name} is invalid.") from exc
    if not math.isfinite(value):
        raise TestPolicyError(f"Attempt metric {name} is invalid.")
    return value


def _accepted_timestamp(attempt: Mapping) -> dt.datetime:
    value = None
    for field in ("accepted_at", "submitted_at", "timestamp"):
        if field in attempt:
            value = attempt[field]
            break
    if isinstance(value, dt.datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        try:
            parsed = dt.datetime.fromisoformat(text)
        except ValueError as exc:
            raise TestPolicyError("Accepted attempt timestamp is malformed.") from exc
    else:
        raise TestPolicyError("Accepted attempt timestamp is required.")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise TestPolicyError("Accepted attempt timestamp must include a UTC offset.")
    return parsed.astimezone(dt.timezone.utc)


def select_best_attempt(attempts):
    """Select an account's best attempt using the participant ranking order."""

    if not isinstance(attempts, (list, tuple)) or not attempts:
        raise TestPolicyError("At least one accepted test attempt is required.")
    if any(not isinstance(attempt, Mapping) for attempt in attempts):
        raise TestPolicyError("Accepted attempts must be objects.")
    return min(
        attempts,
        key=lambda attempt: (
            -_metric(attempt, "answer_accuracy"),
            -_metric(attempt, "evidence_f1"),
            _accepted_timestamp(attempt),
            str(attempt.get("submission_id", "")),
        ),
    )
