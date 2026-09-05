"""Shared structural limits for private DocSem test-ledger records."""

from __future__ import annotations

import re
from collections.abc import Mapping


class TestContractError(ValueError):
    """Raised when participant-controlled test data violates the ledger contract."""


TEST_PREDICTION_KEYS = frozenset({"instance_id", "answer", "evidence"})
MAX_TEST_ROWS = 10_000
MAX_INSTANCE_ID_CHARACTERS = 256
MAX_ANSWER_CHARACTERS = 4_096
MAX_EVIDENCE_IDS = 128
MAX_EVIDENCE_ID_CHARACTERS = 256
MAX_PRIVATE_TEXT_CHARACTERS = 4_096
MAX_PARTICIPANT_NAMES_CHARACTERS = 500
MAX_REPOSITORY_ID_CHARACTERS = 256

PRIVATE_TEXT_LIMITS = {
    "release_id": MAX_PRIVATE_TEXT_CHARACTERS,
    "hf_subject": MAX_PRIVATE_TEXT_CHARACTERS,
    "hf_username": MAX_PRIVATE_TEXT_CHARACTERS,
    "verified_email": MAX_PRIVATE_TEXT_CHARACTERS,
    "team": MAX_PRIVATE_TEXT_CHARACTERS,
    "participant_names": MAX_PARTICIPANT_NAMES_CHARACTERS,
    "submission_name": MAX_PRIVATE_TEXT_CHARACTERS,
}

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_REPOSITORY_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def bounded_private_text(value, field: str) -> str:
    """Return trimmed private text after applying the field's persisted limit."""

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
