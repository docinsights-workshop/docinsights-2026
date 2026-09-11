"""Pure identity, release-window, and participant-feedback policy helpers."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from collections.abc import Mapping

from scoring import normalize_answer
from test_contract import bounded_private_text


class TestPolicyError(ValueError):
    """Raised when a test submission violates a policy invariant."""


def _configured_test_close_at():
    """Use the operator's UTC deadline, also verified against the private release."""
    default = dt.datetime(2026, 9, 11, 12, 0, tzinfo=dt.timezone.utc)
    value = os.getenv("TEST_CLOSE_AT")
    if not isinstance(value, str) or re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z", value
    ) is None:
        return default
    try:
        return dt.datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError:
        return default


OFFICIAL_TEST_CLOSE_AT = _configured_test_close_at()
TEST_ATTEMPT_COOLDOWN_SECONDS = 21_600
_EMAIL_LOCAL = re.compile(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]{1,64}\Z")
_EMAIL_DOMAIN_LABEL = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\Z")


def _is_utc(value: dt.datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() == dt.timedelta(0)


def normalize_contact_email(value) -> str:
    """Return the conservative normalized ASCII email used for anonymous quota."""

    if not isinstance(value, str):
        raise TestPolicyError("Enter a valid contact email for test submissions.")
    normalized = value.strip().casefold()
    if (
        len(normalized) > 320
        or normalized.count("@") != 1
        or any(ord(character) < 32 or ord(character) == 127 for character in normalized)
    ):
        raise TestPolicyError("Enter a valid contact email for test submissions.")
    local, domain = normalized.split("@")
    labels = domain.split(".")
    if (
        _EMAIL_LOCAL.fullmatch(local) is None
        or local.startswith(".")
        or local.endswith(".")
        or ".." in local
        or len(domain) > 253
        or len(labels) < 2
        or any(_EMAIL_DOMAIN_LABEL.fullmatch(label) is None for label in labels)
    ):
        raise TestPolicyError("Enter a valid contact email for test submissions.")
    return normalized


@dataclass(frozen=True)
class TestIdentity:
    """Server-derived Hugging Face or normalized-email test quota identity."""

    identity_kind: str
    identity_subject: str
    hf_username: str
    contact_email: str
    email_verified: bool

    def __post_init__(self):
        try:
            subject = bounded_private_text(self.identity_subject, "identity_subject")
            username = bounded_private_text(self.hf_username, "hf_username")
            contact = normalize_contact_email(self.contact_email)
        except (ValueError, TestPolicyError):
            raise TestPolicyError("Test submission identity is invalid.") from None
        valid = (
            self.identity_kind == "huggingface"
            and self.email_verified is True
            and subject == self.identity_subject
            and username == self.hf_username
            and contact == self.contact_email
        )
        if not valid:
            raise TestPolicyError("Test submission identity is invalid.")

    @classmethod
    def from_profile(cls, profile):
        data = dict(profile or {})
        verified = data.get("email_verified")
        if verified is not True:
            raise TestPolicyError(
                "Test submission requires a verified email and HF identity."
            )
        try:
            subject = bounded_private_text(data.get("sub"), "identity_subject")
            username = bounded_private_text(
                data.get("preferred_username"), "hf_username"
            )
            email = normalize_contact_email(data.get("email"))
        except (ValueError, TestPolicyError):
            raise TestPolicyError(
                "Test submission requires a verified email and HF identity."
            ) from None
        return cls(
            identity_kind="huggingface",
            identity_subject=subject,
            hf_username=username,
            contact_email=email,
            email_verified=True,
        )


def account_key(identity: TestIdentity) -> str:
    """Return the kind-scoped stable repository key for a test identity."""

    if not isinstance(identity, TestIdentity):
        raise TestPolicyError("A valid identity is required for test submissions.")
    envelope = f"{identity.identity_kind}\0{identity.identity_subject}"
    return hashlib.sha256(envelope.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TestReleasePolicy:
    """Pinned test-release configuration and its server-side open-window check."""

    release_id: str | None = None
    task_manifest_sha256: str | None = None
    gold_sha256: str | None = None
    open_at: dt.datetime | None = None
    close_at: dt.datetime | None = None
    enabled: bool = True
    max_attempts: int = 3

    def __post_init__(self):
        if not isinstance(self.max_attempts, int) or isinstance(
            self.max_attempts, bool
        ):
            raise TestPolicyError("Test max_attempts must be an integer.")
        if self.max_attempts != 3:
            raise TestPolicyError("Test max_attempts must be exactly 3.")
        for name in ("open_at", "close_at"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, dt.datetime) or not _is_utc(value)
            ):
                raise TestPolicyError(
                    f"Test release {name} must be a timezone-aware UTC datetime."
                )
        if (
            self.open_at is not None
            and self.close_at is not None
            and self.close_at <= self.open_at
        ):
            raise TestPolicyError("Test release close_at must be after open_at.")
        if self.enabled and any(
            not isinstance(value, str) or not value.strip()
            for value in (self.release_id, self.task_manifest_sha256, self.gold_sha256)
        ):
            raise TestPolicyError("Test release configuration is incomplete.")
        if self.enabled and (self.open_at is None or self.close_at is None):
            raise TestPolicyError("Test release configuration is incomplete.")

    @classmethod
    def disabled(cls) -> "TestReleasePolicy":
        return cls(enabled=False)

    @property
    def task_digest(self) -> str | None:
        return self.task_manifest_sha256

    @property
    def gold_digest(self) -> str | None:
        return self.gold_sha256

    def require_open(self, now: dt.datetime | None = None) -> bool:
        """Fail closed unless the enabled release is currently inside its UTC window."""

        if not self.enabled:
            raise TestPolicyError("Test submissions are not open.")
        if not self.release_id or not self.task_manifest_sha256 or not self.gold_sha256:
            raise TestPolicyError("Test release configuration is incomplete.")
        if self.open_at is None or self.close_at is None:
            raise TestPolicyError("Test release configuration is incomplete.")
        current = dt.datetime.now(dt.timezone.utc) if now is None else now
        if not isinstance(current, dt.datetime) or not _is_utc(current):
            raise TestPolicyError(
                "Test release checks require a timezone-aware UTC datetime."
            )
        if current < self.open_at or current >= self.close_at:
            raise TestPolicyError("Test submissions are not open.")
        return True


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
        if "answer" in normalized and normalized["answer"] is not None:
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
    identity: TestIdentity,
    metadata: Mapping,
) -> str:
    """Hash identity, predictions, and immutable participant metadata."""

    if not isinstance(split, str) or not split.strip():
        raise TestPolicyError("Submission split is required.")
    if not isinstance(release_id, str) or not release_id.strip():
        raise TestPolicyError("Test release ID is required.")
    if not isinstance(identity, TestIdentity):
        raise TestPolicyError("A valid identity is required for test submissions.")
    if not isinstance(metadata, Mapping):
        raise TestPolicyError("Immutable submission metadata is required.")
    try:
        immutable_metadata = {
            field: bounded_private_text(metadata.get(field), field)
            for field in ("team", "participant_names", "submission_name")
        }
    except ValueError:
        raise TestPolicyError("Immutable submission metadata is required.") from None
    envelope = {
        "identity_kind": identity.identity_kind,
        "identity_subject": identity.identity_subject,
        "metadata": _canonical_value(immutable_metadata),
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
    """Parse an accepted timestamp and compare it as a UTC instant."""

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


def next_eligible_at(attempts) -> str | None:
    """Return the next distinct-attempt UTC instant for an account history."""

    if not attempts:
        return None
    if not isinstance(attempts, (list, tuple)) or any(
        not isinstance(attempt, Mapping) for attempt in attempts
    ):
        raise TestPolicyError("Accepted attempts must be objects.")
    latest = max(_accepted_timestamp(attempt) for attempt in attempts)
    eligible = latest + dt.timedelta(seconds=TEST_ATTEMPT_COOLDOWN_SECONDS)
    return eligible.isoformat().replace("+00:00", "Z")


def rank_attempts(attempts):
    """Return accepted attempts in the documented deterministic ranking order."""

    if not isinstance(attempts, (list, tuple)) or not attempts:
        raise TestPolicyError("At least one accepted test attempt is required.")
    if any(not isinstance(attempt, Mapping) for attempt in attempts):
        raise TestPolicyError("Accepted attempts must be objects.")
    return sorted(
        attempts,
        key=lambda attempt: (
            -_metric(attempt, "joint_accuracy"),
            -_metric(attempt, "answer_accuracy"),
            -_metric(attempt, "evidence_f1"),
            _accepted_timestamp(attempt),
            str(attempt.get("submission_id", "")),
        ),
    )


def select_best_attempt(attempts):
    """Select an account's best attempt using the documented deterministic order."""

    return rank_attempts(attempts)[0]


def participant_test_response(attempt: int, metrics: Mapping, receipt: str) -> dict:
    """Build the participant-safe response for one accepted test attempt."""

    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
        raise TestPolicyError("Test attempt number must be positive.")
    if not isinstance(metrics, Mapping):
        raise TestPolicyError("Aggregate test metrics are required.")
    if not isinstance(receipt, str) or not receipt:
        raise TestPolicyError("A test receipt is required.")
    if attempt == 1:
        try:
            joint_accuracy = round(float(metrics["joint_accuracy"]), 6)
            answer_accuracy = round(float(metrics["answer_accuracy"]), 6)
            evidence_f1 = round(float(metrics["evidence_f1"]), 6)
        except (KeyError, TypeError, ValueError) as exc:
            raise TestPolicyError("Aggregate test metrics are incomplete.") from exc
        return {
            "accepted": True,
            "attempt": attempt,
            "receipt": receipt,
            "joint_accuracy": joint_accuracy,
            "answer_accuracy": answer_accuracy,
            "evidence_f1": evidence_f1,
        }
    if attempt in (2, 3):
        return {
            "accepted": True,
            "attempt": attempt,
            "receipt": receipt,
            "score": "withheld",
        }
    raise TestPolicyError("Test attempt limit exceeded.")
