"""Pure identity, release-window, and participant-feedback policy helpers."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
from dataclasses import dataclass
from collections.abc import Mapping

from scoring import normalize_answer


class TestPolicyError(ValueError):
    """Raised when a test submission violates a policy invariant."""


def _is_utc(value: dt.datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() == dt.timedelta(0)


@dataclass(frozen=True)
class OAuthIdentity:
    """The server-injected Hugging Face identity used for test quota accounting."""

    sub: str
    username: str
    email: str

    @classmethod
    def from_profile(cls, profile):
        data = dict(profile or {})
        sub = str(data.get("sub") or "").strip()
        username = str(data.get("preferred_username") or "").strip()
        email = str(data.get("email") or "").strip().casefold()
        verified = data.get("email_verified")
        if verified is not None and str(verified).strip().casefold() in {
            "false",
            "0",
            "no",
        }:
            email = ""
        if not sub or not username or not email:
            raise TestPolicyError("Test submission requires a verified email and HF identity.")
        return cls(sub=sub, username=username, email=email)


def account_key(identity: OAuthIdentity) -> str:
    """Return the stable repository path key derived only from OAuth ``sub``."""

    if not isinstance(identity, OAuthIdentity) or not identity.sub:
        raise TestPolicyError("A valid HF OAuth identity is required for test submissions.")
    return hashlib.sha256(identity.sub.encode("utf-8")).hexdigest()


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
        if not isinstance(self.max_attempts, int) or isinstance(self.max_attempts, bool):
            raise TestPolicyError("Test max_attempts must be an integer.")
        if self.max_attempts != 3:
            raise TestPolicyError("Test max_attempts must be exactly 3.")
        for name in ("open_at", "close_at"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, dt.datetime) or not _is_utc(value)):
                raise TestPolicyError(f"Test release {name} must be a timezone-aware UTC datetime.")
        if self.open_at is not None and self.close_at is not None and self.close_at <= self.open_at:
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
            raise TestPolicyError("Test release checks require a timezone-aware UTC datetime.")
        if current < self.open_at or current >= self.close_at:
            raise TestPolicyError("Test submissions are not open.")
        return True


def _canonical_value(value):
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(value[key]) for key in sorted(value, key=str)}
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
    """Hash normalized predictions together with split, release, and OAuth subject."""

    if not isinstance(split, str) or not split.strip():
        raise TestPolicyError("Submission split is required.")
    if not isinstance(release_id, str) or not release_id.strip():
        raise TestPolicyError("Test release ID is required.")
    if not isinstance(identity, OAuthIdentity) or not identity.sub:
        raise TestPolicyError("A valid HF OAuth identity is required for test submissions.")
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


def _accepted_timestamp(attempt: Mapping) -> str:
    value = attempt.get("accepted_at", attempt.get("submitted_at", attempt.get("timestamp", "")))
    if isinstance(value, dt.datetime):
        return value.isoformat()
    return str(value)


def select_best_attempt(attempts):
    """Select an account's best attempt using the documented deterministic order."""

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
            answer_accuracy = round(float(metrics["answer_accuracy"]), 6)
            evidence_f1 = round(float(metrics["evidence_f1"]), 6)
        except (KeyError, TypeError, ValueError) as exc:
            raise TestPolicyError("Aggregate test metrics are incomplete.") from exc
        return {
            "accepted": True,
            "attempt": attempt,
            "receipt": receipt,
            "answer_accuracy": answer_accuracy,
            "evidence_f1": evidence_f1,
        }
    if attempt in (2, 3):
        return {"accepted": True, "attempt": attempt, "receipt": receipt, "score": "withheld"}
    raise TestPolicyError("Test attempt limit exceeded.")
