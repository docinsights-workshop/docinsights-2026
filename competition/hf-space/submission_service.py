"""Split-aware submission boundary for validation and held-out test scoring."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from huggingface_hub.errors import EntryNotFoundError

from scoring import (
    SubmissionError,
    load_jsonl_text,
    normalize_participant_names,
    parse_submission_text,
    score_predictions,
)
from test_policy import (
    OAuthIdentity,
    TestPolicyError,
    TestReleasePolicy,
    participant_test_response,
)


RELEASE_PATH = "private/test_release.json"
GOLD_PATH = "private/test_labels.jsonl"
TEST_UNAVAILABLE = "Test submission is temporarily unavailable."


class SubmissionSplit(str, Enum):
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True)
class TrustedTestConfig:
    policy: TestReleasePolicy
    labels: list[dict]


class HubTestConfigLoader:
    """Load one pinned private release and its gold labels from server policy."""

    def __init__(self, api, repo_id: str, *, enabled: bool = False):
        self.api = api
        self.repo_id = str(repo_id or "").strip()
        self.enabled = enabled is True

    def __call__(self, now: dt.datetime) -> TrustedTestConfig:
        if not self.enabled or not self.repo_id:
            raise SubmissionError(TEST_UNAVAILABLE)
        try:
            info = self.api.repo_info(
                self.repo_id,
                repo_type="dataset",
                revision="main",
            )
            sha = getattr(info, "sha", None)
            if not isinstance(sha, str) or not sha:
                raise ValueError()
            release_raw = self._read(RELEASE_PATH, sha)
            gold_raw = self._read(GOLD_PATH, sha)
            release = json.loads(release_raw.decode("utf-8"))
            if not isinstance(release, Mapping):
                raise ValueError()
            task_digest = _sha256_digest(release.get("task_manifest_sha256"))
            gold_digest = _sha256_digest(release.get("gold_sha256"))
            policy = TestReleasePolicy(
                release_id=release.get("release_id"),
                task_manifest_sha256=task_digest,
                gold_sha256=gold_digest,
                open_at=_parse_utc(release.get("open_at")),
                close_at=_parse_utc(release.get("close_at")),
                enabled=release.get("enabled", True),
                max_attempts=release.get("max_attempts", 3),
            )
            policy.require_open(now)
            if hashlib.sha256(gold_raw).hexdigest() != policy.gold_sha256:
                raise ValueError()
            labels = load_jsonl_text(gold_raw.decode("utf-8"))
            return TrustedTestConfig(policy=policy, labels=labels)
        except Exception:
            raise SubmissionError(TEST_UNAVAILABLE) from None

    def _read(self, path: str, sha: str) -> bytes:
        try:
            local_path = self.api.hf_hub_download(
                self.repo_id,
                path,
                repo_type="dataset",
                revision=sha,
            )
        except EntryNotFoundError:
            raise ValueError() from None
        return Path(local_path).read_bytes()


class SubmissionService:
    """Route anonymous validation and OAuth-bound test requests explicitly."""

    def __init__(
        self,
        *,
        validation_submitter: Callable,
        test_store,
        test_config_loader: Callable,
        now_provider: Callable[[], dt.datetime] | None = None,
    ):
        self.validation_submitter = validation_submitter
        self.test_store = test_store
        self.test_config_loader = test_config_loader
        self.now_provider = now_provider or (lambda: dt.datetime.now(dt.timezone.utc))

    def submit_for_split(self, split, file_obj, metadata, oauth_profile) -> dict:
        selected = _split(split)
        if selected is SubmissionSplit.VALIDATION:
            return self.validation_submitter(file_obj, metadata)
        return self._submit_test(file_obj, metadata, oauth_profile)

    def _submit_test(self, file_obj, metadata, oauth_profile) -> dict:
        identity = _oauth_identity(oauth_profile)
        try:
            now = self.now_provider()
            config = self.test_config_loader(now)
            if not isinstance(config, TrustedTestConfig):
                raise ValueError()
            config.policy.require_open(now)
            server_metadata = _test_metadata(metadata, config.policy)
        except Exception:
            raise SubmissionError(TEST_UNAVAILABLE) from None

        if file_obj is None:
            raise SubmissionError("Upload a JSONL submission file.")
        try:
            text = Path(file_obj.name).read_text(encoding="utf-8")
            predictions = parse_submission_text(text)
            metrics = score_predictions(predictions, config.labels)
        except Exception:
            raise SubmissionError("Test submission could not be accepted.") from None

        try:
            receipt = self.test_store.submit(
                identity,
                server_metadata,
                predictions,
                metrics,
                now,
            )
            if not receipt.accepted:
                return {
                    "accepted": False,
                    "score": "withheld",
                    "message": "Test attempt limit reached.",
                }
            response = participant_test_response(
                receipt.attempt,
                metrics,
                receipt.submission_id,
            )
            response["accepted_at"] = receipt.accepted_at
            return response
        except Exception:
            raise SubmissionError(TEST_UNAVAILABLE) from None

    def history_for_oauth(self, oauth_profile) -> list[dict]:
        identity = _oauth_identity(oauth_profile)
        try:
            attempts = self.test_store.account_history(identity)
            return [_history_response(attempt) for attempt in attempts]
        except Exception:
            raise SubmissionError("Test submission history is temporarily unavailable.") from None


def _split(value) -> SubmissionSplit:
    try:
        return SubmissionSplit(value)
    except (TypeError, ValueError):
        raise SubmissionError("Select validation or test.") from None


def _oauth_identity(profile) -> OAuthIdentity:
    try:
        return OAuthIdentity.from_profile(profile)
    except (TestPolicyError, TypeError, ValueError):
        raise SubmissionError("Sign in with Hugging Face to submit test predictions.") from None


def _test_metadata(metadata, policy: TestReleasePolicy) -> dict:
    if not isinstance(metadata, Mapping):
        raise ValueError()
    team = metadata.get("team")
    participant_names = metadata.get("participant_names")
    submission_name = metadata.get("submission_name")
    if any(not isinstance(value, str) or not value.strip() for value in (team, submission_name)):
        raise ValueError()
    return {
        "release_id": policy.release_id,
        "task_manifest_sha256": policy.task_manifest_sha256,
        "team": team.strip(),
        "participant_names": normalize_participant_names(participant_names),
        "submission_name": submission_name.strip(),
    }


def _history_response(attempt) -> dict:
    if not isinstance(attempt, Mapping):
        raise ValueError()
    attempt_number = attempt.get("attempt_number")
    receipt = attempt.get("submission_id")
    response = participant_test_response(attempt_number, attempt.get("metrics"), receipt)
    response["submission_name"] = str(attempt.get("submission_name") or "")
    response["accepted_at"] = str(attempt.get("submitted_at") or "")
    return response


def _parse_utc(value) -> dt.datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError()
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    parsed = dt.datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise ValueError()
    return parsed


def _sha256_digest(value) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError()
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError()
    return value
