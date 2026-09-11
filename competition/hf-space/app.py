import datetime as dt
import hashlib
import html
import json
import math
import os
import re
import threading
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
from huggingface_hub import HfApi, hf_hub_download, upload_file
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

from scoring import (
    SubmissionError,
    expand_predictions,
    leaderboard_identity,
    leaderboard_row,
    load_jsonl_text,
    normalize_participant_names,
    parse_submission_text,
    rank_leaderboard,
    safe_slug,
    score_validation_predictions,
)
from submission_service import HubTestConfigLoader, SubmissionService, TrustedTestConfig
from test_contract import is_valid_public_text
from test_policy import (
    OFFICIAL_TEST_CLOSE_AT,
    TestPolicyError,
    TestReleasePolicy,
    normalize_contact_email,
)
from test_store import HubTestStore


PUBLIC_DATASET_REPO = os.getenv(
    "PUBLIC_DATASET_REPO", "amitbcp/docinsights-2026-shared-task-data"
)
WORKSHOP_URL = os.getenv(
    "WORKSHOP_URL",
    "https://docinsights-workshop.github.io/docinsights-2026/shared-task/",
)
SOURCE_REPO_URL = os.getenv(
    "SOURCE_REPO_URL", "https://github.com/oracle-samples/gsm-sem"
)
PARTICIPANT_GUIDE_URL = os.getenv(
    "PARTICIPANT_GUIDE_URL",
    "https://github.com/oracle-samples/gsm-sem/blob/main/docsem/PARTICIPANT_INSTRUCTIONS.md",
)
PUBLIC_DATASET_URL = f"https://huggingface.co/datasets/{PUBLIC_DATASET_REPO}"
DATASET_CITATION_URL = "https://arxiv.org/abs/2605.07053"
DATASET_BIBTEX = """@article{singh2026gsmsem,
  title={GSM-SEM: Benchmark and Framework for Generating Semantically Variant Augmentations},
  author={Jyotika Singh and Fang Tu and Aziza Mirsaidova and Amit Agarwal and Hitesh Laxmichand Patel and Sandip Ghoshal and Miguel Ballesteros and Karan Dua and Yassine Benajiba and Weiyi Sun and Tao Sheng and Graham Horwood and Sujith Ravi and Dan Roth},
  year={2026},
  eprint={2605.07053},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2605.07053}
}"""
GOLD_REPO_ID = os.getenv(
    "GOLD_REPO_ID", "amitbcp/docinsights-2026-shared-task-submissions"
)
GOLD_FILE = os.getenv("GOLD_FILE", "private/val_labels.jsonl")
SUBMISSIONS_REPO_ID = os.getenv("SUBMISSIONS_REPO_ID", GOLD_REPO_ID)
WRITE_TOKEN = os.getenv("HF_WRITE_TOKEN") or os.getenv("HF_TOKEN")
GRADIO_MAJOR_VERSION = int(gr.__version__.split(".", maxsplit=1)[0])
LEADERBOARD_LOCK = threading.Lock()
_TRUE_VALUES = {
    "1",
    "true",
    "yes",
}

_RFC3339_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\Z")
_RELEASE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_REPOSITORY_ID = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}/[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z"
)
FINAL_TEST_RELEASE_PATH = "private/test_release.json"
FINAL_TEST_GOLD_PATH = "private/test_labels.jsonl"
PROVISIONAL_TEST_PROJECTION_PATH = "projections/test/public_provisional.json"
FINAL_TEST_PROJECTION_PATH = "projections/test/public_final.json"
FINAL_TEST_AUDIT_PATH = "private/test_finalization_audit.json"
FINAL_TEST_TASK_MANIFEST_PATH = "test/tasks.jsonl"
FINAL_TEST_FEEDBACK_POLICY = "first-attempt-only"
FINAL_TEST_ARTIFACT_MAX_BYTES = 16 * 1024 * 1024
FINAL_TEST_MAX_ROWS = 30_000
FINAL_TEST_PUBLIC_ROW_FIELDS = frozenset(
    {
        "rank",
        "team",
        "submission_name",
        "selected_attempt",
        "total_attempts",
        "joint_accuracy",
    }
)
FINAL_TEST_PROJECTION_FIELDS = frozenset(
    {"schema_version", "split", "release_id", "task_manifest_sha256", "rows"}
)
PROVISIONAL_TEST_PUBLIC_ROW_FIELDS = frozenset({"rank", "hf_username", "team"})
PROVISIONAL_TEST_PROJECTION_FIELDS = FINAL_TEST_PROJECTION_FIELDS
FINAL_TEST_ELIGIBLE_ATTEMPT_FIELDS = frozenset(
    {
        "account_key",
        "submission_id",
        "attempt_number",
        "record_sha256",
        "selected",
        "joint_accuracy",
        "answer_accuracy",
        "evidence_f1",
        "rescored_metrics_sha256",
    }
)

PORTAL_HEAD = """
<script>
(() => {
    const initialized = new WeakSet();
    const closedCopy = "Test submissions are closed.";

    function countdownCopy(totalSeconds) {
        const days = Math.floor(totalSeconds / 86400);
        const hours = Math.floor((totalSeconds % 86400) / 3600);
        const minutes = Math.floor((totalSeconds % 3600) / 60);
        const seconds = totalSeconds % 60;
        const dayLabel = days === 1 ? "day" : "days";
        return `${days} ${dayLabel}, ${hours} hours, ${minutes} minutes, ${seconds} seconds remaining`;
    }

    function bindCountdown(node) {
        if (initialized.has(node)) return;
        initialized.add(node);
        const closeAt = Date.parse(node.dataset.docsemCloseAt || "");
        if (!Number.isFinite(closeAt)) {
            const unavailableCopy = "Official test deadline unavailable.";
            node.textContent = unavailableCopy;
            node.setAttribute("aria-label", unavailableCopy);
            return;
        }
        function setTimerCopy(copy) {
            node.textContent = copy;
            node.setAttribute(
                "aria-label",
                `Time remaining until test submissions close: ${copy}`
            );
        }
        function update() {
            if (!node.isConnected) return;
            const totalSeconds = Math.max(0, Math.ceil((closeAt - Date.now()) / 1000));
            if (totalSeconds === 0) {
                setTimerCopy(closedCopy);
                node.dataset.state = "closed";
                const notice = node.closest(".test-release-notice");
                const status = notice?.querySelector("[data-docsem-submission-status]");
                if (status) status.textContent = closedCopy;
                return;
            }
            setTimerCopy(countdownCopy(totalSeconds));
            window.setTimeout(update, 1000);
        }
        update();
    }

    function bindAllCountdowns() {
        document.querySelectorAll("[data-docsem-close-at]").forEach(bindCountdown);
    }

    document.addEventListener("DOMContentLoaded", bindAllCountdowns);
    new MutationObserver(bindAllCountdowns).observe(document.documentElement, {
        childList: true,
        subtree: true,
    });
})();
</script>
"""


class FinalLeaderboardError(RuntimeError):
    """Sanitized refusal when a public final projection cannot be proven."""


@dataclass(frozen=True)
class TestDeploymentConfig:
    """Operator configuration that can only activate a complete pinned release."""

    submissions_enabled: bool
    public_leaderboard_enabled: bool
    release_id: str | None
    task_manifest_sha256: str | None
    gold_sha256: str | None
    open_at: dt.datetime | None
    close_at: dt.datetime | None
    release_config_path: str | None
    gold_config_path: str | None
    provisional_leaderboard_enabled: bool = False
    max_attempts: int = 3
    feedback_policy: str = FINAL_TEST_FEEDBACK_POLICY
    task_manifest_path: str = FINAL_TEST_TASK_MANIFEST_PATH

    @property
    def expected_policy(self) -> TestReleasePolicy | None:
        if not self.submissions_enabled:
            return None
        return TestReleasePolicy(
            release_id=self.release_id,
            task_manifest_sha256=self.task_manifest_sha256,
            gold_sha256=self.gold_sha256,
            open_at=self.open_at,
            close_at=self.close_at,
            enabled=True,
            max_attempts=self.max_attempts,
        )


def _enabled_value(value) -> bool:
    return str(value or "").strip().casefold() in _TRUE_VALUES


def load_validation_submissions_enabled(
    environment: Mapping[str, object] | None = None,
) -> bool:
    """Keep legacy validation open unless operators explicitly set the gate."""

    environment = os.environ if environment is None else environment
    if "VALIDATION_SUBMISSIONS_ENABLED" not in environment:
        return True
    return _enabled_value(environment.get("VALIDATION_SUBMISSIONS_ENABLED"))


def _required_value(environment: Mapping[str, object], name: str) -> str | None:
    value = environment.get(name)
    value = str(value).strip() if value is not None else ""
    return value or None


def _parse_rfc3339_utc(value: str | None) -> dt.datetime | None:
    if not value or not _RFC3339_UTC.fullmatch(value):
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError:
        return None
    return parsed if parsed.utcoffset() == dt.timedelta(0) else None


def _safe_server_path(value: str | None) -> str | None:
    if not value or value.startswith("/"):
        return None
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return None
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", part) for part in parts):
        return None
    return value


def load_test_deployment_config(
    environment: Mapping[str, object] | None = None,
) -> TestDeploymentConfig:
    """Parse test activation controls without ever defaulting into an open state."""

    environment = os.environ if environment is None else environment
    requested_submissions = _enabled_value(environment.get("TEST_SUBMISSIONS_ENABLED"))
    requested_public_leaderboard = _enabled_value(
        environment.get("TEST_PUBLIC_LEADERBOARD_ENABLED")
    )
    requested_provisional_leaderboard = _enabled_value(
        environment.get("TEST_PROVISIONAL_LEADERBOARD_ENABLED")
    )
    release_id = _required_value(environment, "TEST_RELEASE_ID")
    task_manifest_sha256 = _required_value(environment, "TEST_TASK_MANIFEST_SHA256")
    gold_sha256 = _required_value(environment, "TEST_GOLD_SHA256")
    open_at = _parse_rfc3339_utc(_required_value(environment, "TEST_OPEN_AT"))
    close_at = _parse_rfc3339_utc(_required_value(environment, "TEST_CLOSE_AT"))
    release_config_path = _safe_server_path(
        _required_value(environment, "TEST_RELEASE_CONFIG_PATH")
    )
    gold_config_path = _safe_server_path(
        _required_value(environment, "TEST_GOLD_CONFIG_PATH")
    )
    task_manifest_path = _safe_server_path(
        _required_value(environment, "TEST_TASKS_FILE") or FINAL_TEST_TASK_MANIFEST_PATH
    )
    configured_attempts = _required_value(environment, "TEST_MAX_ATTEMPTS") or "3"
    valid = (
        bool(release_id and _RELEASE_ID.fullmatch(release_id))
        and bool(task_manifest_sha256 and _SHA256.fullmatch(task_manifest_sha256))
        and bool(gold_sha256 and _SHA256.fullmatch(gold_sha256))
        and open_at is not None
        and close_at is not None
        and open_at < close_at
        and close_at == OFFICIAL_TEST_CLOSE_AT
        and release_config_path == FINAL_TEST_RELEASE_PATH
        and gold_config_path == FINAL_TEST_GOLD_PATH
        and task_manifest_path == FINAL_TEST_TASK_MANIFEST_PATH
        and configured_attempts == "3"
    )
    return TestDeploymentConfig(
        submissions_enabled=requested_submissions and valid,
        public_leaderboard_enabled=requested_public_leaderboard and valid,
        provisional_leaderboard_enabled=requested_provisional_leaderboard and valid,
        release_id=release_id,
        task_manifest_sha256=task_manifest_sha256,
        gold_sha256=gold_sha256,
        open_at=open_at,
        close_at=close_at,
        release_config_path=release_config_path,
        gold_config_path=gold_config_path,
        feedback_policy=FINAL_TEST_FEEDBACK_POLICY,
        task_manifest_path=task_manifest_path,
    )


TEST_DEPLOYMENT = load_test_deployment_config()
VALIDATION_SUBMISSIONS_ENABLED = load_validation_submissions_enabled()
TEST_SUBMISSIONS_ENABLED = TEST_DEPLOYMENT.submissions_enabled
TEST_PUBLIC_LEADERBOARD_ENABLED = TEST_DEPLOYMENT.public_leaderboard_enabled
TEST_PROVISIONAL_LEADERBOARD_ENABLED = TEST_DEPLOYMENT.provisional_leaderboard_enabled
TEST_TASKS_FILE = TEST_DEPLOYMENT.task_manifest_path
VALIDATION_SPLIT_LABEL = "Validation (development)"
TEST_SPLIT_LABEL = "Test (final)"
VALIDATION_LEADERBOARD_LABEL = "Validation leaderboard"
FINAL_TEST_LEADERBOARD_LABEL = "Final test leaderboard"

PORTAL_CSS = """
html,
body {
    height: 100%;
    overflow: hidden !important;
}

:root {
    --docsem-navy: #17365f;
    --docsem-teal: #177f78;
    --docsem-coral: #cc4b2c;
    --docsem-gold: #d9a62e;
    --docsem-ink: #17212f;
    --docsem-muted: #5d6878;
    --docsem-line: #d9dee5;
    --docsem-surface: #ffffff;
    --docsem-page: #f7f8fa;
}

.gradio-container {
    width: 100% !important;
    max-width: none !important;
    height: 100vh !important;
    height: 100dvh !important;
    max-height: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    overscroll-behavior-y: contain;
    -webkit-overflow-scrolling: touch;
    background: var(--docsem-page);
    color: var(--docsem-ink);
    font-size: 16px;
}

.gradio-container > .main {
    box-sizing: border-box;
    width: 100%;
    max-width: 1540px;
    min-width: 0;
    margin: 0 auto;
    padding: 24px 30px 36px;
    overflow: visible !important;
}

#portal-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 24px;
    padding: 6px 0 24px;
    border-bottom: 1px solid var(--docsem-line);
    margin-bottom: 20px;
}

#portal-header .portal-kicker {
    display: block;
    color: var(--docsem-coral);
    font-size: 13px;
    font-weight: 750;
    line-height: 1.3;
    letter-spacing: 0;
    margin-bottom: 6px;
    text-transform: uppercase;
}

#portal-header h1 {
    color: var(--docsem-navy);
    font-size: 36px;
    line-height: 1.15;
    letter-spacing: 0;
    margin: 0 0 8px;
}

#portal-header p {
    color: var(--docsem-muted);
    font-size: 16px;
    line-height: 1.5;
    margin: 0;
}

#portal-header .portal-summary {
    color: var(--docsem-ink);
    font-size: 18px;
    font-weight: 650;
    margin-bottom: 4px;
}

#portal-header .portal-links {
    display: flex;
    align-items: center;
    flex-wrap: nowrap;
    justify-content: flex-end;
    gap: 10px;
}

#portal-header a {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-height: 44px;
    padding: 0 16px;
    color: var(--docsem-navy);
    background: var(--docsem-surface);
    border: 1px solid #b9c4d1;
    border-radius: 6px;
    font-size: 15px;
    font-weight: 700;
    text-decoration: none;
    white-space: nowrap;
}

#portal-header a.primary-link {
    color: #ffffff;
    background: var(--docsem-navy);
    border-color: var(--docsem-navy);
}

#portal-header a:hover {
    border-color: var(--docsem-teal);
    color: var(--docsem-teal);
}

#portal-header a.primary-link:hover {
    color: #ffffff;
    background: var(--docsem-teal);
}

#evaluation-notice {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0 24px;
    margin: 0 0 20px;
    padding: 17px 20px;
    background: #edf7f5;
    border: 1px solid #bdddd8;
    border-left: 4px solid var(--docsem-teal);
    border-radius: 6px;
}

#evaluation-notice h2 {
    grid-column: 1 / -1;
    color: var(--docsem-navy);
    font-size: 20px;
    line-height: 1.25;
    letter-spacing: 0;
    margin: 0 0 9px;
}

#evaluation-notice p {
    color: #34475a;
    font-size: 15px;
    line-height: 1.5;
    margin: 0;
}

#evaluation-notice > p + p {
    border-left: 1px solid #bdddd8;
    padding-left: 24px;
}

#evaluation-notice a {
    color: var(--docsem-teal);
    font-weight: 700;
}

#evaluation-notice .test-release-notice {
    grid-column: 1 / -1;
    margin-top: 14px;
    padding-top: 14px;
    border-top: 1px solid #bdddd8;
}

.test-release-notice p {
    color: #34475a;
    font-size: 15px;
    line-height: 1.5;
    margin: 0;
}

.test-release-notice p + p {
    margin-top: 8px;
}

.test-release-notice a {
    color: var(--docsem-teal);
    font-weight: 700;
}

.test-countdown {
    color: var(--docsem-navy);
    font-variant-numeric: tabular-nums;
    font-weight: 700;
}

.column > #submission-panel {
    padding: 18px 20px 16px;
    background: var(--docsem-surface);
    border: 1px solid var(--docsem-line);
    border-left: 4px solid var(--docsem-coral);
    border-radius: 6px;
}

#submission-panel #submission-panel {
    padding: 0;
    background: transparent;
    border: 0;
    border-radius: 0;
}

#submission-panel .styler {
    background: transparent;
}

#submission-panel h2 {
    color: var(--docsem-ink);
    font-size: 23px;
    line-height: 1.25;
    letter-spacing: 0;
    margin: 0 0 5px;
}

#submission-panel .submission-note {
    color: var(--docsem-muted);
    font-size: 15px;
    line-height: 1.5;
    margin: 0 0 12px;
}

#submission-panel .submission-note a {
    color: var(--docsem-teal);
    font-weight: 700;
    text-decoration: none;
}

#submission-panel .submission-note a:hover {
    text-decoration: underline;
}

#submission-fields {
    gap: 14px;
}

#submission-fields .form {
    border: 0;
}

#submission-fields label span {
    color: #465263;
    font-size: 14px;
    font-weight: 600;
}

#submission-fields input {
    font-size: 16px;
}

#submission-actions {
    align-items: stretch;
    gap: 16px;
    margin-top: 8px;
}

#submission-file {
    min-height: 112px !important;
    height: auto !important;
    font-size: 15px;
}

#submission-file > div {
    min-height: 110px !important;
    height: auto !important;
}

#submission-file .file-preview-holder,
#submission-file .file-preview {
    max-height: none !important;
    overflow: visible !important;
}

#submission-side {
    justify-content: center;
    padding: 2px 0;
}

#submission-side p {
    color: var(--docsem-muted);
    font-size: 14px;
    line-height: 1.5;
    margin: 0 0 10px;
}

#submit-button {
    min-height: 46px;
    border-radius: 6px;
    font-size: 16px;
    font-weight: 700;
}

#score-output {
    max-height: 178px;
    margin-top: 14px;
    overflow: auto;
}

#dataset-citation {
    margin-top: 18px;
    border-color: var(--docsem-line);
}

#dataset-citation-code {
    max-height: 260px;
}

#leaderboard-section {
    margin-top: 28px;
}

#leaderboard-heading {
    align-items: center;
    justify-content: space-between;
    gap: 18px;
    margin-bottom: 10px;
}

#leaderboard-heading h2 {
    color: var(--docsem-navy);
    font-size: 29px;
    line-height: 1.2;
    letter-spacing: 0;
    margin: 0 0 6px;
}

#leaderboard-heading p {
    color: var(--docsem-muted);
    font-size: 15px;
    line-height: 1.45;
    margin: 0;
}

#refresh-button {
    max-width: 170px;
    min-height: 42px;
    border-radius: 6px;
    font-size: 15px;
    font-weight: 700;
}

#leaderboard-table {
    background: var(--docsem-surface);
    border: 1px solid var(--docsem-line);
    border-radius: 6px;
    overflow: hidden;
}

#leaderboard-table .leaderboard-table-wrap {
    width: 100%;
    overflow-x: auto;
}

#leaderboard-table table {
    width: 100%;
    min-width: 920px;
    border-collapse: collapse;
    table-layout: fixed;
    font-size: 15px;
}

#leaderboard-table th {
    color: var(--docsem-navy);
    background: #eef2f6;
    font-size: 14px;
    font-weight: 700;
    line-height: 1.25;
    white-space: normal;
    text-align: left;
}

#leaderboard-table td,
#leaderboard-table th {
    padding: 13px 14px;
    border-bottom: 1px solid var(--docsem-line);
    vertical-align: middle;
}

#leaderboard-table td {
    color: var(--docsem-ink);
    line-height: 1.35;
    overflow-wrap: anywhere;
}

#leaderboard-table tbody tr:last-child td {
    border-bottom: 0;
}

#leaderboard-table tbody tr:hover {
    background: #f8fafc;
}

#leaderboard-table .leaderboard-rank,
#leaderboard-table .leaderboard-attempts,
#leaderboard-table .leaderboard-metric {
    text-align: center;
}

#leaderboard-table .leaderboard-metric {
    color: var(--docsem-navy);
    font-variant-numeric: tabular-nums;
    font-weight: 700;
    white-space: nowrap;
}

#leaderboard-table .leaderboard-date {
    color: var(--docsem-muted);
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
}

#leaderboard-table .leaderboard-empty {
    padding: 34px 20px;
    color: var(--docsem-muted);
    text-align: center;
}

@media (max-width: 760px) {
    .gradio-container > .main {
        width: 100%;
        max-width: 100%;
        min-width: 0;
        padding: 16px 14px 26px;
        overflow-x: hidden !important;
    }

    #split-controls,
    #submission-fields,
    #submission-actions,
    #leaderboard-heading {
        min-width: 0;
        max-width: 100%;
    }

    #submission-actions > *,
    #submission-fields > * {
        min-width: 0 !important;
        max-width: 100%;
    }

    #portal-header {
        align-items: flex-start;
        flex-direction: column;
        gap: 14px;
    }

    #portal-header h1 {
        font-size: 28px;
    }

    #portal-header .portal-summary {
        font-size: 17px;
    }

    #portal-header .portal-links {
        width: 100%;
        flex-wrap: wrap;
    }

    #portal-header a {
        flex: 1 1 140px;
        min-width: 0;
        padding: 0 10px;
    }

    #evaluation-notice {
        grid-template-columns: 1fr;
        gap: 12px;
        padding: 15px 14px;
    }

    #evaluation-notice > p + p {
        border-top: 1px solid #bdddd8;
        border-left: 0;
        padding-top: 12px;
        padding-left: 0;
    }

    .column > #submission-panel {
        padding: 16px 14px;
    }

    #leaderboard-heading {
        align-items: flex-start;
        flex-direction: column;
    }

    #refresh-button {
        max-width: none;
    }
}
"""


def _read_hub_file(
    repo_id,
    filename,
    token=None,
    force_download=False,
    revision=None,
    max_bytes=None,
):
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        token=token,
        force_download=force_download,
        revision=revision,
    )
    file_path = Path(path)
    if max_bytes is not None:
        try:
            size = file_path.stat().st_size
        except OSError as exc:
            raise FinalLeaderboardError(
                "The final test leaderboard is not available."
            ) from exc
        if size > max_bytes:
            raise FinalLeaderboardError("The final test leaderboard is not available.")
    try:
        payload = file_path.read_bytes()
        if max_bytes is not None and len(payload) > max_bytes:
            raise FinalLeaderboardError("The final test leaderboard is not available.")
        return payload.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise FinalLeaderboardError(
            "The final test leaderboard is not available."
        ) from exc


def _load_gold_rows():
    return load_jsonl_text(_read_hub_file(GOLD_REPO_ID, GOLD_FILE, token=WRITE_TOKEN))


def _persist_submission(
    rows, team, contact, submission_name, metrics, participant_names=None
):
    if not SUBMISSIONS_REPO_ID or not WRITE_TOKEN:
        return "Score computed. Persistence is disabled until SUBMISSIONS_REPO_ID and HF_WRITE_TOKEN are configured."

    submitted_at = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    row = leaderboard_row(
        team,
        contact,
        submission_name,
        metrics,
        submitted_at,
        participant_names=participant_names,
    )
    payload = {
        "leaderboard": row,
        "metrics": metrics,
        "predictions": rows,
    }
    timestamp = submitted_at.replace(":", "").replace("-", "")
    team_slug = safe_slug(team)
    name_slug = safe_slug(submission_name)
    filename = f"{timestamp}_{team_slug}_{name_slug}.json"
    tmp_path = Path("/tmp") / filename
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    upload_file(
        path_or_fileobj=str(tmp_path),
        path_in_repo=f"submissions/{filename}",
        repo_id=SUBMISSIONS_REPO_ID,
        repo_type="dataset",
        token=WRITE_TOKEN,
        commit_message=f"Add submission {team_slug}/{name_slug}",
    )
    attempts = _update_leaderboard(row)
    return (
        f"Score computed and saved to {SUBMISSIONS_REPO_ID}/submissions/{filename}. "
        f"This is attempt {attempts} for this team and contact email."
    )


def _load_leaderboard_rows():
    if not SUBMISSIONS_REPO_ID or not WRITE_TOKEN:
        return []
    try:
        text = _read_hub_file(
            SUBMISSIONS_REPO_ID,
            "leaderboard/leaderboard.json",
            token=WRITE_TOKEN,
            force_download=True,
        )
    except (EntryNotFoundError, RepositoryNotFoundError):
        return []
    rows = json.loads(text)
    return rows if isinstance(rows, list) else []


def _sort_leaderboard(rows):
    return rank_leaderboard(rows)


def _update_leaderboard(row):
    with LEADERBOARD_LOCK:
        rows = _load_leaderboard_rows()
        rows.append(row)
        rows.sort(key=lambda item: str(item.get("submitted_at", "")))
        tmp_path = Path("/tmp") / "leaderboard.json"
        tmp_path.write_text(
            json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
        )
        upload_file(
            path_or_fileobj=str(tmp_path),
            path_in_repo="leaderboard/leaderboard.json",
            repo_id=SUBMISSIONS_REPO_ID,
            repo_type="dataset",
            token=WRITE_TOKEN,
            commit_message="Update leaderboard",
        )
        identity = leaderboard_identity(row)
        latest = next(
            item
            for item in rank_leaderboard(rows)
            if leaderboard_identity(item) == identity
        )
        return latest["attempts"]


def _format_metric(value):
    return f"{float(value) * 100:.2f}%"


def _format_joint_metric(value):
    if value is None:
        return "Not yet computed"
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return "Not yet computed"
    if not math.isfinite(metric):
        return "Not yet computed"
    return _format_metric(metric)


def _format_timestamp(value):
    return str(value).replace("T", " ").removesuffix("Z")


def leaderboard_html():
    rows = _sort_leaderboard(_load_leaderboard_rows())
    body_rows = []
    for index, row in enumerate(rows[:100], start=1):
        body_rows.append(
            f"""
            <tr>
                <td class="leaderboard-rank">{index}</td>
                <td>{html.escape(str(row.get("team", "")))}</td>
                <td>{html.escape(str(row.get("submission_name", "")))}</td>
                <td class="leaderboard-attempts">{int(row.get("attempts", 1))}</td>
                <td class="leaderboard-metric">{_format_joint_metric(row.get("joint_accuracy"))}</td>
                <td class="leaderboard-metric">{_format_metric(row.get("answer_accuracy", 0.0))}</td>
                <td class="leaderboard-metric">{_format_metric(row.get("evidence_f1", 0.0))}</td>
                <td class="leaderboard-date">{html.escape(_format_timestamp(row.get("submitted_at", "")))}</td>
            </tr>
            """
        )

    if not body_rows:
        body_rows.append(
            '<tr><td class="leaderboard-empty" colspan="8">No scored submissions yet.</td></tr>'
        )

    return f"""
    <div class="leaderboard-table-wrap">
        <table aria-label="DocSem validation leaderboard">
            <colgroup>
                <col style="width: 6%">
                <col style="width: 17%">
                <col style="width: 18%">
                <col style="width: 9%">
                <col style="width: 13%">
                <col style="width: 13%">
                <col style="width: 11%">
                <col style="width: 13%">
            </colgroup>
            <thead>
                <tr>
                    <th class="leaderboard-rank" scope="col">Rank</th>
                    <th scope="col">Team</th>
                    <th scope="col">Latest submission</th>
                    <th class="leaderboard-attempts" scope="col">Attempts</th>
                    <th class="leaderboard-metric" scope="col">Joint Exact Accuracy</th>
                    <th class="leaderboard-metric" scope="col">Answer Exact Accuracy</th>
                    <th class="leaderboard-metric" scope="col">Evidence F1 (macro)</th>
                    <th scope="col">Submitted (UTC)</th>
                </tr>
            </thead>
            <tbody>
                {"".join(body_rows)}
            </tbody>
        </table>
    </div>
    """


def _decode_final_json(raw):
    """Decode one bounded JSON document while rejecting duplicate keys and NaN."""

    if isinstance(raw, str):
        try:
            raw = raw.encode("utf-8")
        except UnicodeEncodeError:
            raise FinalLeaderboardError(
                "The final test leaderboard is not available."
            ) from None
    if not isinstance(raw, bytes) or len(raw) > FINAL_TEST_ARTIFACT_MAX_BYTES:
        raise FinalLeaderboardError("The final test leaderboard is not available.")

    def object_without_duplicates(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate key")
            value[key] = item
        return value

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=object_without_duplicates,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("constant")),
        )
    except (UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError):
        raise FinalLeaderboardError(
            "The final test leaderboard is not available."
        ) from None


def _artifact_bytes(raw):
    if isinstance(raw, str):
        try:
            raw = raw.encode("utf-8")
        except UnicodeEncodeError:
            raise FinalLeaderboardError(
                "The final test leaderboard is not available."
            ) from None
    if not isinstance(raw, bytes) or len(raw) > FINAL_TEST_ARTIFACT_MAX_BYTES:
        raise FinalLeaderboardError("The final test leaderboard is not available.")
    return raw


def _valid_public_text(value):
    return is_valid_public_text(value)


def _validate_final_projection(projection, release):
    if (
        not isinstance(projection, Mapping)
        or set(projection) != FINAL_TEST_PROJECTION_FIELDS
        or type(projection.get("schema_version")) is not int
        or projection.get("schema_version") != 3
        or projection.get("split") != "test"
        or projection.get("release_id") != release.get("release_id")
        or projection.get("task_manifest_sha256") != release.get("task_manifest_sha256")
        or not isinstance(projection.get("rows"), list)
        or len(projection["rows"]) > FINAL_TEST_MAX_ROWS
    ):
        raise FinalLeaderboardError("The final test leaderboard is not available.")
    for expected_rank, row in enumerate(projection["rows"], start=1):
        if (
            not isinstance(row, Mapping)
            or set(row) != FINAL_TEST_PUBLIC_ROW_FIELDS
            or type(row.get("rank")) is not int
            or row["rank"] != expected_rank
            or type(row.get("total_attempts")) is not int
            or not 1 <= row["total_attempts"] <= 3
            or type(row.get("selected_attempt")) is not int
            or not 1 <= row["selected_attempt"] <= row["total_attempts"]
            or any(
                not _valid_public_text(row.get(field))
                for field in ("team", "submission_name")
            )
        ):
            raise FinalLeaderboardError("The final test leaderboard is not available.")
        metric = row.get("joint_accuracy")
        if (
            type(metric) is not float
            or not math.isfinite(metric)
            or not 0.0 <= metric <= 1.0
        ):
            raise FinalLeaderboardError("The final test leaderboard is not available.")


def _validate_provisional_projection(projection, release):
    if (
        not isinstance(projection, Mapping)
        or set(projection) != PROVISIONAL_TEST_PROJECTION_FIELDS
        or type(projection.get("schema_version")) is not int
        or projection.get("schema_version") != 3
        or projection.get("split") != "test"
        or projection.get("release_id") != release.get("release_id")
        or projection.get("task_manifest_sha256") != release.get("task_manifest_sha256")
        or not isinstance(projection.get("rows"), list)
        or len(projection["rows"]) > FINAL_TEST_MAX_ROWS
    ):
        raise FinalLeaderboardError("The test leaderboard is not available.")
    for expected_rank, row in enumerate(projection["rows"], start=1):
        if (
            not isinstance(row, Mapping)
            or set(row) != PROVISIONAL_TEST_PUBLIC_ROW_FIELDS
            or type(row.get("rank")) is not int
            or row["rank"] != expected_rank
            or any(
                not _valid_public_text(row.get(field))
                for field in ("hf_username", "team")
            )
        ):
            raise FinalLeaderboardError("The test leaderboard is not available.")


def _normalized_utc(value):
    if not isinstance(value, dt.datetime) or value.tzinfo is None:
        return None
    try:
        if value.utcoffset() is None:
            return None
        return value.astimezone(dt.timezone.utc)
    except (OverflowError, ValueError):
        return None


def _validate_final_deployment(deployment):
    opened = _normalized_utc(getattr(deployment, "open_at", None))
    closed = _normalized_utc(getattr(deployment, "close_at", None))
    release_id = getattr(deployment, "release_id", None)
    task_digest = getattr(deployment, "task_manifest_sha256", None)
    gold_digest = getattr(deployment, "gold_sha256", None)
    max_attempts = getattr(deployment, "max_attempts", None)
    feedback_policy = getattr(deployment, "feedback_policy", None)
    task_manifest_path = getattr(deployment, "task_manifest_path", None)
    if (
        not isinstance(release_id, str)
        or _RELEASE_ID.fullmatch(release_id) is None
        or not isinstance(task_digest, str)
        or _SHA256.fullmatch(task_digest) is None
        or not isinstance(gold_digest, str)
        or _SHA256.fullmatch(gold_digest) is None
        or opened is None
        or closed is None
        or opened >= closed
        or closed != OFFICIAL_TEST_CLOSE_AT
        or type(max_attempts) is not int
        or max_attempts != 3
        or feedback_policy != FINAL_TEST_FEEDBACK_POLICY
        or task_manifest_path != FINAL_TEST_TASK_MANIFEST_PATH
        or getattr(deployment, "release_config_path", None) != FINAL_TEST_RELEASE_PATH
        or getattr(deployment, "gold_config_path", None) != FINAL_TEST_GOLD_PATH
    ):
        raise FinalLeaderboardError("The final test leaderboard is not available.")
    return opened, closed


def _validate_active_test_release(release, deployment, now):
    required = {
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
    }
    opened = (
        _parse_rfc3339_utc(release.get("open_at"))
        if isinstance(release, Mapping)
        else None
    )
    closed = (
        _parse_rfc3339_utc(release.get("close_at"))
        if isinstance(release, Mapping)
        else None
    )
    current = _normalized_utc(now)
    configured_opened, configured_closed = _validate_final_deployment(deployment)
    if (
        not isinstance(release, Mapping)
        or not required.issubset(release)
        or type(release.get("schema_version")) is not int
        or release.get("schema_version") != 1
        or release.get("release_id") != getattr(deployment, "release_id", None)
        or release.get("task_manifest_sha256")
        != getattr(deployment, "task_manifest_sha256", None)
        or release.get("gold_sha256") != getattr(deployment, "gold_sha256", None)
        or release.get("enabled") is not True
        or release.get("finalized") is not False
        or type(release.get("max_attempts")) is not int
        or release.get("max_attempts") != 3
        or release.get("feedback_policy") != FINAL_TEST_FEEDBACK_POLICY
        or opened != configured_opened
        or closed != configured_closed
        or current is None
        or not opened <= current < closed
        or _REVISION.fullmatch(str(release.get("public_revision", ""))) is None
        or _REPOSITORY_ID.fullmatch(str(release.get("public_repo_id", ""))) is None
        or release.get("task_manifest_path") != FINAL_TEST_TASK_MANIFEST_PATH
    ):
        raise FinalLeaderboardError("The test leaderboard is not available.")


def _validate_final_release(release, deployment):
    required = {
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
        "finalized_at",
        "finalization_source_revision",
        "finalization_scorer_revision",
        "finalization_scorer_sha256",
        "final_projection_sha256",
        "finalization_audit_sha256",
    }
    opened = (
        _parse_rfc3339_utc(release.get("open_at"))
        if isinstance(release, Mapping)
        else None
    )
    closed = (
        _parse_rfc3339_utc(release.get("close_at"))
        if isinstance(release, Mapping)
        else None
    )
    finalized = (
        _parse_rfc3339_utc(release.get("finalized_at"))
        if isinstance(release, Mapping)
        else None
    )
    configured_opened, configured_closed = _validate_final_deployment(deployment)
    configured_attempts = getattr(deployment, "max_attempts", None)
    configured_feedback = getattr(deployment, "feedback_policy", None)
    configured_task_path = getattr(deployment, "task_manifest_path", None)
    if (
        not isinstance(release, Mapping)
        or not required.issubset(release)
        or type(release.get("schema_version")) is not int
        or release.get("schema_version") != 1
        or release.get("release_id") != getattr(deployment, "release_id", None)
        or release.get("task_manifest_sha256")
        != getattr(deployment, "task_manifest_sha256", None)
        or release.get("gold_sha256") != getattr(deployment, "gold_sha256", None)
        or _SHA256.fullmatch(str(release.get("task_manifest_sha256", ""))) is None
        or _SHA256.fullmatch(str(release.get("gold_sha256", ""))) is None
        or release.get("enabled") is not False
        or release.get("finalized") is not True
        or type(release.get("max_attempts")) is not int
        or release.get("max_attempts") != configured_attempts
        or configured_attempts != 3
        or release.get("feedback_policy") != configured_feedback
        or configured_feedback != FINAL_TEST_FEEDBACK_POLICY
        or opened is None
        or closed is None
        or finalized is None
        or opened != configured_opened
        or closed != configured_closed
        or not opened < closed <= finalized
        or _REVISION.fullmatch(str(release.get("public_revision", ""))) is None
        or _REPOSITORY_ID.fullmatch(str(release.get("public_repo_id", ""))) is None
        or release.get("task_manifest_path") != configured_task_path
        or configured_task_path != FINAL_TEST_TASK_MANIFEST_PATH
        or _REVISION.fullmatch(str(release.get("finalization_source_revision", "")))
        is None
        or _REVISION.fullmatch(str(release.get("finalization_scorer_revision", "")))
        is None
        or any(
            _SHA256.fullmatch(str(release.get(field, ""))) is None
            for field in (
                "finalization_scorer_sha256",
                "final_projection_sha256",
                "finalization_audit_sha256",
            )
        )
    ):
        raise FinalLeaderboardError("The final test leaderboard is not available.")


def _validate_final_audit(audit, release, projection_sha256):
    required = {
        "schema_version",
        "split",
        "release_id",
        "source_revision",
        "finalized_at",
        "close_at",
        "task_manifest_sha256",
        "gold_sha256",
        "scorer_revision",
        "scorer_code_sha256",
        "public_projection_sha256",
        "selected_account_count",
    }
    if (
        not isinstance(audit, Mapping)
        or not required.issubset(audit)
        or type(audit.get("schema_version")) is not int
        or audit.get("schema_version") != 1
        or audit.get("split") != "test"
        or audit.get("release_id") != release.get("release_id")
        or audit.get("source_revision") != release.get("finalization_source_revision")
        or audit.get("finalized_at") != release.get("finalized_at")
        or audit.get("close_at") != release.get("close_at")
        or audit.get("task_manifest_sha256") != release.get("task_manifest_sha256")
        or audit.get("gold_sha256") != release.get("gold_sha256")
        or audit.get("scorer_revision") != release.get("finalization_scorer_revision")
        or audit.get("scorer_code_sha256") != release.get("finalization_scorer_sha256")
        or audit.get("public_projection_sha256") != projection_sha256
        or type(audit.get("selected_account_count")) is not int
        or audit.get("selected_account_count") < 0
    ):
        raise FinalLeaderboardError("The final test leaderboard is not available.")
    eligible_attempts = audit.get("eligible_attempts")
    if not isinstance(eligible_attempts, list) or len(eligible_attempts) != audit.get(
        "eligible_attempt_count"
    ):
        raise FinalLeaderboardError("The final test leaderboard is not available.")
    for attempt in eligible_attempts:
        if (
            not isinstance(attempt, Mapping)
            or set(attempt) != FINAL_TEST_ELIGIBLE_ATTEMPT_FIELDS
            or _SHA256.fullmatch(str(attempt.get("account_key", ""))) is None
            or not _valid_public_text(attempt.get("submission_id"))
            or type(attempt.get("attempt_number")) is not int
            or not 1 <= attempt["attempt_number"] <= 3
            or type(attempt.get("selected")) is not bool
            or _SHA256.fullmatch(str(attempt.get("record_sha256", ""))) is None
            or _SHA256.fullmatch(str(attempt.get("rescored_metrics_sha256", "")))
            is None
        ):
            raise FinalLeaderboardError("The final test leaderboard is not available.")
        for field in ("joint_accuracy", "answer_accuracy", "evidence_f1"):
            metric = attempt.get(field)
            if (
                type(metric) is not float
                or not math.isfinite(metric)
                or not 0.0 <= metric <= 1.0
            ):
                raise FinalLeaderboardError(
                    "The final test leaderboard is not available."
                )


def _load_provisional_test_projection(
    *,
    api=None,
    artifact_reader=None,
    deployment=None,
    repo_id=None,
    token=None,
    now=None,
):
    """Load one rank-only provisional projection from one exact private HEAD."""

    api = _TEST_HUB_API if api is None else api
    deployment = TEST_DEPLOYMENT if deployment is None else deployment
    repo_id = SUBMISSIONS_REPO_ID if repo_id is None else repo_id
    token = WRITE_TOKEN if token is None else token
    current = _server_now() if now is None else now
    _validate_final_deployment(deployment)
    if (
        not isinstance(repo_id, str)
        or _REPOSITORY_ID.fullmatch(repo_id) is None
        or not isinstance(token, str)
        or not token.strip()
    ):
        raise FinalLeaderboardError("The test leaderboard is not available.")
    try:
        info = api.repo_info(
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
        revision = getattr(info, "sha", None)
        if (
            getattr(info, "private", None) is not True
            or not isinstance(revision, str)
            or _REVISION.fullmatch(revision) is None
        ):
            raise FinalLeaderboardError("The test leaderboard is not available.")

        if artifact_reader is None:

            def artifact_reader(path, pinned_revision):
                return _read_hub_file(
                    repo_id,
                    path,
                    token=token,
                    force_download=True,
                    revision=pinned_revision,
                    max_bytes=FINAL_TEST_ARTIFACT_MAX_BYTES,
                )

        release = _decode_final_json(
            _artifact_bytes(artifact_reader(FINAL_TEST_RELEASE_PATH, revision))
        )
        projection = _decode_final_json(
            _artifact_bytes(artifact_reader(PROVISIONAL_TEST_PROJECTION_PATH, revision))
        )
        _validate_active_test_release(release, deployment, current)
        _validate_provisional_projection(projection, release)
        return projection
    except FinalLeaderboardError:
        raise
    except Exception:
        raise FinalLeaderboardError("The test leaderboard is not available.") from None


def _load_final_test_projection(
    *,
    api=None,
    artifact_reader=None,
    deployment=None,
    repo_id=None,
    token=None,
):
    """Load one sanitized finalized projection from an exact private HEAD."""

    api = _TEST_HUB_API if api is None else api
    deployment = TEST_DEPLOYMENT if deployment is None else deployment
    repo_id = SUBMISSIONS_REPO_ID if repo_id is None else repo_id
    token = WRITE_TOKEN if token is None else token
    _validate_final_deployment(deployment)
    if (
        not isinstance(repo_id, str)
        or _REPOSITORY_ID.fullmatch(repo_id) is None
        or not isinstance(token, str)
        or not token.strip()
    ):
        raise FinalLeaderboardError("The final test leaderboard is not available.")
    try:
        info = api.repo_info(
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
        revision = getattr(info, "sha", None)
        if (
            getattr(info, "private", None) is not True
            or not isinstance(revision, str)
            or _REVISION.fullmatch(revision) is None
        ):
            raise FinalLeaderboardError("The final test leaderboard is not available.")

        if artifact_reader is None:

            def artifact_reader(path, pinned_revision):
                return _read_hub_file(
                    repo_id,
                    path,
                    token=token,
                    force_download=True,
                    revision=pinned_revision,
                    max_bytes=FINAL_TEST_ARTIFACT_MAX_BYTES,
                )

        release_raw = _artifact_bytes(
            artifact_reader(FINAL_TEST_RELEASE_PATH, revision)
        )
        projection_raw = _artifact_bytes(
            artifact_reader(FINAL_TEST_PROJECTION_PATH, revision)
        )
        audit_raw = _artifact_bytes(artifact_reader(FINAL_TEST_AUDIT_PATH, revision))

        release = _decode_final_json(release_raw)
        projection = _decode_final_json(projection_raw)
        audit = _decode_final_json(audit_raw)
        _validate_final_release(release, deployment)
        projection_sha256 = hashlib.sha256(projection_raw).hexdigest()
        if projection_sha256 != release.get("final_projection_sha256"):
            raise FinalLeaderboardError("The final test leaderboard is not available.")
        if hashlib.sha256(audit_raw).hexdigest() != release.get(
            "finalization_audit_sha256"
        ):
            raise FinalLeaderboardError("The final test leaderboard is not available.")
        _validate_final_projection(projection, release)
        _validate_final_audit(audit, release, projection_sha256)
        if audit.get("selected_account_count") != len(projection["rows"]):
            raise FinalLeaderboardError("The final test leaderboard is not available.")
        return projection
    except FinalLeaderboardError:
        raise
    except Exception:
        raise FinalLeaderboardError(
            "The final test leaderboard is not available."
        ) from None


def final_test_leaderboard_html(projection):
    """Render only the exact public final projection with escaped text."""

    _validate_final_projection(
        projection,
        {
            "release_id": projection.get("release_id")
            if isinstance(projection, Mapping)
            else None,
            "task_manifest_sha256": projection.get("task_manifest_sha256")
            if isinstance(projection, Mapping)
            else None,
        },
    )
    body_rows = []
    for row in projection["rows"]:
        body_rows.append(
            "<tr>"
            f'<td class="leaderboard-rank">{row["rank"]}</td>'
            f"<td>{html.escape(row['team'])}</td>"
            f"<td>{html.escape(row['submission_name'])}</td>"
            f'<td class="leaderboard-attempts">{row["selected_attempt"]}</td>'
            f'<td class="leaderboard-attempts">{row["total_attempts"]}</td>'
            f'<td class="leaderboard-metric">{_format_metric(row["joint_accuracy"])}</td>'
            "</tr>"
        )
    if not body_rows:
        body_rows.append(
            '<tr><td class="leaderboard-empty" colspan="6">No eligible final test submissions.</td></tr>'
        )
    return f"""
    <div class="leaderboard-table-wrap">
        <table aria-label="DocSem final test leaderboard">
            <colgroup>
                <col style="width: 7%;">
                <col style="width: 22%;">
                <col style="width: 24%;">
                <col style="width: 12%;">
                <col style="width: 12%;">
                <col style="width: 23%;">
            </colgroup>
            <thead>
                <tr>
                    <th class="leaderboard-rank" scope="col">Rank</th>
                    <th scope="col">Team</th>
                    <th scope="col">Submission name</th>
                    <th class="leaderboard-attempts" scope="col">Selected attempt</th>
                    <th class="leaderboard-attempts" scope="col">Total attempts</th>
                    <th class="leaderboard-metric" scope="col">Joint Exact Accuracy</th>
                </tr>
            </thead>
            <tbody>{"".join(body_rows)}</tbody>
        </table>
    </div>
    """


def provisional_test_leaderboard_html(projection):
    """Render only public rank, Hugging Face account, and team fields."""

    _validate_provisional_projection(
        projection,
        {
            "release_id": projection.get("release_id")
            if isinstance(projection, Mapping)
            else None,
            "task_manifest_sha256": projection.get("task_manifest_sha256")
            if isinstance(projection, Mapping)
            else None,
        },
    )
    body_rows = [
        "<tr>"
        f'<td class="leaderboard-rank">{row["rank"]}</td>'
        f"<td>{html.escape(row['hf_username'])}</td>"
        f"<td>{html.escape(row['team'])}</td>"
        "</tr>"
        for row in projection["rows"]
    ]
    if not body_rows:
        body_rows.append(
            '<tr><td class="leaderboard-empty" colspan="3">No accepted test submissions yet.</td></tr>'
        )
    return f"""
    <div class="leaderboard-table-wrap">
        <table aria-label="DocSem provisional test leaderboard">
            <thead>
                <tr>
                    <th class="leaderboard-rank" scope="col">Rank</th>
                    <th scope="col">Hugging Face account</th>
                    <th scope="col">Team</th>
                </tr>
            </thead>
            <tbody>{"".join(body_rows)}</tbody>
        </table>
    </div>
    """


def _validation_leaderboard_heading():
    return """
    <div>
        <h2>Validation leaderboard</h2>
        <p>Provisional validation results from each team's latest attempt. Answer Exact Accuracy is the share with an exact normalized answer; Evidence F1 (macro) gives partial credit for overlap between predicted and gold evidence sets; Joint Exact Accuracy requires both the exact normalized answer and the entire normalized evidence set to be correct on the same example. Ranked by Joint Exact Accuracy, then Answer Exact Accuracy, then Evidence F1, accepted time, and stable submission ID. Leaderboard refreshed September 3, 2026 after the organizer-only ground-truth correction; all existing submissions were rescored. Final standings will use the held-out test set.</p>
    </div>
    """


def _final_test_leaderboard_heading():
    return """
    <div>
        <h2>Final test leaderboard</h2>
        <p>Final standings use each account's best eligible attempt from at most three accepted test submissions. Joint Exact Accuracy is the public score. Total attempts counts all accepted submissions, including any attempt excluded from final selection.</p>
    </div>
    """


def _provisional_test_leaderboard_heading():
    return """
    <div>
        <h2>Provisional test leaderboard</h2>
        <p>During the open window, public standings use only each Hugging Face account's first accepted attempt and show rank, account, and team only. Metric values remain private.</p>
    </div>
    """


def _final_test_notice():
    return (
        '<div class="leaderboard-empty">'
        f"{_test_release_notice_html()}"
        "<p>The final test leaderboard is not available yet; it will publish after "
        "organizer finalization.</p>"
        "</div>"
    )


def leaderboard_view(selection):
    """Render one server-selected leaderboard surface without client paths."""

    if selection == VALIDATION_LEADERBOARD_LABEL:
        return (
            gr.update(value=_validation_leaderboard_heading()),
            gr.update(value=leaderboard_html()),
            gr.update(visible=True),
        )
    if selection != FINAL_TEST_LEADERBOARD_LABEL:
        raise gr.Error("Choose a listed leaderboard view.")
    if not TEST_PUBLIC_LEADERBOARD_ENABLED and not TEST_PROVISIONAL_LEADERBOARD_ENABLED:
        return (
            gr.update(value=_final_test_leaderboard_heading()),
            gr.update(value=_final_test_notice()),
            gr.update(visible=False),
        )
    if not TEST_PUBLIC_LEADERBOARD_ENABLED:
        try:
            projection = _load_provisional_test_projection()
            content = provisional_test_leaderboard_html(projection)
        except FinalLeaderboardError:
            return (
                gr.update(value=_provisional_test_leaderboard_heading()),
                gr.update(value=_final_test_notice()),
                gr.update(visible=False),
            )
        return (
            gr.update(value=_provisional_test_leaderboard_heading()),
            gr.update(value=content),
            gr.update(visible=True),
        )
    try:
        projection = _load_final_test_projection()
        content = final_test_leaderboard_html(projection)
    except FinalLeaderboardError:
        return (
            gr.update(value=_final_test_leaderboard_heading()),
            gr.update(value=_final_test_notice()),
            gr.update(visible=False),
        )
    return (
        gr.update(value=_final_test_leaderboard_heading()),
        gr.update(value=content),
        gr.update(visible=True),
    )


def evaluate_submission(
    file_obj, team, contact, submission_name, participant_names=None
):
    if not VALIDATION_SUBMISSIONS_ENABLED:
        raise gr.Error("Validation submissions are temporarily paused for maintenance.")
    if file_obj is None:
        raise gr.Error("Upload a JSONL submission file.")
    if not team.strip():
        raise gr.Error("Enter a team name.")
    if not contact.strip():
        raise gr.Error("Enter a contact email.")
    if not submission_name.strip():
        raise gr.Error("Enter a submission name.")

    try:
        participant_names = normalize_participant_names(participant_names)
        text = Path(file_obj.name).read_text(encoding="utf-8")
        rows = parse_submission_text(text)
        labels = _load_gold_rows()
        rows = expand_predictions(rows, labels)
        metrics = score_validation_predictions(rows, labels)
        message = _persist_submission(
            rows,
            team.strip(),
            contact.strip(),
            submission_name.strip(),
            metrics,
            participant_names=participant_names,
        )
    except SubmissionError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        raise gr.Error(f"Could not score submission: {exc}") from exc

    return gr.update(
        value={
            "answer_accuracy": metrics["answer_accuracy"],
            "evidence_exact_match": metrics["evidence_exact_match"],
            "evidence_f1": metrics["evidence_f1"],
            "joint_accuracy": metrics["joint_accuracy"],
            "examples": metrics["examples"],
            "message": message,
        },
        visible=True,
    )


def _legacy_validation_submitter(file_obj, metadata):
    return evaluate_submission(
        file_obj,
        metadata.get("team"),
        metadata.get("contact"),
        metadata.get("submission_name"),
        metadata.get("participant_names"),
    )


_TEST_HUB_API = HfApi(token=WRITE_TOKEN)
_PUBLIC_HUB_API = HfApi()
_SUBMISSION_SERVICE = SubmissionService(
    validation_submitter=_legacy_validation_submitter,
    validation_submissions_enabled=VALIDATION_SUBMISSIONS_ENABLED,
    test_store=HubTestStore(
        _TEST_HUB_API,
        repo_id=SUBMISSIONS_REPO_ID,
        release_config_path=TEST_DEPLOYMENT.release_config_path,
        gold_config_path=TEST_DEPLOYMENT.gold_config_path,
    ),
    test_config_loader=HubTestConfigLoader(
        _TEST_HUB_API,
        repo_id=SUBMISSIONS_REPO_ID,
        public_api=_PUBLIC_HUB_API,
        public_repo_id=PUBLIC_DATASET_REPO,
        task_manifest_path=TEST_TASKS_FILE,
        release_config_path=TEST_DEPLOYMENT.release_config_path,
        gold_config_path=TEST_DEPLOYMENT.gold_config_path,
        expected_policy=TEST_DEPLOYMENT.expected_policy,
        enabled=TEST_SUBMISSIONS_ENABLED and bool(WRITE_TOKEN),
    ),
)


def submit_for_split(split, file_obj, metadata, oauth_profile):
    try:
        return _SUBMISSION_SERVICE.submit_for_split(
            split, file_obj, metadata, oauth_profile
        )
    except SubmissionError as exc:
        raise gr.Error(str(exc)) from None


def history_for_identity(contact_email, oauth_profile):
    try:
        return _SUBMISSION_SERVICE.history_for_identity(contact_email, oauth_profile)
    except SubmissionError as exc:
        raise gr.Error(str(exc)) from None


def _selected_split(split_label):
    return {
        VALIDATION_SPLIT_LABEL: "validation",
        TEST_SPLIT_LABEL: "test",
    }.get(split_label, split_label)


def submit_predictions(
    split_label,
    file_obj,
    team,
    participant_names,
    contact,
    submission_name,
    oauth_profile: gr.OAuthProfile | None,
):
    response = submit_for_split(
        _selected_split(split_label),
        file_obj,
        {
            "team": team,
            "participant_names": participant_names,
            "contact": contact,
            "submission_name": submission_name,
        },
        oauth_profile,
    )
    if isinstance(response, dict) and response.get("__type__") == "update":
        return response
    return gr.update(value=response, visible=True)


def _masked_email(profile, contact_email):
    del contact_email
    try:
        data = dict(profile) if profile is not None else {}
    except (TypeError, ValueError):
        data = {}
    try:
        email = normalize_contact_email(data.get("email"))
    except TestPolicyError as exc:
        raise gr.Error(str(exc)) from None
    local, domain = email.rsplit("@", maxsplit=1)
    if not local or not domain:
        raise gr.Error("Enter a valid contact email for test submissions.")
    return f"{local[0]}***@{domain}"


def _test_history_html(attempts, masked_email):
    rows = []
    next_eligible = None
    for attempt in attempts:
        if attempt.get("next_eligible_at"):
            next_eligible = str(attempt["next_eligible_at"])
        number = int(attempt.get("attempt", 0))
        if number == 1:
            feedback = (
                "Joint Exact Accuracy "
                f"{_format_joint_metric(attempt.get('joint_accuracy'))}; "
                "Answer Exact Accuracy "
                f"{_format_metric(attempt.get('answer_accuracy', 0.0))}; "
                "Evidence F1 (macro) "
                f"{_format_metric(attempt.get('evidence_f1', 0.0))}"
            )
        else:
            feedback = "Score withheld until finalization"
        rows.append(
            "<tr>"
            f"<td>{number}</td>"
            f"<td>{html.escape(str(attempt.get('submission_name', '')))}</td>"
            f"<td>{html.escape(str(attempt.get('accepted_at', '')))}</td>"
            f"<td>{html.escape(str(attempt.get('receipt', '')))}</td>"
            f"<td>{html.escape(feedback)}</td>"
            "</tr>"
        )
    if not rows:
        rows.append('<tr><td colspan="5">No accepted test submissions yet.</td></tr>')
    remaining = max(0, 3 - len(attempts))
    cooldown = (
        "<p>Next distinct attempt eligible at "
        f"{html.escape(next_eligible)} UTC. Exact retries remain "
        "available.</p>"
        if next_eligible
        else ""
    )
    return f"""
    <p>Signed in as <strong>{html.escape(masked_email)}</strong>. {remaining} accepted attempts remaining.</p>
    {cooldown}
    <div class="leaderboard-table-wrap">
        <table aria-label="My test submissions">
            <thead>
                <tr>
                    <th scope="col">Attempt</th>
                    <th scope="col">Submission</th>
                    <th scope="col">Accepted (UTC)</th>
                    <th scope="col">Receipt</th>
                    <th scope="col">Feedback</th>
                </tr>
            </thead>
            <tbody>{"".join(rows)}</tbody>
        </table>
    </div>
    """


def my_test_submissions(contact_email, oauth_profile: gr.OAuthProfile | None):
    attempts = history_for_identity(contact_email, oauth_profile)
    return gr.update(
        value=_test_history_html(attempts, _masked_email(oauth_profile, contact_email)),
        visible=True,
    )


def _server_now():
    return dt.datetime.now(dt.timezone.utc)


def _test_release_state(
    now=None,
    *,
    deployment=None,
    submissions_enabled=None,
    write_token=None,
    authoritative_loader=None,
):
    """Return the participant-facing state from the validated server policy."""

    deployment = TEST_DEPLOYMENT if deployment is None else deployment
    submissions_enabled = (
        TEST_SUBMISSIONS_ENABLED
        if submissions_enabled is None
        else bool(submissions_enabled)
    )
    write_token = WRITE_TOKEN if write_token is None else write_token
    current = _normalized_utc(_server_now() if now is None else now)
    if current is None:
        return "unavailable", None
    try:
        opened, close_at = _validate_final_deployment(deployment)
        policy = TestReleasePolicy(
            release_id=deployment.release_id,
            task_manifest_sha256=deployment.task_manifest_sha256,
            gold_sha256=deployment.gold_sha256,
            open_at=opened,
            close_at=close_at,
            enabled=True,
            max_attempts=deployment.max_attempts,
        )
    except Exception:
        return "unavailable", None
    if current >= close_at:
        return "closed", close_at
    if not submissions_enabled or not str(write_token or "").strip():
        return "unavailable", close_at
    try:
        policy.require_open(current)
    except Exception:
        return "scheduled", close_at
    if authoritative_loader is None:
        authoritative_loader = getattr(
            globals().get("_SUBMISSION_SERVICE"), "test_config_loader", None
        )
    try:
        trusted = authoritative_loader(current)
    except Exception:
        return "unavailable", close_at
    if not isinstance(trusted, TrustedTestConfig) or trusted.policy != policy:
        return "unavailable", close_at
    return "open", close_at


def _test_ui_open() -> bool:
    state, _ = _test_release_state()
    return state == "open"


def _deadline_labels(close_at):
    """Render one exclusive UTC close as its inclusive AoE and UTC labels."""

    normalized = _normalized_utc(close_at)
    if normalized is None:
        raise ValueError("A timezone-aware test close instant is required.")
    anywhere_on_earth = dt.timezone(-dt.timedelta(hours=12), name="AoE")
    last_included = (normalized - dt.timedelta(seconds=1)).astimezone(anywhere_on_earth)

    def label(value, zone, *, twenty_four_hour=False):
        clock = (
            value.strftime("%H:%M:%S")
            if twenty_four_hour
            else value.strftime("%I:%M:%S %p").lstrip("0")
        )
        return f"{value.strftime('%B')} {value.day}, {value.year} at {clock} {zone}"

    return label(last_included, "Anywhere on Earth"), label(
        normalized, "UTC", twenty_four_hour=True
    )


def _countdown_text(close_at, now):
    """Return a deterministic, non-negative countdown for the exclusive close."""

    closed = _normalized_utc(close_at)
    current = _normalized_utc(now)
    if closed is None or current is None:
        raise ValueError("Timezone-aware countdown instants are required.")
    total_seconds = max(0, math.ceil((closed - current).total_seconds()))
    if total_seconds == 0:
        return "Test submissions are closed."
    days, remainder = divmod(total_seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, seconds = divmod(remainder, 60)

    def unit(value, singular):
        return f"{value} {singular if value == 1 else singular + 's'}"

    return (
        ", ".join(
            (
                unit(days, "day"),
                unit(hours, "hour"),
                unit(minutes, "minute"),
                unit(seconds, "second"),
            )
        )
        + " remaining"
    )


def _test_release_notice_html(
    now=None,
    *,
    deployment=None,
    submissions_enabled=None,
    write_token=None,
    authoritative_loader=None,
):
    """Render release availability and policy without claiming an unproven opening."""

    current = _server_now() if now is None else now
    state, close_at = _test_release_state(
        current,
        deployment=deployment,
        submissions_enabled=submissions_enabled,
        write_token=write_token,
        authoritative_loader=authoritative_loader,
    )
    if state == "open":
        availability = "Test submissions are open."
    elif state == "closed":
        availability = "Test submissions are closed."
    else:
        availability = (
            "Test submissions are not open yet. Organizers will announce activation "
            "after the private scoring key is installed and verified."
        )

    countdown = ""
    if close_at is not None:
        aoe_label, utc_label = _deadline_labels(close_at)
        close_iso = close_at.isoformat(timespec="seconds").replace("+00:00", "Z")
        anywhere_on_earth = dt.timezone(-dt.timedelta(hours=12), name="AoE")
        last_included_iso = (
            (close_at - dt.timedelta(seconds=1))
            .astimezone(anywhere_on_earth)
            .isoformat(timespec="seconds")
        )
        countdown_copy = _countdown_text(close_at, current)
        countdown_label = (
            f"Time remaining until test submissions close: {countdown_copy}"
        )
        countdown = (
            '<p class="test-deadline">'
            "The final accepted second is "
            f'<time datetime="{html.escape(last_included_iso, quote=True)}">'
            f"{html.escape(aoe_label)}</time>; the exclusive closing instant is "
            f'<time datetime="{html.escape(close_iso, quote=True)}">'
            f"{html.escape(utc_label)}</time>. "
            '<span class="test-countdown" role="timer" aria-live="off" '
            f'aria-label="{html.escape(countdown_label, quote=True)}" '
            f'data-docsem-close-at="{html.escape(close_iso, quote=True)}">'
            f"{html.escape(countdown_copy)}</span>"
            "</p>"
        )

    return (
        '<div class="test-release-notice">'
        "<p><strong>Test data released.</strong> The public test tasks and PDFs are "
        f'available in the <a href="{html.escape(PUBLIC_DATASET_URL, quote=True)}" '
        'target="_blank" rel="noopener">public dataset</a>. '
        '<span data-docsem-submission-status role="status" aria-live="polite" '
        f'aria-atomic="true">{html.escape(availability)}</span> '
        "Review the "
        f'<a href="{html.escape(WORKSHOP_URL, quote=True)}" target="_blank" '
        'rel="noopener">workshop rules</a> and '
        f'<a href="{html.escape(PARTICIPANT_GUIDE_URL, quote=True)}" '
        'target="_blank" rel="noopener">participant guide</a> before uploading.</p>'
        f"{countdown}"
        '<p class="test-policy"><strong>Test policy:</strong> Up to '
        "3 accepted test submissions per Hugging Face account, with at least six hours "
        "between distinct accepted attempts. Exact retries do not consume an attempt or "
        "reset the interval. Partial submissions are allowed; missing tasks count wrong "
        "against the full split. Use answer: null and evidence: [] to abstain. Attempt 1 metrics—Joint Exact "
        "Accuracy, Answer Exact Accuracy, and Evidence F1 (macro)—are private to that "
        "signed-in account. Attempts 2–3 are accepted with their metrics withheld. During the "
        "open window, provisional public ranks use only attempt 1 and display no metrics. "
        "After the window closes, the final ranking uses the best of all 3 eligible "
        "attempts.</p></div>"
    )


def split_ui(split_label):
    if split_label == TEST_SPLIT_LABEL:
        test_open = _test_ui_open()
        return (
            gr.update(
                value=(
                    "### Submit final test predictions\n"
                    "Sign in with Hugging Face (required). Attempts are keyed to your "
                    "immutable HF account subject; the typed contact remains available "
                    "for team communication but does not control identity or quota. Up to "
                    "three unique attempts are accepted, with at least six hours between "
                    "distinct attempts. Partial submissions are allowed: missing tasks "
                    "count wrong against the full denominator; use `answer: null` and "
                    "`evidence: []` to abstain.\n\n"
                    f"{_test_release_notice_html()}"
                )
            ),
            gr.update(visible=True),
            gr.update(
                value="Submit test predictions",
                interactive=test_open,
            ),
            gr.update(visible=True),
        )
    validation_copy = (
        "### Submit validation predictions\n"
        "Upload one JSON object per instance with `instance_id`, `answer`, and "
        f"`evidence`. Review the [participant guide]({PARTICIPANT_GUIDE_URL}) "
        "for the complete format and evaluation protocol."
    )
    if not VALIDATION_SUBMISSIONS_ENABLED:
        validation_copy = (
            "### Validation submissions paused for maintenance\n"
            "Validation submissions are temporarily paused for maintenance. "
            "Existing validation results remain readable below."
        )
    return (
        gr.update(value=validation_copy),
        gr.update(visible=True),
        gr.update(
            value="Validate and score", interactive=VALIDATION_SUBMISSIONS_ENABLED
        ),
        gr.update(visible=False),
    )


class PortalBlocks(gr.Blocks):
    @property
    def expects_oauth(self):
        # A real Space receives OAuth routes from its platform environment.
        # Local and CI imports stay offline and never invoke Gradio's mock login.
        return bool(os.getenv("SPACE_ID")) and super().expects_oauth


blocks_options = {
    "title": "DocInsights 2026 Shared Task: DocSem",
    "fill_width": True,
    "head": PORTAL_HEAD,
}
if GRADIO_MAJOR_VERSION < 6:
    blocks_options["css"] = PORTAL_CSS


with PortalBlocks(**blocks_options) as demo:
    gr.HTML(
        f"""
        <header id="portal-header">
            <div>
                <span class="portal-kicker">Workshop on Document Intelligence and Understanding</span>
                <h1>DocInsights 2026 Shared Task: DocSem</h1>
                <p class="portal-summary">Document-grounded quantitative reasoning with evidence attribution.</p>
                <p>Co-located with EMNLP 2026 in Budapest, Hungary. Beyond Plain Text: Bridging NLP and Document AI.</p>
            </div>
            <nav class="portal-links" aria-label="Shared task links">
                <a class="primary-link" href="{WORKSHOP_URL}" target="_blank" rel="noopener" aria-label="DocInsights shared task workshop page">
                    Workshop
                </a>
                <a href="{PUBLIC_DATASET_URL}" target="_blank" rel="noopener" aria-label="Public DocSem dataset">
                    Dataset
                </a>
                <a href="{SOURCE_REPO_URL}" target="_blank" rel="noopener" aria-label="Canonical GSM-SEM GitHub repository">
                    GitHub
                </a>
            </nav>
        </header>
        """
    )

    gr.HTML(
        f"""
        <section id="evaluation-notice" aria-labelledby="evaluation-notice-title">
            <h2 id="evaluation-notice-title">Dataset update and final evaluation</h2>
            <p>
                <strong>Use the August 31, 2026 training-data release.</strong>
                The training split was updated to correct seven annotation inconsistencies identified through community feedback. These changes affect only the training data; the task definition and data format are unchanged. Pull the
                <a href="https://huggingface.co/datasets/{PUBLIC_DATASET_REPO}" target="_blank" rel="noopener">latest version</a>.
            </p>
            <p>
                <strong>Validation ground truth refreshed September 3, 2026.</strong>
                Three organizer-only validation labels have now been corrected, most recently on September 3, 2026, following additional data review. All existing submissions were rescored, and the leaderboard now reflects the updated results. Public validation inputs, the task definition, and the data format are unchanged.
            </p>
            {_test_release_notice_html()}
        </section>
        """
    )

    with gr.Row(elem_id="split-controls"):
        split_selector = gr.Dropdown(
            choices=[VALIDATION_SPLIT_LABEL, TEST_SPLIT_LABEL],
            value=VALIDATION_SPLIT_LABEL,
            label="Evaluation split",
            interactive=True,
        )
        gr.LoginButton("Sign in with Hugging Face (required)")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The `gr.LogoutButton` component is deprecated.*",
                category=UserWarning,
            )
            gr.LogoutButton("Sign out")

    with gr.Group(elem_id="submission-panel"):
        initial_validation_copy = (
            "### Submit validation predictions\n"
            "Upload one JSON object per instance with `instance_id`, `answer`, and "
            f"`evidence`. Review the [participant guide]({PARTICIPANT_GUIDE_URL}) "
            "for the complete format and evaluation protocol."
            if VALIDATION_SUBMISSIONS_ENABLED
            else "### Validation submissions paused for maintenance\n"
            "Validation submissions are temporarily paused for maintenance. "
            "Existing validation results remain readable below."
        )
        submission_intro = gr.Markdown(initial_validation_copy)
        with gr.Row(elem_id="submission-fields"):
            team = gr.Textbox(label="Team", placeholder="example-team")
            participant_names = gr.Textbox(
                label="Participant name(s)",
                placeholder="A. Researcher, B. Researcher",
            )
            contact = gr.Textbox(
                label="Contact email",
                placeholder="lead@example.org",
            )
            submission_name = gr.Textbox(
                label="Submission name", placeholder="baseline-v1"
            )
        with gr.Row(elem_id="submission-actions"):
            file_input = gr.File(
                label="Submission file",
                file_types=[".jsonl", ".json"],
                scale=4,
                elem_id="submission-file",
            )
            with gr.Column(scale=1, min_width=210, elem_id="submission-side"):
                gr.Markdown(
                    "Accepted: `.jsonl` or `.json`  \n"
                    "Participant details and validation labels remain organizer-only."
                )
                submit = gr.Button(
                    "Validate and score",
                    variant="primary",
                    interactive=VALIDATION_SUBMISSIONS_ENABLED,
                    elem_id="submit-button",
                )
        result = gr.JSON(
            label="Submission score",
            visible=False,
            height=160,
            elem_id="score-output",
        )

    with gr.Accordion(
        "How metrics are computed", open=False, elem_id="metric-explanation"
    ):
        gr.Markdown(
            "**Answer Exact Accuracy** is the mean exact normalized answer match.\n\n"
            "For each task, evidence precision and recall use set overlap, and their "
            "harmonic mean gives the task-level evidence F1. **Evidence F1 (macro)** "
            "is then macro-averaged across tasks. Partial evidence earns partial credit.\n\n"
            "**Joint Exact Accuracy** is the mean of `answer exact AND evidence set "
            "exact` on the same task. It is therefore never greater than Answer Exact "
            "Accuracy or Evidence Exact Match. It does not AND the two aggregate "
            "percentages and does not use evidence F1.\n\n"
            "Ranking order is Joint Exact Accuracy, Answer Exact Accuracy, Evidence F1, "
            "accepted time and stable submission ID."
        )

    with gr.Accordion("Cite this dataset", open=False, elem_id="dataset-citation"):
        gr.Markdown(
            f"Citation source: [GSM-SEM on arXiv]({DATASET_CITATION_URL}). "
            "Use the copy control on the BibTeX block below."
        )
        gr.Code(
            value=DATASET_BIBTEX,
            language=None,
            lines=10,
            label="BibTeX citation",
            interactive=False,
            elem_id="dataset-citation-code",
        )

    with gr.Group(visible=False, elem_id="test-history-section") as test_history_group:
        gr.Markdown(
            "### My test submissions\n"
            "Sign in with Hugging Face to retrieve receipts for your immutable account. "
            "The next eligible UTC time is shown after an accepted attempt."
        )
        refresh_history = gr.Button("Refresh my submissions", variant="secondary")
        test_history = gr.HTML(
            value=(
                "<p>Sign in with Hugging Face (required) to retrieve your test "
                "receipts.</p>"
            )
        )

    submit.click(
        submit_predictions,
        inputs=[
            split_selector,
            file_input,
            team,
            participant_names,
            contact,
            submission_name,
        ],
        outputs=result,
        api_name="submit_predictions",
        # The service already returns JSON-safe receipt data. Gradio 4.42 wraps
        # JSON update values in JsonData, which its SSE serializer cannot encode.
        postprocess=False,
    )
    refresh_history.click(
        my_test_submissions,
        inputs=contact,
        outputs=test_history,
        api_name="my_test_submissions",
    )
    split_selector.change(
        split_ui,
        inputs=split_selector,
        outputs=[submission_intro, contact, submit, test_history_group],
        api_name="select_split",
    )
    with gr.Column(elem_id="leaderboard-section"):
        initial_leaderboard_selection = (
            FINAL_TEST_LEADERBOARD_LABEL
            if TEST_PUBLIC_LEADERBOARD_ENABLED
            else VALIDATION_LEADERBOARD_LABEL
        )
        leaderboard_selector = gr.Dropdown(
            choices=[
                VALIDATION_LEADERBOARD_LABEL,
                FINAL_TEST_LEADERBOARD_LABEL,
            ],
            value=initial_leaderboard_selection,
            label="Leaderboard view",
            interactive=True,
        )
        with gr.Row(elem_id="leaderboard-heading"):
            leaderboard_heading = gr.HTML(
                value=(
                    _final_test_leaderboard_heading()
                    if TEST_PUBLIC_LEADERBOARD_ENABLED
                    else _validation_leaderboard_heading()
                ),
            )
            refresh = gr.Button(
                "Refresh results",
                variant="secondary",
                scale=0,
                min_width=170,
                elem_id="refresh-button",
            )
        leaderboard = gr.HTML(
            value=(
                '<div class="leaderboard-empty">Loading final test results...</div>'
                if TEST_PUBLIC_LEADERBOARD_ENABLED
                else leaderboard_html()
            ),
            elem_id="leaderboard-table",
        )
        demo.load(
            leaderboard_view,
            inputs=leaderboard_selector,
            outputs=[leaderboard_heading, leaderboard, refresh],
            api_name=False,
            show_api=False,
        )
        leaderboard_selector.change(
            leaderboard_view,
            inputs=leaderboard_selector,
            outputs=[leaderboard_heading, leaderboard, refresh],
            api_name=False,
            show_api=False,
        )
        refresh.click(
            leaderboard_view,
            inputs=leaderboard_selector,
            outputs=[leaderboard_heading, leaderboard, refresh],
            api_name=False,
            show_api=False,
        )


if __name__ == "__main__":
    launch_options = {"css": PORTAL_CSS} if GRADIO_MAJOR_VERSION >= 6 else {}
    demo.launch(**launch_options)
