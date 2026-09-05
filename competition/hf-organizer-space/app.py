"""Private, read-only organizer view for the DocSem held-out test ledger.

The Space never loads private state while its Gradio component tree is built.
An organizer must explicitly refresh; that server callback resolves the current
private-dataset HEAD and immediately pins every subsequent read to the returned
40-hex commit SHA.
"""

from __future__ import annotations

import csv
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import gradio as gr
from huggingface_hub import HfApi

from organizer_data import (
    AuditReport,
    OrganizerDataError,
    OrganizerSnapshot,
    load_snapshot,
    organizer_rows,
    verify_snapshot,
)


MAX_EXPORT_BYTES = 8 * 1024 * 1024
MAX_FILTER_CHARACTERS = 512
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_REPOSITORY = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}/[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z"
)

BEST_CHOICES = ("all", "selected", "not-selected")
EXCLUDED_CHOICES = ("all", "excluded", "eligible")
ADJUDICATION_CHOICES = ("all", "has", "none")

TABLE_FIELDS = (
    "submission_id",
    "attempt_number",
    "selected_best",
    "excluded",
    "account_excluded",
    "attempt_excluded",
    "exclusion_count",
    "adjudication_count",
    "account_key",
    "hf_subject",
    "hf_username",
    "verified_email",
    "team",
    "participant_names",
    "submission_name",
    "submitted_at",
    "release_id",
    "task_manifest_sha256",
    "gold_sha256",
    "scoring_private_revision",
    "scoring_public_revision",
    "answer_accuracy",
    "evidence_f1",
    "evidence_exact_match",
    "examples",
)

TABLE_HEADERS = (
    "Submission ID",
    "Attempt",
    "Selected best",
    "Excluded",
    "Account excluded",
    "Attempt excluded",
    "Exclusions",
    "Adjudications",
    "Account key",
    "HF subject",
    "HF username",
    "Verified email",
    "Team",
    "Participant names",
    "Submission name",
    "Submitted (UTC)",
    "Release",
    "Task manifest SHA-256",
    "Gold SHA-256",
    "Evaluator/private revision",
    "Public scoring revision",
    "Answer accuracy",
    "Evidence F1",
    "Evidence exact match",
    "Examples",
)

EXPORT_FIELDS = TABLE_FIELDS


class OrganizerAppError(RuntimeError):
    """Value-free failure at the private organizer UI boundary."""


@dataclass(frozen=True, repr=False)
class OrganizerConfig:
    repo_id: str
    token: str = field(repr=False, compare=False)

    def __repr__(self) -> str:
        return "OrganizerConfig(configured=True)"


@dataclass(frozen=True, repr=False)
class OrganizerViewState:
    revision: str
    snapshot: OrganizerSnapshot = field(repr=False, compare=False)
    audit: AuditReport
    rows: tuple[dict, ...] = field(repr=False, compare=False)

    def __repr__(self) -> str:
        return (
            "OrganizerViewState("
            f"revision={self.revision!r}, row_count={len(self.rows)}, "
            f"integrity={self.audit.valid!r})"
        )


def load_organizer_config(
    environment: Mapping[str, object] | None = None,
) -> OrganizerConfig:
    """Require bounded server-only credentials without revealing bad values."""

    environment = os.environ if environment is None else environment
    token_value = environment.get("ORGANIZER_READ_TOKEN")
    repo_value = environment.get("PRIVATE_REPO_ID")
    token = token_value.strip() if isinstance(token_value, str) else ""
    repo_id = repo_value.strip() if isinstance(repo_value, str) else ""
    if not token or len(token) > 4096 or _REPOSITORY.fullmatch(repo_id) is None:
        raise OrganizerAppError("Organizer Space configuration is unavailable.")
    return OrganizerConfig(repo_id=repo_id, token=token)


def refresh_snapshot(
    config: OrganizerConfig,
    *,
    api=None,
    snapshot_loader=load_snapshot,
) -> tuple[OrganizerViewState, str]:
    """Resolve private HEAD once, then reconstruct and verify that exact SHA."""

    if not isinstance(config, OrganizerConfig):
        raise OrganizerAppError("Organizer snapshot is unavailable.")
    try:
        hub = api if api is not None else HfApi(token=config.token)
        info = hub.repo_info(
            config.repo_id,
            repo_type="dataset",
            token=config.token,
        )
        revision = getattr(info, "sha", None)
        if (
            getattr(info, "private", None) is not True
            or not isinstance(revision, str)
            or _REVISION.fullmatch(revision) is None
        ):
            raise OrganizerAppError("Organizer snapshot is unavailable.")
        snapshot = snapshot_loader(
            config.repo_id,
            revision,
            config.token,
            api=hub,
        )
        if snapshot.repo_id != config.repo_id or snapshot.revision != revision:
            raise OrganizerAppError("Organizer snapshot is unavailable.")
        audit = verify_snapshot(snapshot)
        if not audit.valid:
            raise OrganizerAppError("Organizer snapshot failed integrity verification.")
        rows = tuple(organizer_rows(snapshot))
        state = OrganizerViewState(
            revision=revision,
            snapshot=snapshot,
            audit=audit,
            rows=rows,
        )
        return state, _integrity_summary(state)
    except OrganizerAppError:
        raise
    except OrganizerDataError as exc:
        if str(exc) == "Organizer snapshot failed integrity verification.":
            raise OrganizerAppError(str(exc)) from None
        raise OrganizerAppError("Organizer snapshot is unavailable.") from None
    except Exception:
        raise OrganizerAppError("Organizer snapshot is unavailable.") from None


def filter_rows(
    rows: Sequence[Mapping],
    *,
    account: object = "",
    team: object = "",
    best: object = "all",
    excluded: object = "all",
    adjudication: object = "all",
) -> list[dict]:
    """Apply bounded, deterministic organizer filters to verified row shapes."""

    account_query = _filter_text(account)
    team_query = _filter_text(team)
    if (
        best not in BEST_CHOICES
        or excluded not in EXCLUDED_CHOICES
        or adjudication not in ADJUDICATION_CHOICES
    ):
        raise OrganizerAppError("Organizer filter is invalid.")

    result = []
    for source in rows:
        if not isinstance(source, Mapping):
            raise OrganizerAppError("Organizer filter is invalid.")
        row = dict(source)
        account_haystack = "\n".join(
            str(row.get(name, "")).casefold()
            for name in (
                "account_key",
                "hf_subject",
                "hf_username",
                "verified_email",
                "participant_names",
            )
        )
        if account_query and account_query not in account_haystack:
            continue
        if team_query and team_query not in str(row.get("team", "")).casefold():
            continue
        if best == "selected" and row.get("selected_best") is not True:
            continue
        if best == "not-selected" and row.get("selected_best") is not False:
            continue
        if excluded == "excluded" and row.get("excluded") is not True:
            continue
        if excluded == "eligible" and row.get("excluded") is not False:
            continue
        has_adjudication = int(row.get("adjudication_count", 0)) > 0
        if adjudication == "has" and not has_adjudication:
            continue
        if adjudication == "none" and has_adjudication:
            continue
        result.append(row)
    return result


def attempt_detail(state: OrganizerViewState, submission_id: object) -> dict:
    """Return allowlisted per-example metrics only after a server-side selection."""

    rows = _verified_rows(state)
    if not isinstance(submission_id, str) or not submission_id:
        raise OrganizerAppError("Organizer attempt detail is unavailable.")
    match = next(
        (row for row in rows if row.get("submission_id") == submission_id),
        None,
    )
    if match is None:
        raise OrganizerAppError("Organizer attempt detail is unavailable.")
    details = []
    for item in match.get("per_example", ()):
        details.append(
            {
                "instance_id": item["instance_id"],
                "answer_exact_match": item["answer_exact_match"],
                "evidence_exact_match": item["evidence_exact_match"],
                "evidence_f1": item["evidence_f1"],
            }
        )
    exclusions = []
    for loaded in state.snapshot.exclusions:
        record = loaded.value
        if record.get("account_key") != match["account_key"]:
            continue
        exclusions.append(
            {
                "record_id": record["record_id"],
                "created_at": record["created_at"],
                "reason_code": record["reason_code"],
            }
        )
    adjudications = []
    for loaded in state.snapshot.adjudications:
        record = loaded.value
        if record.get("account_key") != match["account_key"]:
            continue
        governed_submission = record.get("submission_id")
        if governed_submission is not None and governed_submission != submission_id:
            continue
        adjudication = {
            "record_id": record["record_id"],
            "created_at": record["created_at"],
            "action": record["action"],
            "reason_code": record["reason_code"],
        }
        if governed_submission is not None:
            adjudication["submission_id"] = governed_submission
        adjudications.append(adjudication)
    exclusions.sort(key=lambda item: (item["created_at"], item["record_id"]))
    adjudications.sort(key=lambda item: (item["created_at"], item["record_id"]))
    return {
        "submission_id": match["submission_id"],
        "attempt_number": match["attempt_number"],
        "selected_best": match["selected_best"],
        "excluded": match["excluded"],
        "per_example": details,
        "exclusions": exclusions,
        "adjudications": adjudications,
    }


def export_csv(
    state: OrganizerViewState,
    rows: Sequence[Mapping],
    *,
    directory: str | os.PathLike | None = None,
    max_bytes: int = MAX_EXPORT_BYTES,
) -> str:
    """Create one permission-restricted CSV from allowlisted verified fields."""

    authoritative = _verified_rows(state)
    selected = _validated_export_selection(authoritative, rows)
    if (
        not isinstance(max_bytes, int)
        or isinstance(max_bytes, bool)
        or not 1 <= max_bytes <= MAX_EXPORT_BYTES
    ):
        raise OrganizerAppError("Organizer export is unavailable.")

    export_root = _export_root(directory)
    path = None
    descriptor = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix="docsem-organizer-",
            suffix=".csv",
            dir=str(export_root),
        )
        path = Path(name)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            descriptor = None
            writer = csv.writer(stream, lineterminator="\n")
            release = state.snapshot.release
            evaluator_revisions = sorted(
                {str(row["scoring_private_revision"]) for row in authoritative}
            )
            _write_csv_row(
                writer, ["DocSem organizer export", "verified pinned snapshot"]
            )
            _write_csv_row(writer, ["repository_id", state.snapshot.repo_id])
            _write_csv_row(writer, ["repository_revision", state.revision])
            _write_csv_row(writer, ["release_id", release["release_id"]])
            _write_csv_row(
                writer, ["task_manifest_sha256", release["task_manifest_sha256"]]
            )
            _write_csv_row(writer, ["gold_sha256", release["gold_sha256"]])
            _write_csv_row(
                writer, ["evaluator_revisions", ";".join(evaluator_revisions)]
            )
            _write_csv_row(writer, ["attempts_exported", len(selected)])
            _write_csv_row(writer, [])
            _write_csv_row(writer, EXPORT_FIELDS)
            for row in selected:
                _write_csv_row(writer, [row.get(name, "") for name in EXPORT_FIELDS])
        if path.stat().st_size > max_bytes:
            raise OrganizerAppError("Organizer export is unavailable.")
        path.chmod(0o600)
        return str(path)
    except OrganizerAppError:
        if descriptor is not None:
            os.close(descriptor)
        if path is not None:
            path.unlink(missing_ok=True)
        raise
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        if path is not None:
            path.unlink(missing_ok=True)
        raise OrganizerAppError("Organizer export is unavailable.") from None


def build_app(
    *,
    environment: Mapping[str, object] | None = None,
    api=None,
) -> gr.Blocks:
    """Build an empty read-only UI; no private repository read occurs here."""

    config = load_organizer_config(environment)

    def handle_refresh():
        try:
            state, summary = refresh_snapshot(config, api=api)
            filtered = filter_rows(state.rows)
            return (
                state,
                summary,
                _table_values(filtered),
                gr.update(
                    choices=_attempt_choices(filtered),
                    value=None,
                    interactive=True,
                ),
                f"Loaded {len(filtered)} verified attempts.",
            )
        except OrganizerAppError as exc:
            raise gr.Error(str(exc)) from None

    def handle_filters(state, account, team, best, excluded, adjudication):
        try:
            verified = _verified_rows(state)
            filtered = filter_rows(
                verified,
                account=account,
                team=team,
                best=best,
                excluded=excluded,
                adjudication=adjudication,
            )
            return (
                _table_values(filtered),
                gr.update(choices=_attempt_choices(filtered), value=None),
                f"Showing {len(filtered)} of {len(verified)} verified attempts.",
            )
        except OrganizerAppError as exc:
            raise gr.Error(str(exc)) from None

    def handle_detail(state, submission_id):
        try:
            return gr.update(value=attempt_detail(state, submission_id), visible=True)
        except OrganizerAppError as exc:
            raise gr.Error(str(exc)) from None

    def handle_export(state, account, team, best, excluded, adjudication):
        try:
            verified = _verified_rows(state)
            selected = filter_rows(
                verified,
                account=account,
                team=team,
                best=best,
                excluded=excluded,
                adjudication=adjudication,
            )
            return export_csv(state, selected)
        except OrganizerAppError as exc:
            raise gr.Error(str(exc)) from None

    with gr.Blocks(
        title="DocSem organizer test dashboard",
        fill_width=True,
    ) as demo:
        state = gr.State(value=None)
        gr.Markdown(
            "# DocSem organizer test dashboard\n"
            "Private, read-only inspection of one verified test-ledger snapshot."
        )
        with gr.Row():
            refresh = gr.Button("Refresh verified snapshot", variant="primary")
            status = gr.Markdown("No snapshot loaded.")
        integrity = gr.Markdown("### Release integrity\nNo snapshot loaded.")

        with gr.Row():
            account_filter = gr.Textbox(
                label="Account / username / verified email",
                placeholder="Search private account identity",
            )
            team_filter = gr.Textbox(label="Team", placeholder="Filter by team")
        with gr.Row():
            best_filter = gr.Dropdown(
                choices=list(BEST_CHOICES),
                value="all",
                label="Best-attempt marker",
            )
            excluded_filter = gr.Dropdown(
                choices=list(EXCLUDED_CHOICES),
                value="all",
                label="Exclusion status",
            )
            adjudication_filter = gr.Dropdown(
                choices=list(ADJUDICATION_CHOICES),
                value="all",
                label="Adjudication status",
            )
            apply_filters = gr.Button("Apply filters", variant="secondary")

        attempts = gr.Dataframe(
            headers=list(TABLE_HEADERS),
            datatype=["str"] * len(TABLE_HEADERS),
            value=[],
            interactive=False,
            label="Verified test attempts",
            wrap=True,
        )
        with gr.Row():
            attempt_selector = gr.Dropdown(
                choices=[],
                value=None,
                label="Attempt details",
                interactive=False,
            )
            show_detail = gr.Button("Load per-example metrics", variant="secondary")
        detail = gr.JSON(
            label="Selected per-example metrics", value=None, visible=False
        )

        with gr.Row():
            create_export = gr.Button("Export filtered audit CSV", variant="secondary")
            download = gr.File(label="Restricted organizer export", interactive=False)

        refresh.click(
            handle_refresh,
            inputs=None,
            outputs=[state, integrity, attempts, attempt_selector, status],
            api_name=False,
        )
        apply_filters.click(
            handle_filters,
            inputs=[
                state,
                account_filter,
                team_filter,
                best_filter,
                excluded_filter,
                adjudication_filter,
            ],
            outputs=[attempts, attempt_selector, status],
            api_name=False,
        )
        show_detail.click(
            handle_detail,
            inputs=[state, attempt_selector],
            outputs=detail,
            api_name=False,
        )
        create_export.click(
            handle_export,
            inputs=[
                state,
                account_filter,
                team_filter,
                best_filter,
                excluded_filter,
                adjudication_filter,
            ],
            outputs=download,
            api_name=False,
        )
    return demo


def _filter_text(value: object) -> str:
    if value is None:
        return ""
    if not isinstance(value, str) or len(value) > MAX_FILTER_CHARACTERS:
        raise OrganizerAppError("Organizer filter is invalid.")
    return value.strip().casefold()


def _verified_rows(state: OrganizerViewState) -> tuple[dict, ...]:
    if not isinstance(state, OrganizerViewState) or state.audit.valid is not True:
        raise OrganizerAppError("Organizer snapshot is unavailable.")
    if state.snapshot.revision != state.revision:
        raise OrganizerAppError("Organizer snapshot is unavailable.")
    fresh_audit = verify_snapshot(state.snapshot)
    if not fresh_audit.valid or fresh_audit != state.audit:
        raise OrganizerAppError("Organizer snapshot failed integrity verification.")
    fresh_rows = tuple(organizer_rows(state.snapshot))
    if fresh_rows != state.rows:
        raise OrganizerAppError("Organizer snapshot failed integrity verification.")
    return fresh_rows


def _integrity_summary(state: OrganizerViewState) -> str:
    release = state.snapshot.release
    private_revisions = sorted(
        {str(row["scoring_private_revision"]) for row in state.rows}
    )
    evaluator_summary = ", ".join(private_revisions) if private_revisions else "none"
    return (
        "### Release integrity\n"
        f"**Integrity: PASS** — {state.audit.attempt_count} attempts across "
        f"{state.audit.account_count} accounts; {state.audit.exclusion_count} exclusions; "
        f"{state.audit.adjudication_count} adjudications.  \n"
        f"Pinned repository revision: `{state.revision}`  \n"
        f"Release: `{release['release_id']}`  \n"
        f"Task manifest SHA-256: `{release['task_manifest_sha256']}`  \n"
        f"Gold SHA-256: `{release['gold_sha256']}`  \n"
        f"Evaluator/private revision(s): `{evaluator_summary}`"
    )


def _table_values(rows: Sequence[Mapping]) -> list[list[object]]:
    return [[row.get(field, "") for field in TABLE_FIELDS] for row in rows]


def _attempt_choices(rows: Sequence[Mapping]) -> list[tuple[str, str]]:
    return [
        (
            f"{row.get('hf_username', '')} — attempt {row.get('attempt_number', '')} — "
            f"{row.get('submission_name', '')}",
            str(row.get("submission_id", "")),
        )
        for row in rows
    ]


def _validated_export_selection(
    authoritative: Sequence[Mapping],
    requested: Sequence[Mapping],
) -> list[dict]:
    by_submission = {row["submission_id"]: dict(row) for row in authoritative}
    selected = []
    seen = set()
    for supplied in requested:
        if not isinstance(supplied, Mapping):
            raise OrganizerAppError("Organizer export is unavailable.")
        submission_id = supplied.get("submission_id")
        expected = by_submission.get(submission_id)
        if expected is None or dict(supplied) != expected or submission_id in seen:
            raise OrganizerAppError("Organizer export is unavailable.")
        seen.add(submission_id)
        selected.append(expected)
    return selected


def _safe_csv_value(value: object) -> object:
    """Neutralize participant-controlled spreadsheet formula prefixes."""

    if isinstance(value, str) and value.lstrip().startswith(
        ("=", "+", "-", "@", "\t", "\r", "\n")
    ):
        return "'" + value
    return value


def _write_csv_row(writer: csv.writer, values: Sequence[object]) -> None:
    """Apply spreadsheet-formula safety to every string cell in a CSV row."""

    writer.writerow([_safe_csv_value(value) for value in values])


def _export_root(directory: str | os.PathLike | None) -> Path:
    try:
        if directory is None:
            root = Path(tempfile.mkdtemp(prefix="docsem-organizer-export-"))
        else:
            root = Path(directory)
            if root.is_symlink() or not root.is_dir():
                raise OrganizerAppError("Organizer export is unavailable.")
        root.chmod(0o700)
        return root
    except OrganizerAppError:
        raise
    except Exception:
        raise OrganizerAppError("Organizer export is unavailable.") from None


if __name__ == "__main__":
    build_app().launch()
