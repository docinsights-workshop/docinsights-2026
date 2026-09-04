import datetime as dt
import html
import json
import os
import threading
from pathlib import Path

import gradio as gr
from huggingface_hub import HfApi, hf_hub_download, upload_file
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

from scoring import (
    SubmissionError,
    leaderboard_identity,
    leaderboard_row,
    load_jsonl_text,
    normalize_participant_names,
    parse_submission_text,
    rank_leaderboard,
    safe_slug,
    score_predictions,
)
from submission_service import HubTestConfigLoader, SubmissionService
from test_store import HubTestStore


PUBLIC_DATASET_REPO = os.getenv("PUBLIC_DATASET_REPO", "amitbcp/docinsights-2026-shared-task-data")
WORKSHOP_URL = os.getenv(
    "WORKSHOP_URL",
    "https://docinsights-workshop.github.io/docinsights-2026/shared-task/",
)
SOURCE_REPO_URL = os.getenv("SOURCE_REPO_URL", "https://github.com/oracle-samples/gsm-sem")
PARTICIPANT_GUIDE_URL = os.getenv(
    "PARTICIPANT_GUIDE_URL",
    "https://github.com/oracle-samples/gsm-sem/blob/main/docsem/PARTICIPANT_INSTRUCTIONS.md",
)
GOLD_REPO_ID = os.getenv("GOLD_REPO_ID", "amitbcp/docinsights-2026-shared-task-submissions")
GOLD_FILE = os.getenv("GOLD_FILE", "private/val_labels.jsonl")
SUBMISSIONS_REPO_ID = os.getenv("SUBMISSIONS_REPO_ID", GOLD_REPO_ID)
WRITE_TOKEN = os.getenv("HF_WRITE_TOKEN") or os.getenv("HF_TOKEN")
GRADIO_MAJOR_VERSION = int(gr.__version__.split(".", maxsplit=1)[0])
LEADERBOARD_LOCK = threading.Lock()
TEST_SUBMISSIONS_ENABLED = os.getenv("TEST_SUBMISSIONS_ENABLED", "false").strip().casefold() in {
    "1",
    "true",
    "yes",
}
TEST_TASKS_FILE = os.getenv("TEST_TASKS_FILE", "test/tasks.jsonl")

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
    max-width: 1540px !important;
    height: 100vh !important;
    height: 100dvh !important;
    max-height: 100% !important;
    margin: 0 auto !important;
    padding: 24px 30px 36px !important;
    overflow-y: auto !important;
    overscroll-behavior-y: contain;
    -webkit-overflow-scrolling: touch;
    background: var(--docsem-page);
    color: var(--docsem-ink);
    font-size: 16px;
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

#evaluation-notice p + p {
    border-left: 1px solid #bdddd8;
    padding-left: 24px;
}

#evaluation-notice a {
    color: var(--docsem-teal);
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
    height: 112px !important;
}

#submission-file > div {
    min-height: 110px !important;
}

#submission-file {
    font-size: 15px;
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
    .gradio-container {
        padding: 16px 14px 26px !important;
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

    #evaluation-notice p + p {
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


def _read_hub_file(repo_id, filename, token=None, force_download=False):
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        token=token,
        force_download=force_download,
    )
    return Path(path).read_text(encoding="utf-8")


def _load_gold_rows():
    return load_jsonl_text(_read_hub_file(GOLD_REPO_ID, GOLD_FILE, token=WRITE_TOKEN))


def _persist_submission(rows, team, contact, submission_name, metrics, participant_names=None):
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
        tmp_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
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
            item for item in rank_leaderboard(rows) if leaderboard_identity(item) == identity
        )
        return latest["attempts"]


def _format_metric(value):
    return f"{float(value) * 100:.2f}%"


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
                <td class="leaderboard-metric">{_format_metric(row.get("answer_accuracy", 0.0))}</td>
                <td class="leaderboard-metric">{_format_metric(row.get("evidence_f1", 0.0))}</td>
                <td class="leaderboard-date">{html.escape(_format_timestamp(row.get("submitted_at", "")))}</td>
            </tr>
            """
        )

    if not body_rows:
        body_rows.append(
            '<tr><td class="leaderboard-empty" colspan="7">No scored submissions yet.</td></tr>'
        )

    return f"""
    <div class="leaderboard-table-wrap">
        <table aria-label="DocSem validation leaderboard">
            <colgroup>
                <col style="width: 6%">
                <col style="width: 20%">
                <col style="width: 22%">
                <col style="width: 10%">
                <col style="width: 15%">
                <col style="width: 12%">
                <col style="width: 15%">
            </colgroup>
            <thead>
                <tr>
                    <th class="leaderboard-rank" scope="col">Rank</th>
                    <th scope="col">Team</th>
                    <th scope="col">Latest submission</th>
                    <th class="leaderboard-attempts" scope="col">Attempts</th>
                    <th class="leaderboard-metric" scope="col">Answer accuracy</th>
                    <th class="leaderboard-metric" scope="col">Evidence F1</th>
                    <th scope="col">Submitted (UTC)</th>
                </tr>
            </thead>
            <tbody>
                {''.join(body_rows)}
            </tbody>
        </table>
    </div>
    """


def evaluate_submission(file_obj, team, contact, submission_name, participant_names=None):
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
        metrics = score_predictions(rows, labels)
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
    test_store=HubTestStore(_TEST_HUB_API, repo_id=SUBMISSIONS_REPO_ID),
    test_config_loader=HubTestConfigLoader(
        _TEST_HUB_API,
        repo_id=SUBMISSIONS_REPO_ID,
        public_api=_PUBLIC_HUB_API,
        public_repo_id=PUBLIC_DATASET_REPO,
        task_manifest_path=TEST_TASKS_FILE,
        enabled=TEST_SUBMISSIONS_ENABLED and bool(WRITE_TOKEN),
    ),
)


def submit_for_split(split, file_obj, metadata, oauth_profile):
    try:
        return _SUBMISSION_SERVICE.submit_for_split(split, file_obj, metadata, oauth_profile)
    except SubmissionError as exc:
        raise gr.Error(str(exc)) from None


def history_for_oauth(oauth_profile):
    try:
        return _SUBMISSION_SERVICE.history_for_oauth(oauth_profile)
    except SubmissionError as exc:
        raise gr.Error(str(exc)) from None


blocks_options = {
    "title": "DocInsights 2026 Shared Task: DocSem",
    "fill_width": True,
}
if GRADIO_MAJOR_VERSION < 6:
    blocks_options["css"] = PORTAL_CSS


with gr.Blocks(**blocks_options) as demo:
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
                <a href="https://huggingface.co/datasets/{PUBLIC_DATASET_REPO}" target="_blank" rel="noopener" aria-label="Public DocSem dataset">
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
            <p>
                <strong>Final rankings will use a held-out test set.</strong>
                It will be released five days before the September 10, 2026 final submission deadline.
                Participants will be notified when it is available and asked to submit test-set results;
                those results will determine the final leaderboard.
            </p>
        </section>
        """
    )

    with gr.Group(elem_id="submission-panel"):
        gr.HTML(
            f"""
            <h2>Submit validation predictions</h2>
            <p class="submission-note">
                Upload one JSON object per instance with <code>instance_id</code>,
                <code>answer</code>, and <code>evidence</code>.
                Review the <a href="{PARTICIPANT_GUIDE_URL}" target="_blank" rel="noopener">participant guide</a>
                for the complete format and evaluation protocol.
            </p>
            """
        )
        with gr.Row(elem_id="submission-fields"):
            team = gr.Textbox(label="Team", placeholder="example-team")
            participant_names = gr.Textbox(
                label="Participant name(s)",
                placeholder="A. Researcher, B. Researcher",
            )
            contact = gr.Textbox(label="Contact email", placeholder="lead@example.org")
            submission_name = gr.Textbox(label="Submission name", placeholder="baseline-v1")
        with gr.Row(elem_id="submission-actions"):
            file_input = gr.File(
                label="Submission file",
                file_types=[".jsonl", ".json"],
                height=112,
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
                    elem_id="submit-button",
                )
        result = gr.JSON(
            label="Submission score",
            visible=False,
            height=160,
            elem_id="score-output",
        )

    submit.click(
        evaluate_submission,
        inputs=[file_input, team, contact, submission_name, participant_names],
        outputs=result,
    )

    with gr.Column(elem_id="leaderboard-section"):
        with gr.Row(elem_id="leaderboard-heading"):
            gr.HTML(
                """
                <div>
                    <h2>Validation leaderboard</h2>
                    <p>Provisional validation results from each team's latest attempt. Ranked by answer accuracy, then evidence F1. Leaderboard refreshed September 3, 2026 after the organizer-only ground-truth correction; all existing submissions were rescored. Final standings will use the held-out test set.</p>
                </div>
                """
            )
            refresh = gr.Button(
                "Refresh results",
                variant="secondary",
                scale=0,
                min_width=170,
                elem_id="refresh-button",
            )
        leaderboard = gr.HTML(
            value=leaderboard_html,
            elem_id="leaderboard-table",
        )
        refresh.click(leaderboard_html, inputs=None, outputs=leaderboard)


if __name__ == "__main__":
    launch_options = {"css": PORTAL_CSS} if GRADIO_MAJOR_VERSION >= 6 else {}
    demo.launch(**launch_options)
