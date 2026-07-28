import datetime as dt
import json
import os
from pathlib import Path

import gradio as gr
from huggingface_hub import hf_hub_download, upload_file
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

from scoring import (
    SubmissionError,
    leaderboard_row,
    load_jsonl_text,
    parse_submission_text,
    safe_slug,
    score_predictions,
)


PUBLIC_DATASET_REPO = os.getenv("PUBLIC_DATASET_REPO", "amitbcp/docinsights-2026-shared-task-data")
WORKSHOP_URL = os.getenv(
    "WORKSHOP_URL",
    "https://docinsights-workshop.github.io/docinsights-2026/",
)
GOLD_REPO_ID = os.getenv("GOLD_REPO_ID", "amitbcp/docinsights-2026-shared-task-submissions")
GOLD_FILE = os.getenv("GOLD_FILE", "private/val_labels.jsonl")
SUBMISSIONS_REPO_ID = os.getenv("SUBMISSIONS_REPO_ID", GOLD_REPO_ID)
WRITE_TOKEN = os.getenv("HF_WRITE_TOKEN") or os.getenv("HF_TOKEN")
GRADIO_MAJOR_VERSION = int(gr.__version__.split(".", maxsplit=1)[0])

PORTAL_CSS = """
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
    margin: 0 auto !important;
    padding: 24px 30px 36px !important;
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
    min-height: 460px;
    background: var(--docsem-surface);
    border: 1px solid var(--docsem-line);
    border-radius: 6px;
    overflow: hidden;
}

#leaderboard-table table {
    font-size: 15px;
}

#leaderboard-table th {
    color: var(--docsem-navy);
    background: #eef2f6;
    font-size: 14px;
    font-weight: 700;
}

#leaderboard-table th button {
    font-size: 14px;
    font-weight: 700;
}

#leaderboard-table td,
#leaderboard-table th {
    padding: 13px 14px;
    white-space: nowrap;
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
        flex: 1;
        min-width: 0;
        padding: 0 10px;
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


def _read_hub_file(repo_id, filename, token=None):
    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", token=token)
    return Path(path).read_text(encoding="utf-8")


def _load_gold_rows():
    return load_jsonl_text(_read_hub_file(GOLD_REPO_ID, GOLD_FILE, token=WRITE_TOKEN))


def _persist_submission(rows, team, contact, submission_name, metrics):
    if not SUBMISSIONS_REPO_ID or not WRITE_TOKEN:
        return "Score computed. Persistence is disabled until SUBMISSIONS_REPO_ID and HF_WRITE_TOKEN are configured."

    submitted_at = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    row = leaderboard_row(team, contact, submission_name, metrics, submitted_at)
    public_row = {key: value for key, value in row.items() if key != "contact"}
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
    _update_leaderboard(public_row)
    return f"Score computed and saved to {SUBMISSIONS_REPO_ID}/submissions/{filename}."


def _load_leaderboard_rows():
    if not SUBMISSIONS_REPO_ID or not WRITE_TOKEN:
        return []
    try:
        text = _read_hub_file(SUBMISSIONS_REPO_ID, "leaderboard/leaderboard.json", token=WRITE_TOKEN)
    except (EntryNotFoundError, RepositoryNotFoundError):
        return []
    rows = json.loads(text)
    return rows if isinstance(rows, list) else []


def _sort_leaderboard(rows):
    return sorted(
        rows,
        key=lambda row: (
            -float(row.get("answer_accuracy", 0.0)),
            -float(row.get("evidence_exact_match", 0.0)),
            -float(row.get("evidence_f1", 0.0)),
            str(row.get("submitted_at", "")),
        ),
    )


def _update_leaderboard(row):
    rows = _load_leaderboard_rows()
    rows.append(row)
    rows = _sort_leaderboard(rows)
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


def leaderboard_table():
    rows = _sort_leaderboard(_load_leaderboard_rows())
    return [
        [
            index + 1,
            row.get("team", ""),
            row.get("submission_name", ""),
            row.get("answer_accuracy", 0.0),
            row.get("evidence_exact_match", 0.0),
            row.get("evidence_f1", 0.0),
            row.get("submitted_at", ""),
        ]
        for index, row in enumerate(rows[:100])
    ]


def evaluate_submission(file_obj, team, contact, submission_name):
    if file_obj is None:
        raise gr.Error("Upload a JSONL submission file.")
    if not team.strip():
        raise gr.Error("Enter a team name.")
    if not contact.strip():
        raise gr.Error("Enter a contact email.")
    if not submission_name.strip():
        raise gr.Error("Enter a submission name.")

    try:
        text = Path(file_obj.name).read_text(encoding="utf-8")
        rows = parse_submission_text(text)
        labels = _load_gold_rows()
        metrics = score_predictions(rows, labels)
        message = _persist_submission(rows, team.strip(), contact.strip(), submission_name.strip(), metrics)
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
                <a class="primary-link" href="{WORKSHOP_URL}" target="_blank" rel="noopener">
                    Workshop website
                </a>
                <a href="https://huggingface.co/datasets/{PUBLIC_DATASET_REPO}" target="_blank" rel="noopener">
                    Public dataset
                </a>
            </nav>
        </header>
        """
    )

    with gr.Group(elem_id="submission-panel"):
        gr.HTML(
            """
            <h2>Submit validation predictions</h2>
            <p class="submission-note">
                Upload one JSON object per instance with <code>instance_id</code>,
                <code>answer</code>, and <code>evidence</code>.
            </p>
            """
        )
        with gr.Row(elem_id="submission-fields"):
            team = gr.Textbox(label="Team", placeholder="example-team")
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
                    "Validation labels remain organizer-only."
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
        inputs=[file_input, team, contact, submission_name],
        outputs=result,
    )

    with gr.Column(elem_id="leaderboard-section"):
        with gr.Row(elem_id="leaderboard-heading"):
            gr.HTML(
                """
                <div>
                    <h2>Leaderboard</h2>
                    <p>Ranked by answer accuracy, then evidence exact match and evidence F1.</p>
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
        leaderboard = gr.Dataframe(
            headers=[
                "Rank",
                "Team",
                "Submission",
                "Answer accuracy",
                "Evidence exact match",
                "Evidence F1",
                "Submitted at",
            ],
            value=leaderboard_table,
            interactive=False,
            wrap=False,
            column_widths=["6%", "16%", "23%", "13%", "16%", "11%", "15%"],
            elem_id="leaderboard-table",
            **(
                {"max_height": 460}
                if GRADIO_MAJOR_VERSION >= 6
                else {"height": 460}
            ),
        )
        refresh.click(leaderboard_table, inputs=None, outputs=leaderboard)


if __name__ == "__main__":
    launch_options = {"css": PORTAL_CSS} if GRADIO_MAJOR_VERSION >= 6 else {}
    demo.launch(**launch_options)
