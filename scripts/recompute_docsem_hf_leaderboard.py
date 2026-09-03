#!/usr/bin/env python3
"""Re-score stored DocSem submissions after an organizer-only label correction."""

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, get_token, hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "competition" / "hf-space"))

from scoring import load_jsonl_text, rank_leaderboard, score_predictions  # noqa: E402


DEFAULT_REPO_ID = "amitbcp/docinsights-2026-shared-task-submissions"
DEFAULT_GOLD_FILE = "private/val_labels.jsonl"
DEFAULT_LEADERBOARD_FILE = "leaderboard/leaderboard.json"


def apply_label_corrections(labels, corrections):
    """Return labels with corrections applied to existing instance IDs only."""
    label_ids = [str(row["instance_id"]) for row in labels]
    duplicate_ids = sorted(
        instance_id
        for instance_id, count in Counter(label_ids).items()
        if count > 1
    )
    if duplicate_ids:
        raise ValueError("Duplicate validation label IDs found")

    labels_by_id = {str(row["instance_id"]): row for row in labels}
    unknown_ids = sorted(set(corrections) - set(labels_by_id))
    if unknown_ids:
        raise ValueError("Correction references unknown validation instance IDs")

    for instance_id, correction in corrections.items():
        current = str(labels_by_id[instance_id]["answer"])
        expected = correction["expected"]
        if current != expected:
            raise ValueError(
                "A validation answer does not match the expected value; refusing correction"
            )

    updated = []
    for row in labels:
        replacement = dict(row)
        instance_id = str(row["instance_id"])
        if instance_id in corrections:
            replacement["answer"] = corrections[instance_id]["replacement"]
        updated.append(replacement)
    return updated


def recompute_submission_payload(payload, labels):
    """Re-score one stored submission while preserving its identity metadata."""
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("Stored submission is missing a predictions list")

    metrics = score_predictions(predictions, labels)
    leaderboard = dict(payload.get("leaderboard") or {})
    for field in ["answer_accuracy", "evidence_exact_match", "evidence_f1", "examples"]:
        leaderboard[field] = metrics[field]
    return {
        **payload,
        "leaderboard": leaderboard,
        "metrics": metrics,
    }


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_remote_text(repo_id, filename, token, *, revision, cache_dir):
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        force_download=True,
    )
    return Path(path).read_text(encoding="utf-8")


def _parse_corrections(path):
    corrections = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(corrections, dict) or not corrections:
        raise ValueError("Corrections file must contain a non-empty JSON object")

    normalized = {}
    for instance_id, correction in corrections.items():
        if not isinstance(correction, dict) or set(correction) != {
            "expected",
            "replacement",
        }:
            raise ValueError(
                "Each correction must contain exactly expected and replacement values"
            )
        normalized[str(instance_id)] = {
            "expected": str(correction["expected"]),
            "replacement": str(correction["replacement"]),
        }
    return normalized


def _parser():
    parser = argparse.ArgumentParser(
        description="Atomically correct DocSem validation labels and re-score all stored submissions."
    )
    parser.add_argument("--corrections-file", type=Path, required=True)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--gold-file", default=DEFAULT_GOLD_FILE)
    parser.add_argument("--leaderboard-file", default=DEFAULT_LEADERBOARD_FILE)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Create the Hugging Face commit. Without this flag, print a dry-run plan.",
    )
    parser.add_argument(
        "--maintenance-confirmed",
        action="store_true",
        help=(
            "Confirm that the Space submission gate is live and in-flight scoring has "
            "drained before creating the commit. Required with --yes."
        ),
    )
    return parser


def main():
    args = _parser().parse_args()
    if args.yes and not args.maintenance_confirmed:
        raise RuntimeError(
            "Refusing to write until the Space submission maintenance gate is confirmed"
        )

    token = os.getenv("HF_WRITE_TOKEN") or os.getenv("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError("Set HF_WRITE_TOKEN or HF_TOKEN before accessing the private repository")

    api = HfApi(token=token)
    corrections = _parse_corrections(args.corrections_file)
    source_info = api.repo_info(
        args.repo_id,
        repo_type="dataset",
        revision="main",
        token=token,
    )
    source_revision = source_info.sha
    if not source_revision:
        raise RuntimeError("Could not resolve the private repository main revision")

    changed = 0
    rows = []
    with tempfile.TemporaryDirectory(prefix="docsem-recompute-") as temp_dir:
        temp_root = Path(temp_dir)
        cache_dir = temp_root / "hf-cache"
        labels = apply_label_corrections(
            load_jsonl_text(
                _read_remote_text(
                    args.repo_id,
                    args.gold_file,
                    token,
                    revision=source_revision,
                    cache_dir=cache_dir,
                )
            ),
            corrections,
        )
        repo_files = api.list_repo_files(
            args.repo_id,
            repo_type="dataset",
            revision=source_revision,
            token=token,
        )
        submission_files = sorted(
            path
            for path in repo_files
            if path.startswith("submissions/") and path.endswith(".json")
        )
        if not submission_files:
            raise RuntimeError("No stored JSON submissions found")

        gold_path = temp_root / "val_labels.jsonl"
        _write_jsonl(gold_path, labels)
        operations = [
            CommitOperationAdd(path_in_repo=args.gold_file, path_or_fileobj=gold_path)
        ]

        for repo_path in submission_files:
            old_payload = json.loads(
                _read_remote_text(
                    args.repo_id,
                    repo_path,
                    token,
                    revision=source_revision,
                    cache_dir=cache_dir,
                )
            )
            new_payload = recompute_submission_payload(old_payload, labels)
            if old_payload.get("metrics") != new_payload["metrics"]:
                changed += 1
            rows.append(new_payload["leaderboard"])
            local_path = temp_root / Path(repo_path).name
            local_path.write_text(json.dumps(new_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            operations.append(
                CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=local_path)
            )

        leaderboard_path = temp_root / "leaderboard.json"
        leaderboard_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        operations.append(
            CommitOperationAdd(path_in_repo=args.leaderboard_file, path_or_fileobj=leaderboard_path)
        )

        ranked = rank_leaderboard(rows)
        print(
            json.dumps(
                {
                    "source_revision": source_revision,
                    "repo_id": args.repo_id,
                    "correction_count": len(corrections),
                    "submissions_scanned": len(submission_files),
                    "submissions_with_changed_metrics": changed,
                    "leaderboard_rows": len(ranked),
                    "commit": "pending" if not args.yes else "will be created",
                },
                indent=2,
                sort_keys=True,
            )
        )
        if not args.yes:
            print("Dry run only. Re-run with --yes to create the atomic Hugging Face commit.")
            return

        commit = api.create_commit(
            repo_id=args.repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message="Correct DocSem validation ground truth and refresh leaderboard",
            commit_description=(
                f"Correct {len(corrections)} organizer-only validation labels and recompute "
                "every stored submission against the corrected ground truth. Public "
                "validation inputs remain unchanged."
            ),
            revision="main",
            parent_commit=source_revision,
        )
        print(f"Commit complete: {commit.commit_url}")


if __name__ == "__main__":
    main()
