#!/usr/bin/env python3
"""Atomically correct labels or add joint metrics to stored validation submissions."""

import argparse
import copy
import hashlib
import io
import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, get_token, hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "competition" / "hf-space"))

from scoring import (  # noqa: E402
    load_jsonl_text,
    rank_leaderboard,
    score_validation_predictions,
)


DEFAULT_REPO_ID = "amitbcp/docinsights-2026-shared-task-submissions"
DEFAULT_GOLD_FILE = "private/val_labels.jsonl"
DEFAULT_LEADERBOARD_FILE = "leaderboard/leaderboard.json"
CANONICAL_SUBMISSION_PATH = re.compile(
    r"submissions/[A-Za-z0-9][A-Za-z0-9._-]*\.json\Z"
)
PRIOR_AGGREGATE_FIELDS = (
    "answer_accuracy",
    "evidence_exact_match",
    "evidence_f1",
    "examples",
)
PRIOR_EXAMPLE_FIELDS = (
    "answer_exact_match",
    "evidence_exact_match",
    "evidence_f1",
)


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

    metrics = score_validation_predictions(predictions, labels)
    leaderboard = dict(payload.get("leaderboard") or {})
    for field in [
        "answer_accuracy",
        "evidence_exact_match",
        "evidence_f1",
        "joint_accuracy",
        "examples",
    ]:
        leaderboard[field] = metrics[field]
    return {
        **payload,
        "leaderboard": leaderboard,
        "metrics": metrics,
    }


def migrate_joint_metric_payload(payload, labels):
    """Add joint metrics without changing stored predictions or prior metrics."""

    if not isinstance(payload, dict):
        raise ValueError("Stored submission payload is invalid")
    predictions = payload.get("predictions")
    metrics = payload.get("metrics")
    leaderboard = payload.get("leaderboard")
    if not isinstance(predictions, list):
        raise ValueError("Stored submission is missing a predictions list")
    if not isinstance(metrics, dict) or not isinstance(leaderboard, dict):
        raise ValueError("Stored submission metrics are invalid")
    prior_examples = metrics.get("per_example")
    if not isinstance(prior_examples, list):
        raise ValueError("Stored submission prior metrics are invalid")

    try:
        rescored = score_validation_predictions(predictions, labels)
    except (KeyError, TypeError, ValueError):
        raise ValueError("Stored submission could not be re-scored") from None
    for field in PRIOR_AGGREGATE_FIELDS:
        if metrics.get(field) != rescored[field]:
            raise ValueError("Stored submission prior metrics do not match re-scoring")
        if leaderboard.get(field) != rescored[field]:
            raise ValueError("Stored leaderboard prior metrics do not match re-scoring")

    rescored_examples = {row["instance_id"]: row for row in rescored["per_example"]}
    if len(rescored_examples) != len(rescored["per_example"]):
        raise ValueError("Re-scored example metrics are invalid")
    prior_ids = [
        row.get("instance_id") for row in prior_examples if isinstance(row, dict)
    ]
    if (
        len(prior_ids) != len(prior_examples)
        or len(set(prior_ids)) != len(prior_ids)
        or set(prior_ids) != set(rescored_examples)
    ):
        raise ValueError("Stored submission prior metrics are invalid")

    updated = copy.deepcopy(payload)
    updated_examples = updated["metrics"]["per_example"]
    for prior, migrated in zip(prior_examples, updated_examples, strict=True):
        rescored_example = rescored_examples[prior["instance_id"]]
        for field in PRIOR_EXAMPLE_FIELDS:
            if prior.get(field) != rescored_example[field]:
                raise ValueError(
                    "Stored submission prior metrics do not match re-scoring"
                )
        migrated["joint_exact_match"] = rescored_example["joint_exact_match"]

    updated["metrics"]["joint_accuracy"] = rescored["joint_accuracy"]
    updated["leaderboard"]["joint_accuracy"] = rescored["joint_accuracy"]
    return updated


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_remote_bytes(repo_id, filename, token, *, revision, cache_dir):
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        force_download=True,
    )
    return Path(path).read_bytes()


def _read_remote_text(repo_id, filename, token, *, revision, cache_dir):
    return _read_remote_bytes(
        repo_id,
        filename,
        token,
        revision=revision,
        cache_dir=cache_dir,
    ).decode("utf-8")


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


def _require_canonical_joint_migration_config(args):
    if args.gold_file != DEFAULT_GOLD_FILE:
        raise RuntimeError(
            "Joint metric migration requires the canonical validation gold path"
        )
    if args.leaderboard_file != DEFAULT_LEADERBOARD_FILE:
        raise RuntimeError(
            "Joint metric migration requires the canonical validation leaderboard path"
        )


def _validated_submission_destinations(repo_files, *, gold_file, leaderboard_file):
    submission_files = [
        path
        for path in repo_files
        if path.startswith("submissions/") and path.endswith(".json")
    ]
    if any(
        not isinstance(path, str) or not CANONICAL_SUBMISSION_PATH.fullmatch(path)
        for path in submission_files
    ):
        raise RuntimeError(
            "Stored submissions include a non-canonical submission path"
        )

    output_paths = [*submission_files, leaderboard_file]
    if len(set(output_paths)) != len(output_paths) or gold_file in output_paths:
        raise RuntimeError(
            "Joint migration output destinations must be unique and disjoint from gold"
        )
    return sorted(submission_files)


def _parser():
    parser = argparse.ArgumentParser(
        description=(
            "Atomically correct DocSem validation labels or migrate all stored "
            "validation submissions to the joint metric."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--corrections-file", type=Path)
    mode.add_argument(
        "--joint-metric-migration",
        action="store_true",
        help="Re-score stored validation submissions and add only joint metric fields.",
    )
    parser.add_argument(
        "--expected-submission-count",
        type=int,
        help=(
            "Exact stored validation-submission count required for the joint metric "
            "migration."
        ),
    )
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

    joint_migration = bool(args.joint_metric_migration)
    expected_submission_count = args.expected_submission_count
    if joint_migration:
        _require_canonical_joint_migration_config(args)
        if expected_submission_count is None or expected_submission_count < 1:
            raise RuntimeError(
                "A positive expected submission count is required for joint metric migration"
            )
    elif expected_submission_count is not None:
        raise RuntimeError(
            "Expected submission count is only valid for joint metric migration"
        )

    token = os.getenv("HF_WRITE_TOKEN") or os.getenv("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError("Set HF_WRITE_TOKEN or HF_TOKEN before accessing the private repository")

    api = HfApi(token=token)
    corrections = None if joint_migration else _parse_corrections(args.corrections_file)
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
        labels_bytes = _read_remote_bytes(
            args.repo_id,
            args.gold_file,
            token,
            revision=source_revision,
            cache_dir=cache_dir,
        )
        labels_sha256 = hashlib.sha256(labels_bytes).hexdigest()
        try:
            labels_text = labels_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError("Validation labels are not valid UTF-8") from None
        labels = load_jsonl_text(labels_text)
        if not joint_migration:
            labels = apply_label_corrections(labels, corrections)
        repo_files = api.list_repo_files(
            args.repo_id,
            repo_type="dataset",
            revision=source_revision,
            token=token,
        )
        submission_files = _validated_submission_destinations(
            repo_files,
            gold_file=args.gold_file,
            leaderboard_file=args.leaderboard_file,
        )
        if not submission_files:
            raise RuntimeError("No stored JSON submissions found")
        if joint_migration and len(submission_files) != expected_submission_count:
            raise RuntimeError(
                "Stored submissions do not match the expected submission count"
            )

        operations = []
        if not joint_migration:
            gold_path = temp_root / "val_labels.jsonl"
            _write_jsonl(gold_path, labels)
            operations.append(
                CommitOperationAdd(
                    path_in_repo=args.gold_file,
                    path_or_fileobj=gold_path,
                )
            )

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
            new_payload = (
                migrate_joint_metric_payload(old_payload, labels)
                if joint_migration
                else recompute_submission_payload(old_payload, labels)
            )
            if old_payload.get("metrics") != new_payload["metrics"]:
                changed += 1
            rows.append(new_payload["leaderboard"])
            sealed_payload = io.BytesIO(
                (json.dumps(new_payload, indent=2, sort_keys=True) + "\n").encode(
                    "utf-8"
                )
            )
            operations.append(
                CommitOperationAdd(
                    path_in_repo=repo_path,
                    path_or_fileobj=sealed_payload,
                )
            )

        sealed_leaderboard = io.BytesIO(
            (json.dumps(rows, indent=2, sort_keys=True) + "\n").encode("utf-8")
        )
        operations.append(
            CommitOperationAdd(
                path_in_repo=args.leaderboard_file,
                path_or_fileobj=sealed_leaderboard,
            )
        )

        ranked = rank_leaderboard(rows)
        summary = {
            "source_revision": source_revision,
            "repo_id": args.repo_id,
            "mode": (
                "joint_metric_migration" if joint_migration else "label_correction"
            ),
            "submissions_scanned": len(submission_files),
            "submissions_with_changed_metrics": changed,
            "leaderboard_rows": len(ranked),
            "commit": "pending" if not args.yes else "will be created",
        }
        if joint_migration:
            summary["expected_submission_count"] = expected_submission_count
            summary["label_sha256_unchanged"] = (
                "pending_verification" if args.yes else True
            )
        else:
            summary["correction_count"] = len(corrections)
        print(json.dumps(summary, indent=2, sort_keys=True))
        if not args.yes:
            print("Dry run only. Re-run with --yes to create the atomic Hugging Face commit.")
            return

        commit = api.create_commit(
            repo_id=args.repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=(
                "Add validation joint metric and refresh leaderboard"
                if joint_migration
                else "Correct DocSem validation ground truth and refresh leaderboard"
            ),
            commit_description=(
                "Re-score every stored validation submission, add only joint metric "
                "fields, and atomically rebuild the validation leaderboard. Stored "
                "predictions, identities, prior metrics, and validation labels remain "
                "unchanged."
                if joint_migration
                else (
                    f"Correct {len(corrections)} organizer-only validation labels and "
                    "recompute every stored submission against the corrected ground "
                    "truth. Public validation inputs remain unchanged."
                )
            ),
            revision="main",
            parent_commit=source_revision,
        )
        if joint_migration:
            committed_revision = str(getattr(commit, "oid", "") or "").strip()
            if not committed_revision:
                raise RuntimeError("Could not verify the committed migration revision")
            committed_labels_bytes = _read_remote_bytes(
                args.repo_id,
                args.gold_file,
                token,
                revision=committed_revision,
                cache_dir=cache_dir,
            )
            committed_labels_sha256 = hashlib.sha256(committed_labels_bytes).hexdigest()
            if committed_labels_sha256 != labels_sha256:
                raise RuntimeError("Validation label integrity verification failed")
            print(
                json.dumps(
                    {
                        "commit_status": "complete",
                        "label_sha256_unchanged": True,
                    },
                    sort_keys=True,
                )
            )
        else:
            print(f"Commit complete: {commit.commit_url}")


if __name__ == "__main__":
    main()
