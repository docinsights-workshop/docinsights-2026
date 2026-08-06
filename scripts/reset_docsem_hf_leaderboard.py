#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationDelete,
    HfApi,
)


REPO_ID = "amitbcp/docinsights-2026-shared-task-submissions"
LEADERBOARD_PATH = Path(__file__).resolve().parents[1] / (
    "competition/hf-submissions/leaderboard/leaderboard.json"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Atomically clear DocSem submissions and reset the HF leaderboard."
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Perform the reset. Without this flag, print a dry-run plan.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    api = HfApi()
    files = api.list_repo_files(REPO_ID, repo_type="dataset")
    if "private/val_labels.jsonl" not in files:
        raise RuntimeError("Refusing to reset a repo without private validation labels")

    submissions = sorted(
        path
        for path in files
        if path.startswith("submissions/") and path.endswith(".json")
    )
    plan = {
        "repo_id": REPO_ID,
        "submission_files_to_delete": submissions,
        "submission_count": len(submissions),
        "leaderboard": "reset to []",
        "hidden_validation_labels": "preserved",
    }
    print(json.dumps(plan, indent=2))
    if not args.yes:
        print("Dry run only. Re-run with --yes to apply this atomic commit.")
        return

    LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEADERBOARD_PATH.write_text("[]\n", encoding="utf-8")
    operations = [CommitOperationDelete(path_in_repo=path) for path in submissions]
    operations.append(
        CommitOperationAdd(
            path_in_repo="leaderboard/leaderboard.json",
            path_or_fileobj=LEADERBOARD_PATH,
        )
    )
    commit = api.create_commit(
        repo_id=REPO_ID,
        repo_type="dataset",
        operations=operations,
        commit_message="Reset DocSem leaderboard for refreshed data release",
        commit_description=(
            "Clear prior scored submissions after publishing the refreshed task PDFs. "
            "Private validation labels are preserved."
        ),
    )
    print(f"Reset complete: {commit.commit_url}")


if __name__ == "__main__":
    main()
