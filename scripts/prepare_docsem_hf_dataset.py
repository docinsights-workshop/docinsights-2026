#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SOURCE_ROOT = Path("/Users/aamita/Oracle/amitbcp/gsm-sem/docsem")
PUBLIC_LICENSE = PUBLIC_SOURCE_ROOT.parent / "LICENSE.txt"
ORGANIZER_SOURCE_ROOT = Path(
    "/Users/aamita/Oracle/amitbcp/docsem-workshop-final-public"
)
TARGET_ROOT = REPO_ROOT / "competition/hf-dataset"
PRIVATE_ROOT = REPO_ROOT / "competition/hf-submissions"


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def copy_pdfs(split):
    source_dir = PUBLIC_SOURCE_ROOT / split / "documents"
    target_dir = TARGET_ROOT / split / "documents"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for pdf in sorted(source_dir.glob("*.pdf")):
        shutil.copy2(pdf, target_dir / pdf.name)


def public_tasks(split):
    rows = []
    for row in read_jsonl(PUBLIC_SOURCE_ROOT / split / "tasks.jsonl"):
        rows.append(
            {
                "instance_id": row["instance_id"],
                "user_query": row["user_query"],
                "document_pdf": f"{split}/{row['document_pdf']}",
            }
        )
    return rows


def labels(source_root, split):
    return [
        {
            "instance_id": row["instance_id"],
            "answer": str(row["answer"]),
            "evidence": [str(value) for value in row["evidence"]],
        }
        for row in read_jsonl(source_root / split / "labels.jsonl")
    ]


def validate_public_split(split, label_rows=None):
    tasks = public_tasks(split)
    task_ids = {row["instance_id"] for row in tasks}
    pdf_ids = {
        path.stem
        for path in (PUBLIC_SOURCE_ROOT / split / "documents").glob("*.pdf")
    }
    if task_ids != pdf_ids:
        raise ValueError(f"{split} task ids and PDF ids differ")
    if label_rows is not None:
        label_ids = {row["instance_id"] for row in label_rows}
        if task_ids != label_ids:
            raise ValueError(f"{split} task ids and label ids differ")
    return tasks


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the public DocSem HF dataset and private validation labels."
    )
    parser.add_argument(
        "--reset-leaderboard",
        action="store_true",
        help="Reset the local generated leaderboard payload to an empty list.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_labels = labels(PUBLIC_SOURCE_ROOT, "train")
    train_tasks = validate_public_split("train", train_labels)
    val_labels = labels(ORGANIZER_SOURCE_ROOT, "val")
    val_tasks = validate_public_split("val", val_labels)

    write_jsonl(TARGET_ROOT / "train" / "tasks.jsonl", train_tasks)
    write_jsonl(TARGET_ROOT / "train" / "labels.jsonl", train_labels)
    write_jsonl(TARGET_ROOT / "val" / "tasks.jsonl", val_tasks)

    copy_pdfs("train")
    copy_pdfs("val")

    sample_submission = [
        {
            "instance_id": row["instance_id"],
            "answer": "0",
            "evidence": ["b01"],
        }
        for row in val_tasks[:5]
    ]
    write_jsonl(TARGET_ROOT / "examples" / "sample_val_submission.jsonl", sample_submission)

    shutil.copy2(
        PUBLIC_SOURCE_ROOT / "PARTICIPANT_INSTRUCTIONS.md",
        TARGET_ROOT / "INSTRUCTIONS.md",
    )
    shutil.copy2(PUBLIC_LICENSE, TARGET_ROOT / "LICENSE.txt")

    write_jsonl(PRIVATE_ROOT / "private" / "val_labels.jsonl", val_labels)
    (PRIVATE_ROOT / "submissions").mkdir(parents=True, exist_ok=True)
    (PRIVATE_ROOT / "submissions" / ".gitkeep").write_text("", encoding="utf-8")
    if args.reset_leaderboard:
        (PRIVATE_ROOT / "leaderboard").mkdir(parents=True, exist_ok=True)
        (PRIVATE_ROOT / "leaderboard" / "leaderboard.json").write_text(
            "[]\n", encoding="utf-8"
        )

    print(
        json.dumps(
            {
                "train_tasks": len(train_tasks),
                "train_labels": len(train_labels),
                "val_tasks_public": len(val_tasks),
                "val_labels_private": len(val_labels),
                "leaderboard_reset": args.reset_leaderboard,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
