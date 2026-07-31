#!/usr/bin/env python3
import json
import shutil
from pathlib import Path


SOURCE_ROOT = Path("/Users/aamita/Oracle/amitbcp/docsem-workshop-final-public")
TARGET_ROOT = Path("competition/hf-dataset")
PRIVATE_ROOT = Path("competition/hf-submissions")


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def copy_pdfs(split):
    source_dir = SOURCE_ROOT / split / "documents"
    target_dir = TARGET_ROOT / split / "documents"
    target_dir.mkdir(parents=True, exist_ok=True)
    for pdf in sorted(source_dir.glob("*.pdf")):
        shutil.copy2(pdf, target_dir / pdf.name)


def public_tasks(split):
    rows = []
    for row in read_jsonl(SOURCE_ROOT / split / "tasks.jsonl"):
        rows.append(
            {
                "instance_id": row["instance_id"],
                "user_query": row["user_query"],
                "document_pdf": f"{split}/{row['document_pdf']}",
            }
        )
    return rows


def labels(split):
    return [
        {
            "instance_id": row["instance_id"],
            "answer": str(row["answer"]),
            "evidence": [str(value) for value in row["evidence"]],
        }
        for row in read_jsonl(SOURCE_ROOT / split / "labels.jsonl")
    ]


def validate_split(split):
    tasks = public_tasks(split)
    label_rows = labels(split)
    task_ids = {row["instance_id"] for row in tasks}
    label_ids = {row["instance_id"] for row in label_rows}
    pdf_ids = {path.stem for path in (SOURCE_ROOT / split / "documents").glob("*.pdf")}
    if task_ids != label_ids:
        raise ValueError(f"{split} task ids and label ids differ")
    if task_ids != pdf_ids:
        raise ValueError(f"{split} task ids and PDF ids differ")
    return tasks, label_rows


def main():
    train_tasks, train_labels = validate_split("train")
    val_tasks, val_labels = validate_split("val")

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

    if not (TARGET_ROOT / "INSTRUCTIONS.md").is_file():
        raise FileNotFoundError(
            "Keep competition/hf-dataset/INSTRUCTIONS.md synced with the canonical "
            "oracle-samples/gsm-sem main/docsem release."
        )

    write_jsonl(PRIVATE_ROOT / "private" / "val_labels.jsonl", val_labels)
    (PRIVATE_ROOT / "submissions").mkdir(parents=True, exist_ok=True)
    (PRIVATE_ROOT / "submissions" / ".gitkeep").write_text("", encoding="utf-8")
    (PRIVATE_ROOT / "leaderboard").mkdir(parents=True, exist_ok=True)
    (PRIVATE_ROOT / "leaderboard" / "leaderboard.json").write_text("[]\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "train_tasks": len(train_tasks),
                "train_labels": len(train_labels),
                "val_tasks_public": len(val_tasks),
                "val_labels_private": len(val_labels),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
