#!/usr/bin/env python3
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "competition" / "hf-space"))

from scoring import (
    SubmissionError,
    answers_match,
    load_jsonl_file,
    parse_submission_text,
    score_predictions,
)


def test_numeric_equivalence():
    assert answers_match("Final answer: 10.0", "10")
    assert answers_match("1,400.00", "1400")
    assert not answers_match("1401", "1400")


def test_perfect_train_submission_sample():
    labels = load_jsonl_file(ROOT / "competition" / "hf-dataset" / "train" / "labels.jsonl")[:20]
    rows = [
        {
            "instance_id": row["instance_id"],
            "answer": row["answer"],
            "evidence": row["evidence"],
        }
        for row in labels
    ]
    metrics = score_predictions(rows, labels)
    assert metrics["examples"] == 20
    assert metrics["answer_accuracy"] == 1.0
    assert metrics["evidence_exact_match"] == 1.0
    assert metrics["evidence_f1"] == 1.0


def test_validation_sample_submission_is_valid_shape_but_scores_low():
    labels = load_jsonl_file(ROOT / "competition" / "hf-submissions" / "private" / "val_labels.jsonl")[:5]
    text = (ROOT / "competition" / "hf-dataset" / "examples" / "sample_val_submission.jsonl").read_text(encoding="utf-8")
    rows = parse_submission_text(text)
    metrics = score_predictions(rows, labels)
    assert metrics["examples"] == 5
    assert set(metrics) >= {"answer_accuracy", "evidence_exact_match", "evidence_f1"}


def test_duplicate_ids_are_rejected():
    rows = parse_submission_text(
        "\n".join(
            [
                json.dumps({"instance_id": "task_1", "answer": "1", "evidence": ["b01"]}),
                json.dumps({"instance_id": "task_1", "answer": "1", "evidence": ["b01"]}),
            ]
        )
    )
    try:
        score_predictions(rows, [{"instance_id": "task_1", "answer": "1", "evidence": ["b01"]}])
    except SubmissionError as exc:
        assert "Duplicate instance_id" in str(exc)
    else:
        raise AssertionError("duplicate ids should be rejected")


def test_missing_ids_are_rejected():
    rows = [{"instance_id": "task_1", "answer": "1", "evidence": ["b01"]}]
    labels = [
        {"instance_id": "task_1", "answer": "1", "evidence": ["b01"]},
        {"instance_id": "task_2", "answer": "2", "evidence": ["b02"]},
    ]
    try:
        score_predictions(rows, labels)
    except SubmissionError as exc:
        assert "Missing predictions" in str(exc)
    else:
        raise AssertionError("missing ids should be rejected")


def main():
    test_numeric_equivalence()
    test_perfect_train_submission_sample()
    test_validation_sample_submission_is_valid_shape_but_scores_low()
    test_duplicate_ids_are_rejected()
    test_missing_ids_are_rejected()
    print("competition scoring tests passed")


if __name__ == "__main__":
    main()
