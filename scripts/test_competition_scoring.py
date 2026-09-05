#!/usr/bin/env python3
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "competition" / "hf-space"))

from scoring import (
    SubmissionError,
    answers_match,
    parse_submission_text,
    score_predictions,
)
from submission_service import SubmissionService
from test_policy import participant_test_response
import app


TEST_LEAKAGE_FIXTURES = {
    "test gold path": "private/releases/docsem-test-2026/gold.jsonl",
    "test answer": "correct-test-answer-987",
    "test evidence": "b_test_gold_19",
    "later score": "0.987654",
    "email": "test-secret@example.org",
    "oauth subject": "test-oauth-subject-9",
    "private per-example metric": "private_per_example_metric_123",
}

COMPATIBILITY_LABELS = [
    {"instance_id": "fixture-1", "answer": "10", "evidence": ["b01", "b02"]},
    {"instance_id": "fixture-2", "answer": "20", "evidence": ["b03"]},
]
COMPATIBILITY_ROWS = [
    {"instance_id": "fixture-1", "answer": "10.0", "evidence": ["b02", "b01"]},
    {"instance_id": "fixture-2", "answer": "999", "evidence": ["b03", "b04"]},
]


def assert_no_test_leakage(*artifacts):
    """Reject release-specific private values from public deployment artifacts."""

    rendered = "\n".join(str(artifact).casefold() for artifact in artifacts)
    leaked = [
        label for label, value in TEST_LEAKAGE_FIXTURES.items() if value.casefold() in rendered
    ]
    if leaked:
        raise AssertionError(f"test deployment leakage detected: {', '.join(leaked)}")


def test_numeric_equivalence():
    assert answers_match("Final answer: 10.0", "10")
    assert answers_match("1,400.00", "1400")
    assert not answers_match("1401", "1400")


def test_perfect_train_submission_sample():
    labels = COMPATIBILITY_LABELS
    rows = [
        {
            "instance_id": row["instance_id"],
            "answer": row["answer"],
            "evidence": row["evidence"],
        }
        for row in labels
    ]
    metrics = score_predictions(rows, labels)
    assert metrics["examples"] == 2
    assert metrics["answer_accuracy"] == 1.0
    assert metrics["evidence_exact_match"] == 1.0
    assert metrics["evidence_f1"] == 1.0


def test_validation_sample_submission_is_valid_shape_but_scores_low():
    labels = COMPATIBILITY_LABELS
    text = "\n".join(json.dumps(row) for row in COMPATIBILITY_ROWS)
    rows = parse_submission_text(text)
    metrics = score_predictions(rows, labels)
    assert metrics == {
        "answer_accuracy": 0.5,
        "evidence_exact_match": 0.5,
        "evidence_f1": 0.833333,
        "examples": 2,
        "per_example": [
            {
                "instance_id": "fixture-1",
                "answer_exact_match": 1.0,
                "evidence_exact_match": 1.0,
                "evidence_f1": 1.0,
            },
            {
                "instance_id": "fixture-2",
                "answer_exact_match": 0.0,
                "evidence_exact_match": 0.0,
                "evidence_f1": 2 / 3,
            },
        ],
    }


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


def test_disabled_test_deployment_has_no_release_specific_leakage():
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "competition" / "hf-space").glob("*.py"))
        if not path.name.startswith("test_")
    )
    rendered_config = json.dumps(app.demo.get_config_file(), sort_keys=True)
    api_schema = json.dumps(
        {
            "named_endpoints": sorted(
                f"/{block_fn.api_name}"
                for block_fn in app.demo.fns.values()
                if block_fn.api_name
            )
        }
    )
    assert '"/submit_predictions"' in api_schema
    assert '"/my_test_submissions"' in api_schema
    participant_response = participant_test_response(
        2,
        {
            "answer_accuracy": 0.987654,
            "evidence_f1": 0.75,
            "per_example": [
                {
                    "answer": "correct-test-answer-987",
                    "evidence": ["b_test_gold_19"],
                    "private_per_example_metric_123": 1.0,
                }
            ],
        },
        "test-receipt",
    )
    profile = {
        "sub": "test-oauth-subject-9",
        "preferred_username": "private-test-user",
        "email": "test-secret@example.org",
        "email_verified": True,
    }
    service = SubmissionService(
        validation_submitter=lambda file_obj, metadata: None,
        test_store=None,
        test_config_loader=lambda now: (_ for _ in ()).throw(
            RuntimeError("private/releases/docsem-test-2026/gold.jsonl")
        ),
    )
    try:
        service.submit_for_split("test", None, {}, profile)
    except SubmissionError as exc:
        failure_log = f"test submission failed: {exc}"
    else:
        raise AssertionError("failure fixture must fail closed")

    assert_no_test_leakage(
        source,
        rendered_config,
        api_schema,
        json.dumps(participant_response, sort_keys=True),
        failure_log,
    )


def test_test_leakage_scanner_rejects_every_forbidden_class():
    for label, value in TEST_LEAKAGE_FIXTURES.items():
        try:
            assert_no_test_leakage(f"positive fixture {value}")
        except AssertionError as exc:
            assert label in str(exc)
        else:
            raise AssertionError(f"scanner accepted leaked {label}")


def main():
    test_numeric_equivalence()
    test_perfect_train_submission_sample()
    test_validation_sample_submission_is_valid_shape_but_scores_low()
    test_duplicate_ids_are_rejected()
    test_missing_ids_are_rejected()
    test_disabled_test_deployment_has_no_release_specific_leakage()
    test_test_leakage_scanner_rejects_every_forbidden_class()
    print("competition scoring tests passed")


if __name__ == "__main__":
    main()
