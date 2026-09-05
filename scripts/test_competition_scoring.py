#!/usr/bin/env python3
import json
import re
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


PRIVATE_TEST_PATH = re.compile(r"(?:private|sealed)/[A-Za-z0-9._/-]+")
ALLOWED_VALIDATION_SERVER_PATHS = {"private/val_labels.jsonl"}
EMAIL_VALUE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PUBLIC_COMPONENT_EMAILS = {"lead@example.org"}
PRIVATE_TEST_FIELDS = {
    "gold_sha256",
    "oauth_sub",
    "predictions",
    "scoring_gold_sha256",
}
LATER_SCORE_FIELDS = {"answer_accuracy", "evidence_f1", "evidence_exact_match"}

COMPATIBILITY_LABELS = [
    {"instance_id": "fixture-1", "answer": "10", "evidence": ["b01", "b02"]},
    {"instance_id": "fixture-2", "answer": "20", "evidence": ["b03"]},
]
COMPATIBILITY_ROWS = [
    {"instance_id": "fixture-1", "answer": "10.0", "evidence": ["b02", "b01"]},
    {"instance_id": "fixture-2", "answer": "999", "evidence": ["b03", "b04"]},
]


def _assert_no_private_path(*artifacts):
    paths = {
        path
        for artifact in artifacts
        for path in PRIVATE_TEST_PATH.findall(str(artifact))
    }
    if paths - ALLOWED_VALIDATION_SERVER_PATHS:
        raise AssertionError("test deployment leakage detected: private path")


def _assert_only_public_component_emails(*artifacts):
    emails = {
        email.casefold()
        for artifact in artifacts
        for email in EMAIL_VALUE.findall(str(artifact))
    }
    if not emails <= PUBLIC_COMPONENT_EMAILS:
        raise AssertionError("test deployment leakage detected: email")


def configured_test_secret_paths(deployment=app.TEST_DEPLOYMENT):
    return tuple(
        path
        for path in (
            getattr(deployment, "release_config_path", None),
            getattr(deployment, "gold_config_path", None),
        )
        if isinstance(path, str) and path
    )


def assert_no_test_leakage(
    source,
    rendered_config,
    api_schema,
    response,
    failure_log,
    *,
    configured_secret_paths=None,
):
    """Apply class-aware rules without rejecting participant input schema terms."""

    artifacts = (source, rendered_config, api_schema, response, failure_log)
    _assert_no_private_path(*artifacts)
    configured_secret_paths = (
        configured_test_secret_paths()
        if configured_secret_paths is None
        else tuple(path for path in configured_secret_paths if isinstance(path, str) and path)
    )
    if any(
        secret_path in str(artifact)
        for secret_path in configured_secret_paths
        for artifact in artifacts
    ):
        raise AssertionError("test deployment leakage detected: configured path")
    _assert_only_public_component_emails(rendered_config, api_schema)
    if not isinstance(response, dict):
        raise AssertionError("test deployment leakage detected: malformed response")
    if PRIVATE_TEST_FIELDS & set(response):
        raise AssertionError("test deployment leakage detected: private field")
    if "per_example" in response:
        raise AssertionError("test deployment leakage detected: per_example")
    if response.get("attempt") in {2, 3} and LATER_SCORE_FIELDS & set(response):
        raise AssertionError("test deployment leakage detected: later score")
    _assert_only_public_component_emails(response, failure_log)


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
    api_schema = app.demo.get_api_info()
    serialized_api_schema = json.dumps(api_schema, sort_keys=True)
    assert "/submit_predictions" in api_schema["named_endpoints"]
    assert "/my_test_submissions" in api_schema["named_endpoints"]
    participant_response = participant_test_response(
        2,
        {
            "answer_accuracy": 0.5,
            "evidence_f1": 0.75,
            "per_example": [
                {
                    "answer": "withheld",
                    "evidence": ["b1"],
                }
            ],
        },
        "test-receipt",
    )
    profile = {
        "sub": "subject-fixture",
        "preferred_username": "private-test-user",
        "email": "fixture@example.org",
        "email_verified": True,
    }
    service = SubmissionService(
        validation_submitter=lambda file_obj, metadata: None,
        test_store=None,
        test_config_loader=lambda now: (_ for _ in ()).throw(
            RuntimeError("sensitive release unavailable")
        ),
    )
    try:
        service.submit_for_split("test", None, {}, profile)
    except SubmissionError as exc:
        failure_log = f"test submission failed: {exc}"
    else:
        raise AssertionError("failure fixture must fail closed")

    assert_no_test_leakage(source, rendered_config, serialized_api_schema, participant_response, failure_log)


def test_test_leakage_scanner_rejects_every_forbidden_class():
    clean = ("source", "{}", "{}", {"attempt": 2, "score": "withheld"}, "failure")
    for label, artifacts in (
        ("private path", ("sealed/gold.jsonl", *clean[1:])),
        ("email", (clean[0], "{\"email\": \"leak@example.org\"}", *clean[2:])),
        ("private field", (*clean[:3], {"attempt": 1, "predictions": []}, clean[4])),
        ("per_example", (*clean[:3], {"attempt": 1, "per_example": []}, clean[4])),
        ("later score", (*clean[:3], {"attempt": 2, "answer_accuracy": 0.5}, clean[4])),
    ):
        try:
            assert_no_test_leakage(*artifacts)
        except AssertionError as exc:
            assert label in str(exc)
        else:
            raise AssertionError(f"scanner accepted leaked {label}")

    configured_paths = ("organizer/release.json", "organizer/gold.jsonl")
    for configured_path in configured_paths:
        try:
            assert_no_test_leakage(
                "source",
                json.dumps({"leak": configured_path}),
                "{}",
                {"attempt": 2, "score": "withheld"},
                "failure",
                configured_secret_paths=configured_paths,
            )
        except AssertionError as exc:
            assert "configured path" in str(exc)
            assert configured_path not in str(exc)
        else:
            raise AssertionError("scanner accepted an exact configured secret path")


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
