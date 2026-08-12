import collections
import decimal
import json
import re
from pathlib import Path


class SubmissionError(ValueError):
    pass


FINAL_MARKER = re.compile(r"^\s*(final\s*answer\s*:|answer\s*:)\s*", re.IGNORECASE)
MAX_PARTICIPANT_NAMES_LENGTH = 500


def normalize_answer(value):
    text = FINAL_MARKER.sub("", str(value)).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_participant_names(value):
    names = re.sub(r"\s+", " ", str(value or "").strip())
    if not names:
        raise SubmissionError("Enter participant name(s).")
    if len(names) > MAX_PARTICIPANT_NAMES_LENGTH:
        raise SubmissionError(
            f"Participant name(s) must be {MAX_PARTICIPANT_NAMES_LENGTH} characters or fewer."
        )
    return names


def _decimal(value):
    text = normalize_answer(value).replace(",", "")
    if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text):
        return None
    try:
        return decimal.Decimal(text).normalize()
    except decimal.InvalidOperation:
        return None


def answers_match(prediction, reference):
    pred_decimal = _decimal(prediction)
    ref_decimal = _decimal(reference)
    if pred_decimal is not None and ref_decimal is not None:
        return pred_decimal == ref_decimal
    return normalize_answer(prediction) == normalize_answer(reference)


def evidence_set(values):
    if not isinstance(values, list) or not values:
        raise SubmissionError("evidence must be a non-empty list of block ids")
    normalized = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise SubmissionError("evidence entries must be non-empty strings")
        normalized.append(value.strip().lower())
    return set(normalized)


def evidence_f1(predicted, reference):
    pred = evidence_set(predicted)
    gold = evidence_set(reference)
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    overlap = len(pred & gold)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def parse_submission_text(text):
    stripped = text.strip()
    if not stripped:
        raise SubmissionError("Submission file is empty.")

    if stripped.startswith("["):
        rows = json.loads(stripped)
    elif stripped.startswith("{") and "\n" not in stripped:
        obj = json.loads(stripped)
        rows = obj["predictions"] if isinstance(obj, dict) and "predictions" in obj else [obj]
    else:
        rows = []
        for line_number, line in enumerate(stripped.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SubmissionError(f"Invalid JSONL at line {line_number}: {exc}") from exc

    if not isinstance(rows, list):
        raise SubmissionError("Submission must be JSONL, a JSON list, or an object with a predictions list.")
    return rows


def validate_predictions(rows, expected_ids=None):
    if not rows:
        raise SubmissionError("Submission has no predictions.")
    seen = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SubmissionError(f"Prediction at index {index} must be a JSON object.")
        for key in ["instance_id", "answer", "evidence"]:
            if key not in row:
                raise SubmissionError(f"Prediction at index {index} is missing {key}.")
        instance_id = row["instance_id"]
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise SubmissionError(f"Prediction at index {index} has an invalid instance_id.")
        if instance_id in seen:
            raise SubmissionError(f"Duplicate instance_id: {instance_id}")
        seen.add(instance_id)
        if not isinstance(row["answer"], str):
            raise SubmissionError(f"Prediction for {instance_id} must have a string answer.")
        evidence_set(row["evidence"])

    if expected_ids is not None:
        expected = set(expected_ids)
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        if missing:
            raise SubmissionError(f"Missing predictions for {len(missing)} ids: {', '.join(missing[:5])}")
        if extra:
            raise SubmissionError(f"Unexpected prediction ids ({len(extra)}): {', '.join(extra[:5])}")
    return True


def load_jsonl_text(text):
    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SubmissionError(f"Invalid JSONL at line {line_number}: {exc}") from exc
    if not rows:
        raise SubmissionError("JSONL file is empty.")
    return rows


def load_jsonl_file(path):
    return load_jsonl_text(Path(path).read_text(encoding="utf-8"))


def score_predictions(rows, labels):
    gold_by_id = {row["instance_id"]: row for row in labels}
    validate_predictions(rows, expected_ids=gold_by_id.keys())
    predictions = {row["instance_id"]: row for row in rows}

    answer_scores = []
    evidence_exact_scores = []
    evidence_f1_scores = []
    per_example = []

    for instance_id, gold in gold_by_id.items():
        pred = predictions[instance_id]
        answer_score = float(answers_match(pred["answer"], gold["answer"]))
        pred_evidence = evidence_set(pred["evidence"])
        gold_evidence = evidence_set(gold["evidence"])
        evidence_exact = float(pred_evidence == gold_evidence)
        ev_f1 = evidence_f1(pred["evidence"], gold["evidence"])
        answer_scores.append(answer_score)
        evidence_exact_scores.append(evidence_exact)
        evidence_f1_scores.append(ev_f1)
        per_example.append(
            {
                "instance_id": instance_id,
                "answer_exact_match": answer_score,
                "evidence_exact_match": evidence_exact,
                "evidence_f1": ev_f1,
            }
        )

    count = len(labels)
    return {
        "answer_accuracy": round(sum(answer_scores) / count, 6),
        "evidence_exact_match": round(sum(evidence_exact_scores) / count, 6),
        "evidence_f1": round(sum(evidence_f1_scores) / count, 6),
        "examples": count,
        "per_example": per_example,
    }


def leaderboard_row(
    team,
    contact,
    submission_name,
    metrics,
    submitted_at,
    participant_names=None,
):
    row = {
        "team": team,
        "contact": contact,
        "submission_name": submission_name,
        "submitted_at": submitted_at,
        "answer_accuracy": metrics["answer_accuracy"],
        "evidence_exact_match": metrics["evidence_exact_match"],
        "evidence_f1": metrics["evidence_f1"],
        "examples": metrics["examples"],
    }
    if participant_names is not None:
        row["participant_names"] = normalize_participant_names(participant_names)
    return row


def normalize_identity(value):
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def leaderboard_identity(row):
    return (
        normalize_identity(row.get("team", "")),
        normalize_identity(row.get("contact", "")),
    )


def latest_leaderboard_rows(rows):
    grouped = {}
    attempts = collections.Counter()
    latest_names = {}

    for index, row in enumerate(rows):
        identity = leaderboard_identity(row)
        attempts[identity] += max(int(row.get("attempts", 1)), 1)
        candidate_recency = (str(row.get("submitted_at", "")), index)
        participant_names = str(row.get("participant_names", "")).strip()
        if participant_names:
            current_metadata = latest_names.get(identity)
            if current_metadata is None or candidate_recency >= current_metadata[0]:
                latest_names[identity] = (candidate_recency, participant_names)
        current = grouped.get(identity)
        current_recency = None
        if current is not None:
            current_recency = (
                str(current.get("submitted_at", "")),
                int(current.get("_source_index", -1)),
            )
        if current_recency is None or candidate_recency >= current_recency:
            grouped[identity] = {**row, "_source_index": index}

    latest_rows = []
    for identity, row in grouped.items():
        row.pop("_source_index", None)
        row["attempts"] = attempts[identity]
        if identity in latest_names:
            row["participant_names"] = latest_names[identity][1]
        latest_rows.append(row)
    return latest_rows


def rank_leaderboard(rows):
    return sorted(
        latest_leaderboard_rows(rows),
        key=lambda row: (
            -float(row.get("answer_accuracy", 0.0)),
            -float(row.get("evidence_f1", 0.0)),
            str(row.get("submitted_at", "")),
            normalize_identity(row.get("team", "")),
        ),
    )


def safe_slug(value):
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value).strip()).strip("-")
    return slug[:80] or "submission"
