import unittest

from scoring import (
    SubmissionError,
    leaderboard_row,
    normalize_participant_names,
    rank_leaderboard,
    score_predictions,
    score_validation_predictions,
)


def leaderboard_entry(
    *,
    team,
    contact,
    submission,
    submitted_at,
    answer_accuracy,
    evidence_exact_match,
    evidence_f1,
    joint_accuracy=0.0,
    participant_names=None,
):
    row = {
        "team": team,
        "contact": contact,
        "submission_name": submission,
        "submitted_at": submitted_at,
        "answer_accuracy": answer_accuracy,
        "evidence_exact_match": evidence_exact_match,
        "evidence_f1": evidence_f1,
        "joint_accuracy": joint_accuracy,
        "examples": 217,
    }
    if participant_names is not None:
        row["participant_names"] = participant_names
    return row


class LeaderboardRankingTests(unittest.TestCase):
    def test_joint_accuracy_is_the_primary_ranking_metric(self):
        rows = [
            leaderboard_entry(
                team="Answer Leader",
                contact="answer@example.org",
                submission="answer-first",
                submitted_at="2026-07-30T01:00:00Z",
                answer_accuracy=1.0,
                evidence_exact_match=0.0,
                evidence_f1=1.0,
                joint_accuracy=0.5,
            ),
            leaderboard_entry(
                team="Joint Leader",
                contact="joint@example.org",
                submission="joint-first",
                submitted_at="2026-07-30T02:00:00Z",
                answer_accuracy=0.7,
                evidence_exact_match=0.7,
                evidence_f1=0.7,
                joint_accuracy=0.6,
            ),
        ]

        ranked = rank_leaderboard(rows)

        self.assertEqual(
            [row["team"] for row in ranked], ["Joint Leader", "Answer Leader"]
        )

    def test_joint_ties_use_answer_then_evidence_f1(self):
        rows = [
            leaderboard_entry(
                team="Evidence Leader",
                contact="evidence@example.org",
                submission="evidence-first",
                submitted_at="2026-07-30T01:00:00Z",
                answer_accuracy=0.8,
                evidence_exact_match=0.8,
                evidence_f1=0.9,
                joint_accuracy=0.7,
            ),
            leaderboard_entry(
                team="Answer Leader",
                contact="answer@example.org",
                submission="answer-first",
                submitted_at="2026-07-30T02:00:00Z",
                answer_accuracy=0.9,
                evidence_exact_match=0.8,
                evidence_f1=0.1,
                joint_accuracy=0.7,
            ),
            leaderboard_entry(
                team="Evidence Trailer",
                contact="trailer@example.org",
                submission="evidence-last",
                submitted_at="2026-07-30T03:00:00Z",
                answer_accuracy=0.8,
                evidence_exact_match=0.8,
                evidence_f1=0.2,
                joint_accuracy=0.7,
            ),
        ]

        ranked = rank_leaderboard(rows)

        self.assertEqual(
            [row["team"] for row in ranked],
            ["Answer Leader", "Evidence Leader", "Evidence Trailer"],
        )

    def test_metric_ties_use_time_then_normalized_team_and_contact(self):
        rows = [
            leaderboard_entry(
                team="Same Team",
                contact="z@example.org",
                submission="z-contact",
                submitted_at="2026-07-30T02:00:00Z",
                answer_accuracy=0.8,
                evidence_exact_match=0.8,
                evidence_f1=0.8,
                joint_accuracy=0.8,
            ),
            leaderboard_entry(
                team="  same   team ",
                contact=" A@example.org ",
                submission="a-contact",
                submitted_at="2026-07-30T02:00:00Z",
                answer_accuracy=0.8,
                evidence_exact_match=0.8,
                evidence_f1=0.8,
                joint_accuracy=0.8,
            ),
            leaderboard_entry(
                team="Later Alphabetically",
                contact="later@example.org",
                submission="earlier-time",
                submitted_at="2026-07-30T01:00:00Z",
                answer_accuracy=0.8,
                evidence_exact_match=0.8,
                evidence_f1=0.8,
                joint_accuracy=0.8,
            ),
        ]

        ranked = rank_leaderboard(rows)

        self.assertEqual(
            [row["submission_name"] for row in ranked],
            ["earlier-time", "a-contact", "z-contact"],
        )

    def test_same_team_and_email_show_latest_attempt(self):
        rows = [
            leaderboard_entry(
                team="Example Team",
                contact="Lead@Example.org",
                submission="best",
                submitted_at="2026-07-30T01:00:00Z",
                answer_accuracy=0.8,
                evidence_exact_match=0.5,
                evidence_f1=0.75,
            ),
            leaderboard_entry(
                team="  example   team ",
                contact=" lead@example.org ",
                submission="latest-regression",
                submitted_at="2026-07-30T02:00:00Z",
                answer_accuracy=0.0,
                evidence_exact_match=0.6,
                evidence_f1=1.0,
            ),
        ]

        ranked = rank_leaderboard(rows)

        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["submission_name"], "latest-regression")
        self.assertEqual(ranked[0]["answer_accuracy"], 0.0)
        self.assertEqual(ranked[0]["evidence_f1"], 1.0)
        self.assertEqual(ranked[0]["attempts"], 2)

    def test_same_identity_uses_latest_attempt_and_participant_names(self):
        rows = [
            leaderboard_entry(
                team="Example Team",
                contact="lead@example.org",
                participant_names="Alice Example",
                submission="best",
                submitted_at="2026-07-30T01:00:00Z",
                answer_accuracy=0.8,
                evidence_exact_match=1.0,
                evidence_f1=1.0,
            ),
            leaderboard_entry(
                team="Example Team",
                contact="lead@example.org",
                participant_names="Alice Example, Bob Example",
                submission="latest-regression",
                submitted_at="2026-07-30T02:00:00Z",
                answer_accuracy=0.7,
                evidence_exact_match=1.0,
                evidence_f1=1.0,
            ),
        ]

        ranked = rank_leaderboard(rows)

        self.assertEqual(ranked[0]["submission_name"], "latest-regression")
        self.assertEqual(ranked[0]["participant_names"], "Alice Example, Bob Example")
        self.assertEqual(ranked[0]["attempts"], 2)

    def test_legacy_rows_without_participant_names_still_rank(self):
        row = leaderboard_entry(
            team="Legacy Team",
            contact="legacy@example.org",
            submission="legacy",
            submitted_at="2026-07-30T01:00:00Z",
            answer_accuracy=0.8,
            evidence_exact_match=1.0,
            evidence_f1=1.0,
        )

        ranked = rank_leaderboard([row])

        self.assertNotIn("participant_names", ranked[0])
        self.assertEqual(ranked[0]["attempts"], 1)

    def test_participant_names_are_required_and_normalized(self):
        self.assertEqual(
            normalize_participant_names("  Alice Example,\n Bob Example  "),
            "Alice Example, Bob Example",
        )
        with self.assertRaises(SubmissionError):
            normalize_participant_names("   ")

    def test_equal_scores_show_most_recent_attempt(self):
        rows = [
            leaderboard_entry(
                team="Example Team",
                contact="lead@example.org",
                submission="first",
                submitted_at="2026-07-30T01:00:00Z",
                answer_accuracy=0.8,
                evidence_exact_match=1.0,
                evidence_f1=0.75,
            ),
            leaderboard_entry(
                team="Example Team",
                contact="lead@example.org",
                submission="latest-equal",
                submitted_at="2026-07-30T02:00:00Z",
                answer_accuracy=0.8,
                evidence_exact_match=1.0,
                evidence_f1=0.75,
            ),
        ]

        ranked = rank_leaderboard(rows)

        self.assertEqual(ranked[0]["submission_name"], "latest-equal")
        self.assertEqual(ranked[0]["attempts"], 2)

    def test_same_team_with_different_email_is_a_separate_identity(self):
        rows = [
            leaderboard_entry(
                team="Example Team",
                contact="one@example.org",
                submission="one",
                submitted_at="2026-07-30T01:00:00Z",
                answer_accuracy=0.7,
                evidence_exact_match=1.0,
                evidence_f1=1.0,
            ),
            leaderboard_entry(
                team="Example Team",
                contact="two@example.org",
                submission="two",
                submitted_at="2026-07-30T02:00:00Z",
                answer_accuracy=0.8,
                evidence_exact_match=1.0,
                evidence_f1=1.0,
            ),
        ]

        ranked = rank_leaderboard(rows)

        self.assertEqual(len(ranked), 2)
        self.assertEqual([row["attempts"] for row in ranked], [1, 1])

    def test_evidence_f1_is_the_public_tie_breaker(self):
        rows = [
            leaderboard_entry(
                team="Exact Team",
                contact="exact@example.org",
                submission="higher-exact",
                submitted_at="2026-07-30T01:00:00Z",
                answer_accuracy=0.8,
                evidence_exact_match=1.0,
                evidence_f1=0.5,
            ),
            leaderboard_entry(
                team="F1 Team",
                contact="f1@example.org",
                submission="higher-f1",
                submitted_at="2026-07-30T02:00:00Z",
                answer_accuracy=0.8,
                evidence_exact_match=0.0,
                evidence_f1=0.9,
            ),
        ]

        ranked = rank_leaderboard(rows)

        self.assertEqual(ranked[0]["team"], "F1 Team")

    def test_tum_tse_alias_merges_into_tzachristas_team(self):
        rows = [
            leaderboard_entry(
                team="TUM-TSE",
                contact="jtzach@gmail.com",
                participant_names="Ioannis Tzachristas",
                submission="baseline-27-Aug-dummy",
                submitted_at="2026-08-27T14:19:13Z",
                answer_accuracy=1.0,
                evidence_exact_match=1.0,
                evidence_f1=1.0,
            ),
            leaderboard_entry(
                team="Tzachristas team",
                contact="jtzach@gmail.com",
                participant_names="Ioannis Tzachristas, Georgios Tzachristas, Theofanis Tzachristas, Constantinos Antoniou",
                submission="baseline-27-Aug-dummy",
                submitted_at="2026-08-28T14:27:21Z",
                answer_accuracy=1.0,
                evidence_exact_match=1.0,
                evidence_f1=1.0,
            ),
        ]

        ranked = rank_leaderboard(rows)

        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["team"], "Tzachristas team")
        self.assertEqual(ranked[0]["attempts"], 2)
        self.assertEqual(
            ranked[0]["participant_names"],
            "Ioannis Tzachristas, Georgios Tzachristas, Theofanis Tzachristas, Constantinos Antoniou",
        )


class JointMetricScoringTests(unittest.TestCase):
    def test_shared_scorer_keeps_the_legacy_test_metric_contract(self):
        metrics = score_predictions(
            [{"instance_id": "one", "answer": "10", "evidence": ["b01"]}],
            [{"instance_id": "one", "answer": "10", "evidence": ["b01"]}],
        )

        self.assertEqual(
            set(metrics),
            {
                "answer_accuracy",
                "evidence_exact_match",
                "evidence_f1",
                "examples",
                "per_example",
            },
        )
        self.assertEqual(
            set(metrics["per_example"][0]),
            {
                "instance_id",
                "answer_exact_match",
                "evidence_exact_match",
                "evidence_f1",
            },
        )

    def test_joint_exact_match_requires_answer_and_exact_evidence_on_same_example(self):
        labels = [
            {"instance_id": "both", "answer": "10", "evidence": ["b01", "b02"]},
            {"instance_id": "answer-only", "answer": "20", "evidence": ["b03", "b04"]},
            {"instance_id": "evidence-only", "answer": "30", "evidence": ["b05"]},
            {"instance_id": "neither", "answer": "40", "evidence": ["b06"]},
        ]
        predictions = [
            {"instance_id": "both", "answer": "10.0", "evidence": [" B02 ", "b01"]},
            {"instance_id": "answer-only", "answer": "20", "evidence": ["b03"]},
            {"instance_id": "evidence-only", "answer": "31", "evidence": ["B05"]},
            {"instance_id": "neither", "answer": "41", "evidence": ["b07"]},
        ]

        metrics = score_validation_predictions(predictions, labels)

        self.assertEqual(metrics["joint_accuracy"], 0.25)
        self.assertEqual(
            [row["joint_exact_match"] for row in metrics["per_example"]],
            [1.0, 0.0, 0.0, 0.0],
        )

    def test_leaderboard_row_carries_joint_accuracy(self):
        row = leaderboard_row(
            "Example Team",
            "lead@example.org",
            "baseline",
            {
                "answer_accuracy": 0.75,
                "evidence_exact_match": 0.5,
                "evidence_f1": 0.625,
                "joint_accuracy": 0.375,
                "examples": 8,
            },
            "2026-09-05T12:00:00Z",
        )

        self.assertEqual(row["joint_accuracy"], 0.375)


if __name__ == "__main__":
    unittest.main()
