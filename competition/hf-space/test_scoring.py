import unittest

from scoring import rank_leaderboard


def leaderboard_entry(
    *,
    team,
    contact,
    submission,
    submitted_at,
    answer_accuracy,
    evidence_exact_match,
    evidence_f1,
):
    return {
        "team": team,
        "contact": contact,
        "submission_name": submission,
        "submitted_at": submitted_at,
        "answer_accuracy": answer_accuracy,
        "evidence_exact_match": evidence_exact_match,
        "evidence_f1": evidence_f1,
        "examples": 217,
    }


class LeaderboardRankingTests(unittest.TestCase):
    def test_same_team_and_email_show_best_attempt(self):
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
        self.assertEqual(ranked[0]["submission_name"], "best")
        self.assertEqual(ranked[0]["attempts"], 2)

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


if __name__ == "__main__":
    unittest.main()
