import csv
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app import (
    MAX_EXPORT_BYTES,
    OrganizerAppError,
    attempt_detail,
    build_app,
    export_csv,
    filter_rows,
    load_organizer_config,
    refresh_snapshot,
)
from test_organizer_data import (
    ACCOUNT_B,
    FakeHub,
    GOLD_DIGEST,
    ID_A1,
    ID_A2,
    ID_B1,
    PRIVATE_TOKEN,
    REVISION,
    TASK_DIGEST,
    fixture_files,
)
from test_policy import OAuthIdentity, canonical_submission_hash


PRIVATE_REPO = "private/docsem-organizer-ledger"


def organizer_environment(**overrides):
    values = {
        "ORGANIZER_READ_TOKEN": PRIVATE_TOKEN,
        "PRIVATE_REPO_ID": PRIVATE_REPO,
    }
    values.update(overrides)
    return values


class HeadAwareHub(FakeHub):
    """Read-only Hub fake that records every call and resolves HEAD once."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []
        self.mutation_calls = []

    def repo_info(self, repo_id, *, repo_type, revision=None, token):
        self.calls.append(("repo_info", repo_id, repo_type, revision, token))
        if repo_id != PRIVATE_REPO or repo_type != "dataset" or token != PRIVATE_TOKEN:
            raise RuntimeError("unauthorized")
        if revision is None:
            return SimpleNamespace(sha=self.revision, private=self.private)
        return super().repo_info(
            repo_id,
            repo_type=repo_type,
            revision=revision,
            token=token,
        )

    def list_repo_commits(self, repo_id, *, repo_type, revision, token):
        self.calls.append(("list_repo_commits", revision))
        return super().list_repo_commits(
            repo_id,
            repo_type=repo_type,
            revision=revision,
            token=token,
        )

    def list_repo_files(self, repo_id, *, repo_type, revision, token):
        self.calls.append(("list_repo_files", revision))
        return super().list_repo_files(
            repo_id,
            repo_type=repo_type,
            revision=revision,
            token=token,
        )

    def hf_hub_download(
        self,
        repo_id,
        filename,
        *,
        repo_type,
        revision,
        token,
        cache_dir,
    ):
        self.calls.append(("download", filename, revision))
        return super().hf_hub_download(
            repo_id,
            filename,
            repo_type=repo_type,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
        )

    def create_commit(self, *args, **kwargs):
        self.mutation_calls.append(("create_commit", args, kwargs))
        raise AssertionError("organizer Space must never write")

    def upload_file(self, *args, **kwargs):
        self.mutation_calls.append(("upload_file", args, kwargs))
        raise AssertionError("organizer Space must never write")


class NoNetworkHub:
    def __getattr__(self, name):
        raise AssertionError(f"build must not call Hub method {name}")


class OrganizerAppTests(unittest.TestCase):
    def setUp(self):
        self.config = load_organizer_config(organizer_environment())

    def refresh(self, hub=None):
        return refresh_snapshot(self.config, api=hub or HeadAwareHub())

    def test_build_refuses_missing_required_server_credentials(self):
        for environment in (
            {},
            {"PRIVATE_REPO_ID": PRIVATE_REPO},
            {"ORGANIZER_READ_TOKEN": PRIVATE_TOKEN},
            {"PRIVATE_REPO_ID": "not a repo", "ORGANIZER_READ_TOKEN": PRIVATE_TOKEN},
        ):
            with self.subTest(environment=tuple(sorted(environment))):
                with self.assertRaisesRegex(
                    OrganizerAppError,
                    r"Organizer Space configuration is unavailable\.",
                ):
                    build_app(environment=environment)

    def test_initial_gradio_config_is_empty_secret_free_and_read_only(self):
        demo = build_app(environment=organizer_environment(), api=NoNetworkHub())
        response = TestClient(demo.app, raise_server_exceptions=False).get("/config")
        self.assertEqual(response.status_code, 200, response.text[:200])
        rendered = json.dumps(response.json(), sort_keys=True)
        lowered = rendered.casefold()
        for sentinel in (
            PRIVATE_TOKEN,
            PRIVATE_REPO,
            "user-a@example.org",
            "subject-a",
            "raw-private-prediction",
            "gold-must-never-be-read",
            ID_A1,
        ):
            self.assertNotIn(sentinel.casefold(), lowered)
        self.assertNotIn("submit predictions", lowered)
        self.assertNotIn("finalize leaderboard", lowered)
        self.assertNotIn("create_commit", lowered)
        self.assertNotIn("upload_file", lowered)
        self.assertIn("refresh verified snapshot", lowered)
        self.assertIn("no snapshot loaded", lowered)

    def test_build_is_offline_and_does_not_resolve_the_private_repository(self):
        demo = build_app(environment=organizer_environment(), api=NoNetworkHub())
        self.assertIsNotNone(demo)

    def test_readme_app_file_starts_from_a_clean_organizer_only_bundle(self):
        """Catches undeclared imports from the sibling participant Space."""

        source = Path(__file__).resolve().parent
        readme = (source / "README.md").read_text(encoding="utf-8")
        app_file = next(
            line.split(":", 1)[1].strip()
            for line in readme.splitlines()
            if line.startswith("app_file:")
        )
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "organizer-space"
            shutil.copytree(
                source,
                bundle,
                ignore=shutil.ignore_patterns("test_*.py", "__pycache__"),
            )
            runner = """
import importlib.util
import sys
from pathlib import Path
from fastapi.testclient import TestClient

bundle = Path(sys.argv[1])
app_path = bundle / sys.argv[2]
sys.path.insert(0, str(bundle))
spec = importlib.util.spec_from_file_location("docsem_organizer_app", app_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

class NoNetworkHub:
    def __getattr__(self, name):
        raise AssertionError(f"startup must not call Hub method {name}")

demo = module.build_app(
    environment={
        "ORGANIZER_READ_TOKEN": "clean-bundle-token",
        "PRIVATE_REPO_ID": "private/docsem-ledger",
    },
    api=NoNetworkHub(),
)
client = TestClient(demo.app, raise_server_exceptions=False)
assert client.get("/").status_code == 200
assert client.get("/config").status_code == 200
"""
            result = subprocess.run(
                [sys.executable, "-c", runner, str(bundle), app_file],
                cwd=bundle,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(
            result.returncode,
            0,
            f"declared app_file failed in clean bundle:\n{result.stdout}\n{result.stderr}",
        )

    def test_refresh_resolves_head_then_reads_one_exact_private_sha(self):
        hub = HeadAwareHub()
        state, summary = self.refresh(hub)

        self.assertEqual(state.revision, REVISION)
        self.assertEqual(len(state.rows), 3)
        self.assertTrue(state.audit.valid)
        self.assertIn("Integrity: PASS", summary)
        self.assertIn("3 attempts", summary)
        self.assertIn("2 accounts", summary)
        self.assertIn("docsem-test-2026", summary)
        self.assertIn(TASK_DIGEST, summary)
        self.assertIn(GOLD_DIGEST, summary)

        self.assertEqual(
            hub.calls[0], ("repo_info", PRIVATE_REPO, "dataset", None, PRIVATE_TOKEN)
        )
        pinned_calls = [call for call in hub.calls[1:] if REVISION in call]
        self.assertTrue(pinned_calls)
        self.assertTrue(all("main" not in call for call in hub.calls))
        self.assertEqual(hub.mutation_calls, [])
        downloaded = [call[1] for call in hub.calls if call[0] == "download"]
        self.assertNotIn("private/test_labels.jsonl", downloaded)

    def test_refresh_rejects_public_or_moving_repository_without_private_detail(self):
        cases = (
            HeadAwareHub(private=False),
            HeadAwareHub(revision="moving-main"),
        )
        for hub in cases:
            with self.subTest(private=hub.private, revision=hub.revision):
                with self.assertRaisesRegex(
                    OrganizerAppError,
                    r"Organizer snapshot is unavailable\.",
                ) as caught:
                    self.refresh(hub)
                self.assertNotIn(PRIVATE_TOKEN, str(caught.exception))
                self.assertNotIn(PRIVATE_REPO, str(caught.exception))
                self.assertEqual(hub.mutation_calls, [])

    def test_refresh_fails_closed_on_projection_or_release_mismatch(self):
        files = fixture_files()
        projection_path = "projections/test/organizer_leaderboard.json"
        projection = json.loads(files[projection_path])
        projection["accounts"][0]["attempt_count"] = 99
        files[projection_path] = (
            json.dumps(projection, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()

        with self.assertRaisesRegex(
            OrganizerAppError,
            r"Organizer snapshot failed integrity verification\.",
        ):
            self.refresh(HeadAwareHub(files=files))

    def test_filters_cover_account_team_best_exclusion_and_adjudication(self):
        state, _ = self.refresh()

        self.assertEqual(
            [row["submission_id"] for row in filter_rows(state.rows, account="user-b")],
            [ID_B1],
        )
        self.assertEqual(
            {row["submission_id"] for row in filter_rows(state.rows, team="shared")},
            {ID_A1, ID_A2, ID_B1},
        )
        self.assertEqual(
            {row["submission_id"] for row in filter_rows(state.rows, best="selected")},
            {ID_B1},
        )
        self.assertEqual(
            {
                row["submission_id"]
                for row in filter_rows(state.rows, excluded="excluded")
            },
            {ID_A1, ID_A2},
        )
        self.assertEqual(
            {
                row["submission_id"]
                for row in filter_rows(state.rows, adjudication="has")
            },
            {ID_A1, ID_A2},
        )
        self.assertEqual(
            [
                row["submission_id"]
                for row in filter_rows(
                    state.rows,
                    account=ACCOUNT_B[:12],
                    best="selected",
                    excluded="eligible",
                    adjudication="none",
                )
            ],
            [ID_B1],
        )

    def test_filter_rejects_unknown_selector_values(self):
        state, _ = self.refresh()
        for field, value in (
            ("best", "best-ish"),
            ("excluded", "maybe"),
            ("adjudication", "sometimes"),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    OrganizerAppError,
                    r"Organizer filter is invalid\.",
                ):
                    filter_rows(state.rows, **{field: value})

    def test_per_example_detail_is_loaded_only_for_selected_verified_attempt(self):
        state, _ = self.refresh()
        detail = attempt_detail(state, ID_A2)

        self.assertEqual(detail["submission_id"], ID_A2)
        self.assertEqual(detail["attempt_number"], 2)
        self.assertEqual(
            [item["instance_id"] for item in detail["per_example"]],
            ["task-1", "task-2"],
        )
        self.assertEqual(
            detail["exclusions"],
            [
                {
                    "record_id": "exclude-smoke",
                    "created_at": "2026-09-04T12:00:00Z",
                    "reason_code": "organizer-smoke-test",
                }
            ],
        )
        self.assertEqual(
            detail["adjudications"],
            [
                {
                    "record_id": "appeal-reviewed",
                    "submission_id": ID_A2,
                    "created_at": "2026-09-05T12:00:00Z",
                    "action": "note",
                    "reason_code": "technical-review-complete",
                }
            ],
        )
        rendered = json.dumps(detail, sort_keys=True)
        self.assertNotIn("predictions", rendered)
        self.assertNotIn("raw-private-prediction", rendered)
        self.assertNotIn("gold", rendered.casefold())
        self.assertNotIn("correct_answer", rendered.casefold())

        for invalid in (None, "", "unknown-submission"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    OrganizerAppError,
                    r"Organizer attempt detail is unavailable\.",
                ):
                    attempt_detail(state, invalid)

    def test_csv_export_is_restricted_bounded_audited_and_contains_no_answers(self):
        state, _ = self.refresh()
        rows = filter_rows(state.rows, best="selected", excluded="eligible")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(export_csv(state, rows, directory=directory))
            self.assertTrue(path.is_file())
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(path.parent.stat().st_mode), 0o700)
            raw = path.read_bytes()
            self.assertLessEqual(len(raw), MAX_EXPORT_BYTES)
            text = raw.decode("utf-8")

        self.assertIn(f"repository_revision,{REVISION}", text)
        self.assertIn("release_id,docsem-test-2026", text)
        self.assertIn("evaluator_revisions,", text)
        self.assertIn("submission_id,attempt_number,selected_best,excluded", text)
        self.assertIn(ID_B1, text)
        self.assertNotIn(ID_A1, text)
        self.assertNotIn(ID_A2, text)
        for forbidden in (
            PRIVATE_TOKEN,
            "raw-private-prediction",
            "another-private-prediction",
            "gold-must-never-be-read",
            "correct_answer",
            "oauth_token",
            "access_token",
        ):
            self.assertNotIn(forbidden.casefold(), text.casefold())

    def test_csv_export_fails_closed_before_leaving_an_oversized_file(self):
        state, _ = self.refresh()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                OrganizerAppError,
                r"Organizer export is unavailable\.",
            ):
                export_csv(state, list(state.rows), directory=directory, max_bytes=64)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_csv_export_rejects_client_supplied_or_mutated_rows(self):
        state, _ = self.refresh()
        injected = dict(state.rows[0])
        injected["correct_answer"] = "must-not-export"
        injected["verified_email"] = "substituted@example.org"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                OrganizerAppError,
                r"Organizer export is unavailable\.",
            ):
                export_csv(state, [injected], directory=directory)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_csv_export_neutralizes_participant_controlled_spreadsheet_formulas(self):
        files = fixture_files()
        attempt_paths = sorted(
            path for path in files if path.startswith("attempts/test/")
        )
        attempt_digests = {}
        for path in attempt_paths:
            record = json.loads(files[path])
            record["team"] = "=1+1"
            files[path] = (
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode()
            attempt_digests[record["submission_id"]] = hashlib.sha256(
                files[path]
            ).hexdigest()
        for path in sorted(
            path for path in files if path.startswith("projections/test/accounts/")
        ):
            projection = json.loads(files[path])
            for reference in projection["attempts"]:
                reference["record_sha256"] = attempt_digests[reference["submission_id"]]
            files[path] = (
                json.dumps(projection, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode()
        organizer_path = "projections/test/organizer_leaderboard.json"
        projection = json.loads(files[organizer_path])
        for account in projection["accounts"]:
            account["team"] = "=1+1"
        files[organizer_path] = (
            json.dumps(projection, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()

        state, _ = self.refresh(HeadAwareHub(files=files))
        with tempfile.TemporaryDirectory() as directory:
            text = Path(
                export_csv(state, list(state.rows), directory=directory)
            ).read_text(encoding="utf-8")
        self.assertNotIn(",=1+1,", text)
        self.assertIn(",'=1+1,", text)

    def test_csv_export_neutralizes_formula_cells_in_audit_headers(self):
        """Catches a release identifier becoming an executable header cell."""

        files = fixture_files()
        malicious_release = '=HYPERLINK("https://invalid.example","open")'

        def replace_release_id(value):
            if isinstance(value, dict):
                return {
                    key: malicious_release
                    if key == "release_id"
                    else replace_release_id(item)
                    for key, item in value.items()
                }
            if isinstance(value, list):
                return [replace_release_id(item) for item in value]
            return value

        governed_paths = [
            path
            for path in files
            if path == "private/test_release.json"
            or path.startswith("attempts/test/")
            or path.startswith("projections/test/")
            or path.startswith("exclusions/test/")
            or path.startswith("adjudications/test/")
        ]
        for path in governed_paths:
            value = replace_release_id(json.loads(files[path]))
            if path.startswith("attempts/test/"):
                value["submission_hash"] = canonical_submission_hash(
                    value["predictions"],
                    "test",
                    malicious_release,
                    OAuthIdentity(
                        value["hf_subject"],
                        value["hf_username"],
                        value["verified_email"],
                    ),
                )
            files[path] = (
                json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode()

        attempt_digests = {}
        for path in governed_paths:
            if path.startswith("attempts/test/"):
                record = json.loads(files[path])
                attempt_digests[record["submission_id"]] = hashlib.sha256(
                    files[path]
                ).hexdigest()
        for path in governed_paths:
            if not path.startswith("projections/test/accounts/"):
                continue
            projection = json.loads(files[path])
            for reference in projection["attempts"]:
                reference["record_sha256"] = attempt_digests[reference["submission_id"]]
            files[path] = (
                json.dumps(projection, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode()

        state, _ = self.refresh(HeadAwareHub(files=files))
        with tempfile.TemporaryDirectory() as directory:
            path = export_csv(state, list(state.rows), directory=directory)
            with Path(path).open(encoding="utf-8", newline="") as stream:
                rows = list(csv.reader(stream))

        release_header = next(row for row in rows if row[:1] == ["release_id"])
        self.assertEqual(release_header, ["release_id", "'" + malicious_release])
        for row in rows:
            for cell in row:
                self.assertFalse(
                    cell.lstrip().startswith(("=", "+", "-", "@", "\t", "\r", "\n")),
                    (row, cell),
                )

    def test_callbacks_reject_unverified_or_mutated_state(self):
        state, _ = self.refresh()
        invalid_state = SimpleNamespace(
            revision=state.revision,
            snapshot=state.snapshot,
            audit=SimpleNamespace(valid=False),
            rows=state.rows,
        )
        for operation in (
            lambda: attempt_detail(invalid_state, ID_A1),
            lambda: export_csv(invalid_state, list(state.rows)),
        ):
            with self.assertRaises(OrganizerAppError):
                operation()

    def test_runtime_dependencies_are_the_proven_exact_versions(self):
        readme = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")
        requirements = dict(
            line.split("==", maxsplit=1)
            for line in (Path(__file__).parent / "requirements.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if line and not line.startswith("#")
        )
        self.assertEqual(
            requirements,
            {
                "gradio": "4.42.0",
                "huggingface_hub": "0.29.3",
                "fastapi": "0.112.2",
                "starlette": "0.38.6",
                "pydantic": "2.10.6",
            },
        )
        self.assertIn("sdk: gradio\n", readme)
        self.assertIn("sdk_version: 4.42.0\n", readme)


if __name__ == "__main__":
    unittest.main()
