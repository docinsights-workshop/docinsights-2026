#!/usr/bin/env python3
"""Offline tests for the guarded private DocSem organizer-Space publisher."""

from __future__ import annotations

import io
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import publish_docsem_organizer_space as publisher


WRITE_TOKEN = "hf_deploy_write_sentinel"
READ_TOKEN = "hf_runtime_read_sentinel"
DENIED_TOKEN = "hf_denied_probe_sentinel"
SOURCE_REVISION = "a" * 40
SPACE_PARENT = "b" * 40
PRIVATE_REVISION = "c" * 40


def whoami(role: str, name: str = "amitbcp") -> dict:
    """Mirror the documented ``/api/whoami-v2`` access-token shape."""

    return {
        "type": "user",
        "name": name,
        "auth": {
            "type": "access_token",
            "accessToken": {"displayName": "fixture", "role": role},
        },
    }


class FakeSourceBackend:
    def __init__(self) -> None:
        self.revision = SOURCE_REVISION
        self.clean = True
        self.inspect_count = 0
        self.move_on_second_inspect = False
        self.files = {
            name: f"pinned::{name}\n".encode("utf-8") for name in publisher.BUNDLE_PATHS
        }
        self.working_files = dict(self.files)

    def inspect_source(self, source_root):
        self.inspect_count += 1
        if self.move_on_second_inspect and self.inspect_count == 2:
            self.revision = "9" * 40
        return publisher.SourceState(
            root=Path(source_root),
            revision=self.revision,
            clean=self.clean,
        )

    def read_revision_file(self, source_root, revision, relative_path):
        if revision != self.revision:
            raise AssertionError("unexpected revision")
        return self.files[relative_path]

    def read_worktree_file(self, source_root, relative_path):
        return self.working_files[relative_path]


class FakeHubBackend:
    """Stateful fake at the network boundary; all publisher logic stays real."""

    def __init__(self, source: FakeSourceBackend, *, space_exists: bool = True):
        self.identities = {
            WRITE_TOKEN: whoami("write"),
            READ_TOKEN: whoami("read"),
            DENIED_TOKEN: whoami("read", "outside-reviewer"),
        }
        self.organizer = (
            publisher.SpaceState(
                exists=True,
                revision=SPACE_PARENT,
                private=True,
                sdk="gradio",
                host="https://organizer.example.test",
                runtime_stage="RUNNING",
            )
            if space_exists
            else publisher.SpaceState.missing()
        )
        self.participant = publisher.SpaceState(
            exists=True,
            revision="e" * 40,
            private=False,
            sdk="gradio",
            host="https://participant.example.test",
            runtime_stage="RUNNING",
        )
        self.dataset = publisher.DatasetState(
            revision=PRIVATE_REVISION,
            private=True,
        )
        self.dataset_files = ("private/val_labels.jsonl",)
        self.space_tree = {
            "README.md": b"old organizer\n",
            "test_app.py": b"must not deploy\n",
            "__pycache__/app.pyc": b"cache\n",
        }
        self.variables = {"EXISTING_PUBLIC_SETTING": "preserve"}
        self.secrets = {"EXISTING_PRIVATE_SECRET": "preserve"}
        self.writes = []
        self.secret_updates = []
        self.create_calls = []
        self.reserved_variable_after_secret = False
        self.commit_runtime_stage = "RUNNING"
        self.request_calls = []
        self.source = source
        self.organizer_config = {
            "components": [],
            "dependencies": [
                {"id": 0, "api_name": False, "targets": [[1, "click"]]},
            ],
        }
        self.organizer_info = {"named_endpoints": {}, "unnamed_endpoints": {}}
        self.participant_config = {
            "components": [
                {
                    "id": 4,
                    "type": "dropdown",
                    "props": {
                        "label": "Evaluation split",
                        "value": "Validation (development)",
                        "choices": [
                            ["Validation (development)", "Validation (development)"],
                            ["Test (final)", "Test (final)"],
                        ],
                    },
                },
                {
                    "id": 27,
                    "type": "dropdown",
                    "props": {
                        "label": "Leaderboard view",
                        "value": "Validation leaderboard",
                        "choices": [
                            ["Validation leaderboard", "Validation leaderboard"],
                            ["Final test leaderboard", "Final test leaderboard"],
                        ],
                    },
                },
            ],
            "dependencies": [
                {
                    "id": 4,
                    "targets": [[27, "change"]],
                    "inputs": [27],
                    "outputs": [29, 31, 30],
                    "api_name": False,
                }
            ],
        }

    def whoami(self, token):
        return self.identities[token]

    def inspect_space(self, repo_id, token):
        if repo_id == publisher.SPACE_REPO_ID:
            return self.organizer
        if repo_id == publisher.PARTICIPANT_SPACE_REPO_ID:
            return self.participant
        raise AssertionError(repo_id)

    def inspect_dataset(self, repo_id, revision, token):
        if repo_id != publisher.PRIVATE_DATASET_REPO_ID or token != READ_TOKEN:
            raise publisher.DeploymentError("Private dataset is unavailable.")
        if revision != self.dataset.revision:
            raise publisher.DeploymentError("Private dataset revision moved.")
        return self.dataset

    def list_dataset_files(self, repo_id, revision, token):
        self.inspect_dataset(repo_id, revision, token)
        return self.dataset_files

    def get_space_variables(self, repo_id, token):
        self.assert_organizer(repo_id)
        return dict(self.variables)

    def create_private_space(self, repo_id, token):
        self.assert_organizer(repo_id)
        if self.organizer.exists:
            raise AssertionError("already exists")
        self.create_calls.append({"repo_id": repo_id, "private": True, "sdk": "gradio"})
        self.organizer = publisher.SpaceState(
            exists=True,
            revision="1" * 40,
            private=True,
            sdk="gradio",
            host="https://organizer.example.test",
            runtime_stage="BUILDING",
        )
        self.space_tree = {}
        return self.organizer

    def list_space_files(self, repo_id, revision, token):
        self.assert_organizer(repo_id)
        if revision != self.organizer.revision:
            raise publisher.DeploymentError("Space revision moved.")
        return tuple(sorted(self.space_tree))

    def read_space_files(self, repo_id, revision, paths, token):
        self.list_space_files(repo_id, revision, token)
        return {name: self.space_tree[name] for name in paths}

    def commit_space(
        self,
        repo_id,
        expected_parent,
        additions,
        deletions,
        token,
    ):
        self.assert_organizer(repo_id)
        if expected_parent != self.organizer.revision:
            raise publisher.DeploymentError("Space revision moved.")
        self.writes.append(
            {
                "parent": expected_parent,
                "additions": {name: value for name, value in additions.items()},
                "deletions": tuple(deletions),
            }
        )
        for name in deletions:
            self.space_tree.pop(name, None)
        self.space_tree.update(additions)
        self.organizer = publisher.SpaceState(
            exists=True,
            revision="d" * 40,
            private=True,
            sdk="gradio",
            host="https://organizer.example.test",
            runtime_stage=self.commit_runtime_stage,
        )
        return self.organizer.revision

    def set_space_secret(self, repo_id, name, value, token):
        self.assert_organizer(repo_id)
        self.secret_updates.append(name)
        self.secrets[name] = value
        if self.reserved_variable_after_secret:
            self.variables[name] = "wrong-channel"

    def request(self, method, url, *, token=None, json_body=None):
        self.request_calls.append((method, url, token, json_body))
        if self.organizer.host and url.startswith(self.organizer.host):
            if token is None:
                return publisher.HttpResponse(401, b"private", {})
            if token == DENIED_TOKEN:
                return publisher.HttpResponse(403, b"forbidden", {})
            if token != READ_TOKEN:
                return publisher.HttpResponse(403, b"forbidden", {})
            path = url.removeprefix(self.organizer.host)
            if path == "/":
                return publisher.HttpResponse(200, b"<html>organizer</html>", {})
            if path == "/config":
                return publisher.HttpResponse(
                    200, json.dumps(self.organizer_config).encode(), {}
                )
            if path == "/info":
                return publisher.HttpResponse(
                    200, json.dumps(self.organizer_info).encode(), {}
                )
        if url.startswith(self.participant.host):
            path = url.removeprefix(self.participant.host)
            if method == "GET" and path == "/config":
                return publisher.HttpResponse(
                    200, json.dumps(self.participant_config).encode(), {}
                )
            if method == "POST" and path == "/api/select_split":
                return publisher.HttpResponse(
                    200,
                    json.dumps(
                        {
                            "data": [
                                {"value": "Test submissions are not open yet."},
                                {"visible": False},
                                {
                                    "interactive": False,
                                    "value": "Submit test predictions",
                                },
                                {"visible": True},
                            ]
                        }
                    ).encode(),
                    {},
                )
            if method == "POST" and path == "/api/predict":
                return publisher.HttpResponse(
                    200,
                    json.dumps(
                        {
                            "data": [
                                {"value": "Final test leaderboard"},
                                {
                                    "value": "The final test leaderboard is not available yet."
                                },
                                {"visible": False},
                            ]
                        }
                    ).encode(),
                    {},
                )
        return publisher.HttpResponse(404, b"not found", {})

    @staticmethod
    def assert_organizer(repo_id):
        if repo_id != publisher.SPACE_REPO_ID:
            raise AssertionError(repo_id)


def valid_request(**changes):
    values = {
        "expected_source_revision": SOURCE_REVISION,
        "expected_private_revision": PRIVATE_REVISION,
        "expected_space_parent": SPACE_PARENT,
        "expect_absent": False,
        "visibility": "private",
        "collaborators": ("amitbcp",),
        "publish": False,
        "verify_only": False,
        "confirmation": None,
    }
    values.update(changes)
    return publisher.DeploymentRequest(**values)


class SourceBundleTests(unittest.TestCase):
    def test_bundle_is_exactly_five_pinned_production_files(self):
        source = FakeSourceBackend()

        bundle = publisher.capture_source_bundle(source, SOURCE_REVISION)

        self.assertEqual(tuple(bundle.files), publisher.BUNDLE_PATHS)
        self.assertEqual(
            {name: item.payload for name, item in bundle.files.items()},
            source.files,
        )
        self.assertNotIn("test_app.py", bundle.files)
        self.assertNotIn("__pycache__", "\n".join(bundle.files))
        self.assertNotIn(next(iter(source.files.values())).decode(), repr(bundle))

    def test_bundle_rejects_dirty_or_mismatched_worktree(self):
        source = FakeSourceBackend()
        source.clean = False
        with self.assertRaisesRegex(publisher.DeploymentError, "clean"):
            publisher.capture_source_bundle(source, SOURCE_REVISION)

        source.clean = True
        source.working_files["app.py"] = b"uncommitted change\n"
        with self.assertRaisesRegex(publisher.DeploymentError, "revision"):
            publisher.capture_source_bundle(source, SOURCE_REVISION)

    def test_bundle_rejects_wrong_expected_source_revision(self):
        with self.assertRaisesRegex(publisher.DeploymentError, "source revision"):
            publisher.capture_source_bundle(FakeSourceBackend(), "d" * 40)

    def test_bundle_rejects_hugging_face_write_capabilities(self):
        for method in (
            "upload_file",
            "create_repo",
            "restart_space",
            "add_space_variable",
        ):
            with self.subTest(method=method):
                source = FakeSourceBackend()
                source.files["app.py"] += (
                    f"\napi.{method}(repo_id='private')\n".encode()
                )
                source.working_files["app.py"] = source.files["app.py"]

                with self.assertRaisesRegex(publisher.DeploymentError, "read-only"):
                    publisher.capture_source_bundle(source, SOURCE_REVISION)


class IdentityGateTests(unittest.TestCase):
    def test_documented_read_token_shape_is_accepted_for_owner_only_allowlist(self):
        username = publisher.verify_runtime_identity(
            whoami("read"),
            collaborators=("amitbcp",),
        )

        self.assertEqual(username, "amitbcp")

    def test_write_fine_grained_oauth_and_malformed_runtime_tokens_are_rejected(self):
        bad_profiles = (
            whoami("write"),
            whoami("fineGrained"),
            {"name": "amitbcp", "auth": {"type": "oauth"}},
            {"name": "amitbcp", "auth": {}},
        )
        for profile in bad_profiles:
            with self.subTest(profile=profile):
                with self.assertRaisesRegex(publisher.DeploymentError, "read-only"):
                    publisher.verify_runtime_identity(
                        profile,
                        collaborators=("amitbcp",),
                    )

    def test_personal_namespace_allowlist_is_explicit_and_owner_only(self):
        for collaborators in ((), ("other",), ("amitbcp", "other")):
            with self.subTest(collaborators=collaborators):
                with self.assertRaisesRegex(publisher.DeploymentError, "allowlist"):
                    publisher.verify_runtime_identity(
                        whoami("read"),
                        collaborators=collaborators,
                    )

    def test_deploy_token_must_be_documented_owner_write_token(self):
        self.assertEqual(publisher.verify_deploy_identity(whoami("write")), "amitbcp")
        for profile in (
            whoami("read"),
            whoami("fineGrained"),
            whoami("write", "other"),
        ):
            with self.subTest(profile=profile):
                with self.assertRaises(publisher.DeploymentError):
                    publisher.verify_deploy_identity(profile)

    def test_denied_probe_identity_must_be_documented_and_outside_allowlist(self):
        self.assertEqual(
            publisher.verify_denied_identity(
                whoami("read", "outside-reviewer"),
                collaborators=("amitbcp",),
            ),
            "outside-reviewer",
        )
        self.assertEqual(
            publisher.verify_denied_identity(
                whoami("write", "outside-reviewer"),
                collaborators=("amitbcp",),
            ),
            "outside-reviewer",
        )
        for profile in (
            whoami("read"),
            {"name": "outside-reviewer", "auth": {"type": "oauth"}},
        ):
            with self.subTest(profile=profile):
                with self.assertRaisesRegex(publisher.DeploymentError, "denied-probe"):
                    publisher.verify_denied_identity(
                        profile,
                        collaborators=("amitbcp",),
                    )


class RequestGateTests(unittest.TestCase):
    def valid_request(self, **changes):
        values = {
            "expected_source_revision": SOURCE_REVISION,
            "expected_private_revision": PRIVATE_REVISION,
            "expected_space_parent": SPACE_PARENT,
            "expect_absent": False,
            "visibility": "private",
            "collaborators": ("amitbcp",),
            "publish": False,
            "verify_only": False,
            "confirmation": None,
        }
        values.update(changes)
        return publisher.DeploymentRequest(**values)

    def test_request_requires_exact_existing_or_explicit_absent_state(self):
        publisher.validate_request(self.valid_request())
        publisher.validate_request(
            self.valid_request(expected_space_parent=None, expect_absent=True)
        )

        for request in (
            self.valid_request(expected_space_parent=None),
            self.valid_request(expect_absent=True),
            self.valid_request(expected_space_parent="short"),
        ):
            with self.subTest(request=request):
                with self.assertRaises(publisher.DeploymentError):
                    publisher.validate_request(request)

    def test_request_rejects_public_visibility_and_publish_without_confirmation(self):
        with self.assertRaisesRegex(publisher.DeploymentError, "private"):
            publisher.validate_request(self.valid_request(visibility="public"))
        with self.assertRaisesRegex(publisher.DeploymentError, "confirmation"):
            publisher.validate_request(self.valid_request(publish=True))
        publisher.validate_request(
            self.valid_request(
                publish=True,
                confirmation=publisher.PUBLISH_CONFIRMATION,
            )
        )

        with self.assertRaisesRegex(publisher.DeploymentError, "verify-only"):
            publisher.validate_request(
                self.valid_request(publish=True, verify_only=True)
            )
        with self.assertRaisesRegex(publisher.DeploymentError, "existing"):
            publisher.validate_request(
                self.valid_request(
                    expected_space_parent=None,
                    expect_absent=True,
                    verify_only=True,
                )
            )

    def test_all_three_tokens_must_be_distinct_and_never_rendered(self):
        with self.assertRaisesRegex(publisher.DeploymentError, "separate"):
            publisher.require_separate_tokens(WRITE_TOKEN, WRITE_TOKEN, DENIED_TOKEN)
        with self.assertRaisesRegex(publisher.DeploymentError, "separate"):
            publisher.require_separate_tokens(WRITE_TOKEN, READ_TOKEN, READ_TOKEN)

        request = self.valid_request()
        rendered = repr(request)
        self.assertNotIn(WRITE_TOKEN, rendered)
        self.assertNotIn(READ_TOKEN, rendered)


class DeploymentWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.source = FakeSourceBackend()
        self.hub = FakeHubBackend(self.source)

    def execute(self, request=None, snapshot_auditor=None):
        return publisher.run_deployment(
            request or valid_request(),
            deploy_token=WRITE_TOKEN,
            runtime_token=READ_TOKEN,
            denied_token=DENIED_TOKEN,
            source_backend=self.source,
            hub_backend=self.hub,
            snapshot_auditor=snapshot_auditor,
        )

    def test_dry_run_inspects_every_boundary_without_writing_or_leaking_tokens(self):
        result = self.execute()

        self.assertFalse(result.published)
        self.assertEqual(result.action, "update")
        self.assertEqual(result.space_revision, SPACE_PARENT)
        self.assertEqual(result.private_dataset_revision, PRIVATE_REVISION)
        self.assertEqual(result.organizer_reconciliation, "disabled/no-release")
        self.assertTrue(result.participant_test_submissions_disabled)
        self.assertTrue(result.participant_final_leaderboard_disabled)
        self.assertEqual(self.hub.writes, [])
        self.assertEqual(self.hub.secret_updates, [])
        rendered = publisher.render_result(result)
        for secret in (WRITE_TOKEN, READ_TOKEN, DENIED_TOKEN):
            self.assertNotIn(secret, rendered)
            self.assertNotIn(secret, repr(result))

    def test_existing_space_must_match_private_gradio_parent(self):
        invalid_states = (
            publisher.SpaceState(
                True,
                SPACE_PARENT,
                False,
                "gradio",
                "https://organizer.example.test",
                "RUNNING",
            ),
            publisher.SpaceState(
                True,
                SPACE_PARENT,
                True,
                "docker",
                "https://organizer.example.test",
                "RUNNING",
            ),
            publisher.SpaceState(
                True,
                "f" * 40,
                True,
                "gradio",
                "https://organizer.example.test",
                "RUNNING",
            ),
        )
        for state in invalid_states:
            with self.subTest(state=state):
                self.hub.organizer = state
                with self.assertRaises(publisher.DeploymentError):
                    self.execute()

    def test_missing_space_requires_explicit_absent_expectation(self):
        self.hub.organizer = publisher.SpaceState.missing()
        with self.assertRaisesRegex(publisher.DeploymentError, "absent"):
            self.execute()

        result = self.execute(
            valid_request(expected_space_parent=None, expect_absent=True)
        )
        self.assertEqual(result.action, "create")
        self.assertFalse(result.published)
        self.assertEqual(self.hub.create_calls, [])

    def test_runtime_read_token_must_access_exact_private_dataset_revision(self):
        self.hub.dataset = publisher.DatasetState(
            revision="f" * 40,
            private=True,
        )
        with self.assertRaisesRegex(publisher.DeploymentError, "revision"):
            self.execute()

        self.hub.dataset = publisher.DatasetState(
            revision=PRIVATE_REVISION,
            private=False,
        )
        with self.assertRaisesRegex(publisher.DeploymentError, "private"):
            self.execute()

    def test_reserved_space_variables_are_rejected_before_secret_updates(self):
        for name in ("ORGANIZER_READ_TOKEN", "PRIVATE_REPO_ID"):
            with self.subTest(name=name):
                self.hub.variables = {name: "wrong-channel"}
                with self.assertRaisesRegex(publisher.DeploymentError, "variable"):
                    self.execute()

    def test_present_release_uses_exact_snapshot_loader_and_reports_counts(self):
        self.hub.dataset_files = (
            "private/test_release.json",
            "attempts/test/abc/record.json",
        )
        calls = []

        def audit(repo_id, revision, token, *, api):
            calls.append((repo_id, revision, token, api))
            return publisher.SnapshotAudit(
                status="verified",
                account_count=2,
                attempt_count=3,
            )

        result = self.execute(snapshot_auditor=audit)

        self.assertEqual(
            calls,
            [
                (
                    publisher.PRIVATE_DATASET_REPO_ID,
                    PRIVATE_REVISION,
                    READ_TOKEN,
                    self.hub,
                )
            ],
        )
        self.assertEqual(result.organizer_reconciliation, "verified")
        self.assertEqual(result.organizer_account_count, 2)
        self.assertEqual(result.organizer_attempt_count, 3)

    def test_absent_release_rejects_orphaned_test_ledger_state(self):
        for path in (
            "private/test_labels.jsonl",
            "private/test_finalization_audit.json",
            "private/test_future",
            "attempts/test",
            "attempts/test/account/record.json",
            "projections/test",
            "projections/test/accounts/account.json",
            "projections/test/organizer_leaderboard.json",
            "exclusions/test",
            "exclusions/test/example.json",
            "adjudications/test",
            "adjudications/test/example.json",
        ):
            with self.subTest(path=path):
                self.hub.dataset_files = (path,)
                with self.assertRaisesRegex(publisher.DeploymentError, "release"):
                    self.execute()

    def test_verify_only_is_zero_write_and_performs_full_three_identity_probe(self):
        self.hub.space_tree = dict(self.source.files)
        result = self.execute(valid_request(verify_only=True))

        self.assertEqual(result.outcome, "verified")
        self.assertEqual(result.action, "verify")
        self.assertEqual(result.runtime_access, "verified")
        self.assertFalse(result.published)
        self.assertEqual(self.hub.writes, [])
        self.assertEqual(self.hub.secret_updates, [])
        self.assertEqual(self.hub.create_calls, [])
        host = self.hub.organizer.host
        for path in ("/", "/config", "/info"):
            self.assertIn(("GET", host + path, None, None), self.hub.request_calls)
            self.assertIn(
                ("GET", host + path, DENIED_TOKEN, None), self.hub.request_calls
            )
            self.assertIn(
                ("GET", host + path, READ_TOKEN, None), self.hub.request_calls
            )

    def test_publish_pending_then_verify_only_converges_without_second_write(self):
        self.hub.commit_runtime_stage = "BUILDING"
        request = valid_request(
            publish=True,
            confirmation=publisher.PUBLISH_CONFIRMATION,
        )
        pending = self.execute(request)

        self.assertEqual(pending.outcome, "published-pending-verification")
        self.assertEqual(pending.runtime_access, "pending")
        self.assertNotEqual(pending.outcome, "published-verified")
        self.assertEqual(len(self.hub.writes), 1)
        first_secret_updates = list(self.hub.secret_updates)
        self.hub.organizer = publisher.SpaceState(
            exists=True,
            revision=pending.space_revision,
            private=True,
            sdk="gradio",
            host="https://organizer.example.test",
            runtime_stage="RUNNING",
        )

        verified = self.execute(
            valid_request(
                expected_space_parent=pending.space_revision,
                verify_only=True,
            )
        )

        self.assertEqual(verified.outcome, "verified")
        self.assertEqual(verified.runtime_access, "verified")
        self.assertEqual(len(self.hub.writes), 1)
        self.assertEqual(self.hub.secret_updates, first_secret_updates)

    def test_verify_only_refuses_to_claim_success_before_runtime_is_running(self):
        self.hub.space_tree = dict(self.source.files)
        self.hub.organizer = publisher.SpaceState(
            exists=True,
            revision=SPACE_PARENT,
            private=True,
            sdk="gradio",
            host="https://organizer.example.test",
            runtime_stage="BUILDING",
        )

        with self.assertRaisesRegex(publisher.DeploymentError, "RUNNING"):
            self.execute(valid_request(verify_only=True))

        self.assertEqual(self.hub.writes, [])
        self.assertEqual(self.hub.secret_updates, [])

    def test_verify_only_requires_denied_identity_to_be_denied_at_every_endpoint(self):
        self.hub.space_tree = dict(self.source.files)
        original = self.hub.request

        def outsider_can_read(method, url, *, token=None, json_body=None):
            if token == DENIED_TOKEN and url.endswith("/info"):
                return publisher.HttpResponse(200, b'{"named_endpoints":{}}', {})
            return original(method, url, token=token, json_body=json_body)

        self.hub.request = outsider_can_read
        with self.assertRaisesRegex(publisher.DeploymentError, "Outside"):
            self.execute(valid_request(verify_only=True))

        self.assertEqual(self.hub.writes, [])
        self.assertEqual(self.hub.secret_updates, [])

    def test_publish_updates_exact_tree_sets_only_two_secrets_and_verifies_access(self):
        request = valid_request(
            publish=True,
            confirmation=publisher.PUBLISH_CONFIRMATION,
        )

        result = self.execute(request)

        self.assertTrue(result.published)
        self.assertEqual(result.outcome, "published-verified")
        self.assertEqual(result.action, "update")
        self.assertEqual(result.space_revision, "d" * 40)
        self.assertEqual(len(self.hub.writes), 1)
        write = self.hub.writes[0]
        self.assertEqual(write["parent"], SPACE_PARENT)
        self.assertEqual(tuple(write["additions"]), publisher.BUNDLE_PATHS)
        self.assertEqual(
            write["deletions"],
            ("__pycache__/app.pyc", "test_app.py"),
        )
        self.assertEqual(
            tuple(sorted(self.hub.space_tree)), tuple(sorted(publisher.BUNDLE_PATHS))
        )
        self.assertEqual(
            self.hub.secret_updates,
            ["ORGANIZER_READ_TOKEN", "PRIVATE_REPO_ID"],
        )
        self.assertEqual(self.hub.secrets["EXISTING_PRIVATE_SECRET"], "preserve")
        self.assertEqual(self.hub.secrets["ORGANIZER_READ_TOKEN"], READ_TOKEN)
        self.assertEqual(
            self.hub.secrets["PRIVATE_REPO_ID"],
            publisher.PRIVATE_DATASET_REPO_ID,
        )
        self.assertNotIn(DENIED_TOKEN, self.hub.secrets.values())

    def test_publish_can_create_only_an_explicit_private_space(self):
        self.hub = FakeHubBackend(self.source, space_exists=False)
        request = valid_request(
            expected_space_parent=None,
            expect_absent=True,
            publish=True,
            confirmation=publisher.PUBLISH_CONFIRMATION,
        )

        result = self.execute(request)

        self.assertTrue(result.published)
        self.assertEqual(result.action, "create")
        self.assertEqual(
            self.hub.create_calls,
            [
                {
                    "repo_id": publisher.SPACE_REPO_ID,
                    "private": True,
                    "sdk": "gradio",
                }
            ],
        )

    def test_publish_rechecks_clean_source_immediately_before_commit(self):
        request = valid_request(
            publish=True,
            confirmation=publisher.PUBLISH_CONFIRMATION,
        )
        self.source.move_on_second_inspect = True

        with self.assertRaisesRegex(publisher.DeploymentError, "source revision"):
            self.execute(request)

    def test_postdeploy_rejects_named_endpoint_or_secret_in_organizer_config(self):
        request = valid_request(
            publish=True,
            confirmation=publisher.PUBLISH_CONFIRMATION,
        )
        self.hub.organizer_info = {
            "named_endpoints": {"/finalize": {}},
            "unnamed_endpoints": {},
        }
        with self.assertRaisesRegex(publisher.DeploymentError, "endpoint"):
            self.execute(request)

        self.setUp()
        self.hub.organizer_config["leak"] = READ_TOKEN
        with self.assertRaisesRegex(publisher.DeploymentError, "configuration"):
            self.execute(request)

        self.setUp()
        self.hub.organizer_config["leak"] = WRITE_TOKEN
        with self.assertRaisesRegex(publisher.DeploymentError, "configuration"):
            self.execute(request)

        self.setUp()
        self.hub.organizer_config["components"] = [
            {
                "type": "button",
                "props": {"value": "Finalize leaderboard", "interactive": True},
            }
        ]
        with self.assertRaisesRegex(publisher.DeploymentError, "mutation"):
            self.execute(request)

    def test_postdeploy_scans_root_config_and_info_for_every_forbidden_value(self):
        forbidden = (
            WRITE_TOKEN,
            READ_TOKEN,
            DENIED_TOKEN,
            publisher.PRIVATE_DATASET_REPO_ID,
            "ORGANIZER_READ_TOKEN",
            "PRIVATE_REPO_ID",
            "DOCSEM_ORGANIZER_DEPLOY_TOKEN",
            "DOCSEM_ORGANIZER_READ_TOKEN",
            "DOCSEM_ORGANIZER_DENIED_TOKEN",
            "HF_WRITE_TOKEN",
        )
        request = valid_request(
            publish=True,
            confirmation=publisher.PUBLISH_CONFIRMATION,
        )
        for path in ("/", "/config", "/info"):
            for value in forbidden:
                with self.subTest(path=path, value=value):
                    self.setUp()
                    original = self.hub.request

                    def leaking(
                        method,
                        url,
                        *,
                        token=None,
                        json_body=None,
                        _original=original,
                        _path=path,
                        _value=value,
                    ):
                        response = _original(
                            method,
                            url,
                            token=token,
                            json_body=json_body,
                        )
                        if token == READ_TOKEN and url.endswith(_path):
                            if _path == "/":
                                body = response.body + _value.encode("utf-8")
                            else:
                                body_value = json.loads(response.body)
                                body_value["leak"] = _value
                                body = json.dumps(body_value).encode("utf-8")
                            return publisher.HttpResponse(200, body, {})
                        return response

                    self.hub.request = leaking
                    with self.assertRaisesRegex(
                        publisher.DeploymentError, "exposes a secret"
                    ):
                        self.execute(request)

    def test_denied_probe_must_be_available_before_any_publish_write(self):
        del self.hub.identities[DENIED_TOKEN]
        request = valid_request(
            publish=True,
            confirmation=publisher.PUBLISH_CONFIRMATION,
        )

        with self.assertRaises(publisher.DeploymentError):
            self.execute(request)

        self.assertEqual(self.hub.writes, [])
        self.assertEqual(self.hub.secret_updates, [])

    def test_postdeploy_rejects_reserved_name_becoming_public_variable(self):
        self.hub.reserved_variable_after_secret = True
        request = valid_request(
            publish=True,
            confirmation=publisher.PUBLISH_CONFIRMATION,
        )

        with self.assertRaisesRegex(publisher.DeploymentError, "variable"):
            self.execute(request)

    def test_participant_reconciliation_rejects_open_test_or_final_rows(self):
        self.hub.participant_config["components"][0]["props"]["value"] = "Test (final)"
        with self.assertRaisesRegex(publisher.DeploymentError, "participant"):
            self.execute()

        self.setUp()
        original = self.hub.request

        def open_test(method, url, **kwargs):
            response = original(method, url, **kwargs)
            if url.endswith("/api/select_split"):
                body = json.loads(response.body)
                body["data"][2]["interactive"] = True
                return publisher.HttpResponse(200, json.dumps(body).encode(), {})
            return response

        self.hub.request = open_test
        with self.assertRaisesRegex(publisher.DeploymentError, "participant"):
            self.execute()

        self.setUp()
        original = self.hub.request

        def final_rows(method, url, **kwargs):
            response = original(method, url, **kwargs)
            if method == "POST" and url.endswith("/api/predict"):
                body = json.loads(response.body)
                body["data"][1]["value"] = "<table><tr><td>99</td></tr></table>"
                body["data"][2]["visible"] = True
                return publisher.HttpResponse(200, json.dumps(body).encode(), {})
            return response

        self.hub.request = final_rows
        with self.assertRaisesRegex(publisher.DeploymentError, "participant"):
            self.execute()


class FakeApi:
    def __init__(self):
        self.calls = []
        self.space_sha = SPACE_PARENT
        self.dataset_sha = PRIVATE_REVISION
        self.space_tree = {"old.txt": b"old\n"}
        self.variables = {"PRESERVE": SimpleNamespace(value="yes")}

    def whoami(self, *, token):
        self.calls.append(("whoami", token))
        return whoami("read" if token == READ_TOKEN else "write")

    def space_info(self, repo_id, *, token):
        self.calls.append(("space_info", repo_id, token))
        return SimpleNamespace(
            sha=self.space_sha,
            private=True,
            sdk="gradio",
            host="https://organizer.example.test",
            runtime=SimpleNamespace(stage="RUNNING"),
        )

    def repo_info(self, repo_id, *, repo_type, revision, token):
        self.calls.append(("repo_info", repo_id, repo_type, revision, token))
        return SimpleNamespace(sha=self.dataset_sha, private=True)

    def list_repo_files(self, repo_id, *, repo_type, revision, token):
        self.calls.append(("list_repo_files", repo_id, repo_type, revision, token))
        return tuple(sorted(self.space_tree))

    def get_space_variables(self, repo_id, *, token):
        self.calls.append(("get_space_variables", repo_id, token))
        return self.variables

    def create_repo(self, **kwargs):
        self.calls.append(("create_repo", kwargs))
        self.space_sha = "1" * 40
        return SimpleNamespace(repo_id=publisher.SPACE_REPO_ID)

    def create_commit(self, **kwargs):
        self.calls.append(("create_commit", kwargs))
        for operation in kwargs["operations"]:
            if operation.__class__.__name__ == "CommitOperationDelete":
                self.space_tree.pop(operation.path_in_repo, None)
            else:
                self.space_tree[operation.path_in_repo] = operation.path_or_fileobj
        self.space_sha = "d" * 40
        return SimpleNamespace(oid=self.space_sha)

    def add_space_secret(self, repo_id, key, value, *, token):
        self.calls.append(("add_space_secret", repo_id, key, value, token))

    def list_repo_commits(self, *args, **kwargs):
        self.calls.append(("list_repo_commits", args, kwargs))
        return ()

    def hf_hub_download(self, *args, **kwargs):
        self.calls.append(("hf_hub_download", args, kwargs))
        raise AssertionError("injected download should be used")


class FakeHttpSession:
    def __init__(self):
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return SimpleNamespace(
            status_code=200,
            content=b'{"ok":true}',
            headers={"Content-Type": "application/json"},
        )


class HuggingFaceAdapterTests(unittest.TestCase):
    def setUp(self):
        self.api = FakeApi()
        self.session = FakeHttpSession()
        self.backend = publisher.HuggingFaceBackend(
            api_factory=lambda token: self.api,
            download=lambda **kwargs: self._download(**kwargs),
            session=self.session,
        )

    def _download(self, **kwargs):
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.api.space_tree[kwargs["filename"]])
        return str(path)

    def test_adapter_uses_documented_private_space_and_exact_parent_calls(self):
        state = self.backend.inspect_space(publisher.SPACE_REPO_ID, WRITE_TOKEN)
        self.assertEqual(state.revision, SPACE_PARENT)
        self.assertTrue(state.private)
        self.assertEqual(state.sdk, "gradio")

        self.backend.create_private_space(publisher.SPACE_REPO_ID, WRITE_TOKEN)
        create = next(item[1] for item in self.api.calls if item[0] == "create_repo")
        self.assertEqual(
            create,
            {
                "repo_id": publisher.SPACE_REPO_ID,
                "repo_type": "space",
                "private": True,
                "space_sdk": "gradio",
                "exist_ok": False,
                "token": WRITE_TOKEN,
            },
        )

        revision = self.backend.commit_space(
            publisher.SPACE_REPO_ID,
            "1" * 40,
            {name: f"bundle::{name}\n".encode() for name in publisher.BUNDLE_PATHS},
            ("old.txt",),
            WRITE_TOKEN,
        )
        self.assertEqual(revision, "d" * 40)
        commit = next(item[1] for item in self.api.calls if item[0] == "create_commit")
        self.assertEqual(commit["repo_id"], publisher.SPACE_REPO_ID)
        self.assertEqual(commit["repo_type"], "space")
        self.assertEqual(commit["revision"], "main")
        self.assertEqual(commit["parent_commit"], "1" * 40)
        self.assertFalse(commit["create_pr"])

    def test_adapter_resolves_private_dataset_head_before_pinning_snapshot(self):
        state = self.backend.inspect_dataset(
            publisher.PRIVATE_DATASET_REPO_ID,
            PRIVATE_REVISION,
            READ_TOKEN,
        )

        self.assertEqual(state.revision, PRIVATE_REVISION)
        call = next(item for item in self.api.calls if item[0] == "repo_info")
        self.assertEqual(
            call,
            (
                "repo_info",
                publisher.PRIVATE_DATASET_REPO_ID,
                "dataset",
                "main",
                READ_TOKEN,
            ),
        )

    def test_adapter_sets_one_secret_without_reading_or_deleting_others(self):
        self.backend.set_space_secret(
            publisher.SPACE_REPO_ID,
            "ORGANIZER_READ_TOKEN",
            READ_TOKEN,
            WRITE_TOKEN,
        )

        matching = [item for item in self.api.calls if item[0] == "add_space_secret"]
        self.assertEqual(
            matching,
            [
                (
                    "add_space_secret",
                    publisher.SPACE_REPO_ID,
                    "ORGANIZER_READ_TOKEN",
                    READ_TOKEN,
                    WRITE_TOKEN,
                )
            ],
        )
        self.assertFalse(any("delete" in item[0] for item in self.api.calls))

    def test_adapter_http_auth_is_per_request_and_response_is_bounded(self):
        response = self.backend.request(
            "POST",
            "https://organizer.example.test/config",
            token=READ_TOKEN,
            json_body={"data": []},
        )

        self.assertEqual(response.status_code, 200)
        method, _, kwargs = self.session.calls[-1]
        self.assertEqual(method, "POST")
        self.assertEqual(kwargs["headers"], {"Authorization": f"Bearer {READ_TOKEN}"})
        self.assertEqual(kwargs["json"], {"data": []})
        self.assertFalse(kwargs["allow_redirects"])

    def test_adapter_rejects_unknown_secret_name(self):
        with self.assertRaisesRegex(publisher.DeploymentError, "secret"):
            self.backend.set_space_secret(
                publisher.SPACE_REPO_ID,
                "HF_WRITE_TOKEN",
                WRITE_TOKEN,
                WRITE_TOKEN,
            )

    def test_adapter_refuses_every_mutation_outside_exact_organizer_space(self):
        with self.assertRaisesRegex(publisher.DeploymentError, "target"):
            self.backend.create_private_space("amitbcp/other-space", WRITE_TOKEN)
        with self.assertRaisesRegex(publisher.DeploymentError, "target"):
            self.backend.set_space_secret(
                "amitbcp/other-space",
                "ORGANIZER_READ_TOKEN",
                READ_TOKEN,
                WRITE_TOKEN,
            )
        with self.assertRaisesRegex(publisher.DeploymentError, "target"):
            self.backend.commit_space(
                "amitbcp/other-space",
                SPACE_PARENT,
                {name: f"bundle::{name}\n".encode() for name in publisher.BUNDLE_PATHS},
                (),
                WRITE_TOKEN,
            )


class ImportAndCliTests(unittest.TestCase):
    def test_cli_defaults_to_dry_run_and_passes_no_token_on_argv(self):
        result = publisher._request_from_argv(
            [
                "--expected-source-revision",
                SOURCE_REVISION,
                "--expected-private-revision",
                PRIVATE_REVISION,
                "--expected-space-parent",
                SPACE_PARENT,
                "--visibility",
                "private",
                "--collaborator",
                "amitbcp",
            ]
        )

        self.assertFalse(result.publish)
        self.assertFalse(result.verify_only)
        self.assertIsNone(result.confirmation)

    def test_cli_verify_only_is_explicit_and_mutually_exclusive_with_publish(self):
        args = [
            "--expected-source-revision",
            SOURCE_REVISION,
            "--expected-private-revision",
            PRIVATE_REVISION,
            "--expected-space-parent",
            SPACE_PARENT,
            "--visibility",
            "private",
            "--collaborator",
            "amitbcp",
            "--verify-only",
        ]
        request = publisher._request_from_argv(args)
        self.assertTrue(request.verify_only)
        self.assertFalse(request.publish)
        with self.assertRaises(SystemExit):
            publisher._request_from_argv(args + ["--publish"])

    def test_main_uses_only_dedicated_token_environment_names_and_safe_output(self):
        expected = publisher.DeploymentResult(
            published=False,
            action="update",
            outcome="dry-run",
            source_revision=SOURCE_REVISION,
            bundle_tree_sha256="f" * 64,
            space_revision=SPACE_PARENT,
            private_dataset_revision=PRIVATE_REVISION,
            organizer_reconciliation="disabled/no-release",
            organizer_account_count=0,
            organizer_attempt_count=0,
            participant_test_submissions_disabled=True,
            participant_final_leaderboard_disabled=True,
            runtime_access="not-probed",
        )
        output = io.StringIO()
        error = io.StringIO()
        argv = [
            "--expected-source-revision",
            SOURCE_REVISION,
            "--expected-private-revision",
            PRIVATE_REVISION,
            "--expected-space-parent",
            SPACE_PARENT,
            "--visibility",
            "private",
            "--collaborator",
            "amitbcp",
        ]
        environment = {
            "DOCSEM_ORGANIZER_DEPLOY_TOKEN": WRITE_TOKEN,
            "DOCSEM_ORGANIZER_READ_TOKEN": READ_TOKEN,
            "DOCSEM_ORGANIZER_DENIED_TOKEN": DENIED_TOKEN,
            "HF_TOKEN": "must-not-be-used",
        }
        with mock.patch.object(
            publisher, "run_deployment", return_value=expected
        ) as run:
            status = publisher.main(
                argv,
                environment=environment,
                stdout=output,
                stderr=error,
            )

        self.assertEqual(status, 0)
        self.assertEqual(error.getvalue(), "")
        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["deploy_token"], WRITE_TOKEN)
        self.assertEqual(kwargs["runtime_token"], READ_TOKEN)
        self.assertEqual(kwargs["denied_token"], DENIED_TOKEN)
        rendered = output.getvalue()
        for secret in (*environment.values(),):
            self.assertNotIn(secret, rendered)

    def test_main_returns_distinct_nonzero_pending_receipt_with_verify_instruction(
        self,
    ):
        pending = publisher.DeploymentResult(
            published=True,
            action="update",
            outcome="published-pending-verification",
            source_revision=SOURCE_REVISION,
            bundle_tree_sha256="f" * 64,
            space_revision="d" * 40,
            private_dataset_revision=PRIVATE_REVISION,
            organizer_reconciliation="disabled/no-release",
            organizer_account_count=0,
            organizer_attempt_count=0,
            participant_test_submissions_disabled=True,
            participant_final_leaderboard_disabled=True,
            runtime_access="pending",
        )
        output = io.StringIO()
        error = io.StringIO()
        argv = [
            "--expected-source-revision",
            SOURCE_REVISION,
            "--expected-private-revision",
            PRIVATE_REVISION,
            "--expected-space-parent",
            SPACE_PARENT,
            "--visibility",
            "private",
            "--collaborator",
            "amitbcp",
            "--publish",
            "--confirm",
            publisher.PUBLISH_CONFIRMATION,
        ]
        environment = {
            "DOCSEM_ORGANIZER_DEPLOY_TOKEN": WRITE_TOKEN,
            "DOCSEM_ORGANIZER_READ_TOKEN": READ_TOKEN,
            "DOCSEM_ORGANIZER_DENIED_TOKEN": DENIED_TOKEN,
        }
        with mock.patch.object(publisher, "run_deployment", return_value=pending):
            status = publisher.main(
                argv,
                environment=environment,
                stdout=output,
                stderr=error,
            )

        self.assertEqual(status, 3)
        self.assertEqual(error.getvalue(), "")
        receipt = json.loads(output.getvalue())
        self.assertEqual(receipt["outcome"], "published-pending-verification")
        self.assertFalse(receipt["verification_complete"])
        self.assertEqual(receipt["space_revision"], "d" * 40)
        self.assertIn("--verify-only", receipt["next_action"])
        for secret in environment.values():
            self.assertNotIn(secret, output.getvalue())

    def test_main_missing_denied_probe_token_is_incomplete_and_never_runs(self):
        output = io.StringIO()
        error = io.StringIO()
        argv = [
            "--expected-source-revision",
            SOURCE_REVISION,
            "--expected-private-revision",
            PRIVATE_REVISION,
            "--expected-space-parent",
            SPACE_PARENT,
            "--visibility",
            "private",
            "--collaborator",
            "amitbcp",
            "--verify-only",
        ]
        environment = {
            "DOCSEM_ORGANIZER_DEPLOY_TOKEN": WRITE_TOKEN,
            "DOCSEM_ORGANIZER_READ_TOKEN": READ_TOKEN,
        }
        with mock.patch.object(publisher, "run_deployment") as run:
            status = publisher.main(
                argv,
                environment=environment,
                stdout=output,
                stderr=error,
            )

        self.assertEqual(status, 2)
        self.assertEqual(output.getvalue(), "")
        self.assertIn("three", error.getvalue().casefold())
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
