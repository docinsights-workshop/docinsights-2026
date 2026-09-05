#!/usr/bin/env python3
"""Guarded deployment for the private DocSem organizer-only Space.

The command is a dry run unless ``--publish`` and the exact confirmation are
both supplied.  It deploys only the five tracked production files from one
clean, exact Git revision and never mutates participant or dataset repos.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "competition" / "hf-organizer-space"
SPACE_REPO_ID = "amitbcp/docsem-docinsights-organizer"
PRIVATE_DATASET_REPO_ID = "amitbcp/docinsights-2026-shared-task-submissions"
PARTICIPANT_SPACE_REPO_ID = "amitbcp/docsem-docinsights"
SPACE_OWNER = "amitbcp"
PUBLISH_CONFIRMATION = "PUBLISH_PRIVATE_ORGANIZER_SPACE"
BUNDLE_PATHS = (
    "README.md",
    "app.py",
    "organizer_data.py",
    "organizer_contract.py",
    "requirements.txt",
)

_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_USERNAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_RESERVED_SECRET_NAMES = ("ORGANIZER_READ_TOKEN", "PRIVATE_REPO_ID")
_READY_STAGE = "RUNNING"
_MAX_HTTP_BYTES = 4 * 1024 * 1024
_MAX_BUNDLE_FILE_BYTES = 4 * 1024 * 1024
_WRITE_CAPABILITY = re.compile(
    rb"\b(?:create_commit|create_repo|upload_file|upload_folder|delete_file|"
    rb"delete_repo|update_repo_settings|add_space_secret|delete_space_secret|"
    rb"add_space_variable|delete_space_variable|restart_space|pause_space|"
    rb"request_space_hardware|request_space_storage|grant_access|"
    rb"accept_access_request)\s*\("
)
_MUTATION_CONTROL = re.compile(
    r"(?:submit|finaliz|upload|delete|commit|write|mutat|repair|reset)",
    re.IGNORECASE,
)


class DeploymentError(RuntimeError):
    """Sanitized refusal at the organizer deployment boundary."""


@dataclass(frozen=True)
class SourceState:
    root: Path
    revision: str
    clean: bool


@dataclass(frozen=True)
class SpaceState:
    exists: bool
    revision: str | None
    private: bool | None
    sdk: str | None
    host: str | None
    runtime_stage: str | None

    @classmethod
    def missing(cls) -> "SpaceState":
        return cls(False, None, None, None, None, None)


@dataclass(frozen=True)
class DatasetState:
    revision: str
    private: bool


@dataclass(frozen=True, repr=False)
class HttpResponse:
    status_code: int
    body: bytes = field(repr=False, compare=True)
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)

    def __repr__(self) -> str:
        return f"HttpResponse(status_code={self.status_code}, size={len(self.body)})"


@dataclass(frozen=True)
class SnapshotAudit:
    status: str
    account_count: int
    attempt_count: int


@dataclass(frozen=True)
class DeploymentResult:
    published: bool
    action: str
    source_revision: str
    bundle_tree_sha256: str
    space_revision: str | None
    private_dataset_revision: str
    organizer_reconciliation: str
    organizer_account_count: int
    organizer_attempt_count: int
    participant_test_submissions_disabled: bool
    participant_final_leaderboard_disabled: bool
    runtime_access: str


@dataclass(frozen=True, repr=False)
class BundleFile:
    payload: bytes = field(repr=False, compare=True)
    sha256: str
    size: int

    def __repr__(self) -> str:
        return f"BundleFile(size={self.size}, sha256={self.sha256!r})"


@dataclass(frozen=True, repr=False)
class SourceBundle:
    source_revision: str
    files: Mapping[str, BundleFile] = field(repr=False, compare=True)
    tree_sha256: str

    def __repr__(self) -> str:
        return (
            "SourceBundle("
            f"source_revision={self.source_revision!r}, "
            f"file_count={len(self.files)}, tree_sha256={self.tree_sha256!r})"
        )


@dataclass(frozen=True)
class DeploymentRequest:
    expected_source_revision: str
    expected_private_revision: str
    expected_space_parent: str | None
    expect_absent: bool
    visibility: str
    collaborators: tuple[str, ...]
    publish: bool = False
    confirmation: str | None = field(default=None, repr=False)


class SourceBackend(Protocol):
    def inspect_source(self, source_root: Path) -> SourceState: ...

    def read_revision_file(
        self, source_root: Path, revision: str, relative_path: str
    ) -> bytes: ...

    def read_worktree_file(self, source_root: Path, relative_path: str) -> bytes: ...


class HubBackend(Protocol):
    def whoami(self, token: str) -> Mapping[str, object]: ...

    def inspect_space(self, repo_id: str, token: str) -> SpaceState: ...

    def inspect_dataset(
        self, repo_id: str, revision: str, token: str
    ) -> DatasetState: ...

    def list_dataset_files(
        self, repo_id: str, revision: str, token: str
    ) -> Sequence[str]: ...

    def get_space_variables(self, repo_id: str, token: str) -> Mapping[str, object]: ...

    def create_private_space(self, repo_id: str, token: str) -> SpaceState: ...

    def list_space_files(
        self, repo_id: str, revision: str, token: str
    ) -> Sequence[str]: ...

    def read_space_files(
        self,
        repo_id: str,
        revision: str,
        paths: Sequence[str],
        token: str,
    ) -> Mapping[str, bytes]: ...

    def commit_space(
        self,
        repo_id: str,
        expected_parent: str,
        additions: Mapping[str, bytes],
        deletions: Sequence[str],
        token: str,
    ) -> str: ...

    def set_space_secret(
        self, repo_id: str, name: str, value: str, token: str
    ) -> None: ...

    def request(
        self,
        method: str,
        url: str,
        *,
        token: str | None = None,
        json_body: object | None = None,
    ) -> HttpResponse: ...


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _exact_revision(value: object, description: str) -> str:
    if not isinstance(value, str) or _REVISION.fullmatch(value) is None:
        raise DeploymentError(f"An exact {description} is required.")
    return value


def capture_source_bundle(
    backend: SourceBackend,
    expected_revision: str,
    *,
    source_root: Path = SOURCE_ROOT,
) -> SourceBundle:
    """Seal the exact five-file bundle from a clean pinned Git revision."""

    expected = _exact_revision(expected_revision, "source revision")
    try:
        state = backend.inspect_source(source_root)
    except DeploymentError:
        raise
    except Exception as exc:
        raise DeploymentError("The organizer source state is unavailable.") from exc
    if (
        not isinstance(state, SourceState)
        or Path(state.root).resolve() != Path(source_root).resolve()
        or state.revision != expected
    ):
        raise DeploymentError("The expected source revision is not checked out.")
    if state.clean is not True:
        raise DeploymentError("The organizer source worktree must be clean.")

    files: dict[str, BundleFile] = {}
    tree_hasher = hashlib.sha256()
    for relative_path in BUNDLE_PATHS:
        try:
            committed = backend.read_revision_file(source_root, expected, relative_path)
            working = backend.read_worktree_file(source_root, relative_path)
        except DeploymentError:
            raise
        except Exception as exc:
            raise DeploymentError(
                "An organizer bundle file is unavailable at the pinned revision."
            ) from exc
        if (
            not isinstance(committed, bytes)
            or not isinstance(working, bytes)
            or committed != working
            or len(committed) > _MAX_BUNDLE_FILE_BYTES
        ):
            raise DeploymentError(
                "Organizer bundle bytes do not match the pinned source revision."
            )
        digest = _sha256(committed)
        files[relative_path] = BundleFile(
            payload=committed,
            sha256=digest,
            size=len(committed),
        )
        tree_hasher.update(relative_path.encode("utf-8"))
        tree_hasher.update(b"\0")
        tree_hasher.update(len(committed).to_bytes(8, "big"))
        tree_hasher.update(committed)
    python_payload = b"\n".join(
        files[path].payload for path in BUNDLE_PATHS if path.endswith(".py")
    )
    if _WRITE_CAPABILITY.search(python_payload) or b"HF_WRITE_TOKEN" in python_payload:
        raise DeploymentError(
            "The organizer Space bundle is not a read-only application."
        )
    return SourceBundle(
        source_revision=expected,
        files=files,
        tree_sha256=tree_hasher.hexdigest(),
    )


def _documented_access_token_role(profile: object) -> tuple[str, str]:
    if not isinstance(profile, Mapping):
        raise DeploymentError("Hugging Face token identity is unavailable.")
    name = profile.get("name")
    auth = profile.get("auth")
    if (
        not isinstance(name, str)
        or _USERNAME.fullmatch(name) is None
        or not isinstance(auth, Mapping)
        or auth.get("type") != "access_token"
    ):
        raise DeploymentError("Hugging Face token identity is unavailable.")
    access_token = auth.get("accessToken")
    if not isinstance(access_token, Mapping):
        raise DeploymentError("Hugging Face token role is unavailable.")
    role = access_token.get("role")
    if not isinstance(role, str):
        raise DeploymentError("Hugging Face token role is unavailable.")
    return name, role


def _owner_only_allowlist(collaborators: Sequence[str]) -> tuple[str, ...]:
    if isinstance(collaborators, (str, bytes)) or tuple(collaborators) != (
        SPACE_OWNER,
    ):
        raise DeploymentError(
            "The personal-namespace collaborator allowlist must be owner-only."
        )
    return (SPACE_OWNER,)


def verify_runtime_identity(
    profile: object,
    *,
    collaborators: Sequence[str],
) -> str:
    """Accept only a documented classic read token for the owner-only Space."""

    allowlist = _owner_only_allowlist(collaborators)
    try:
        username, role = _documented_access_token_role(profile)
    except DeploymentError as exc:
        raise DeploymentError(
            "The organizer runtime token is not a proven read-only token."
        ) from exc
    if role != "read" or username not in allowlist:
        raise DeploymentError(
            "The organizer runtime token is not a proven read-only allowlisted token."
        )
    return username


def verify_deploy_identity(profile: object) -> str:
    """Require the personal-namespace owner with a documented write token."""

    username, role = _documented_access_token_role(profile)
    if username != SPACE_OWNER or role != "write":
        raise DeploymentError(
            "The organizer deploy token must be the owner write token."
        )
    return username


def require_separate_tokens(deploy_token: object, runtime_token: object) -> None:
    if (
        not isinstance(deploy_token, str)
        or not deploy_token
        or len(deploy_token) > 4096
        or not isinstance(runtime_token, str)
        or not runtime_token
        or len(runtime_token) > 4096
    ):
        raise DeploymentError("Both organizer deployment tokens are required.")
    if deploy_token == runtime_token:
        raise DeploymentError(
            "separate deploy-write and runtime-read tokens are required."
        )


def validate_request(request: DeploymentRequest) -> None:
    if not isinstance(request, DeploymentRequest):
        raise DeploymentError("Organizer deployment request is invalid.")
    _exact_revision(request.expected_source_revision, "source revision")
    _exact_revision(request.expected_private_revision, "private dataset revision")
    if request.visibility != "private":
        raise DeploymentError("Organizer Space visibility must be explicitly private.")
    _owner_only_allowlist(request.collaborators)
    has_parent = request.expected_space_parent is not None
    if has_parent == request.expect_absent:
        raise DeploymentError(
            "Specify exactly one expected Space parent or explicit absent state."
        )
    if has_parent:
        _exact_revision(request.expected_space_parent, "Space parent revision")
    if request.publish and request.confirmation != PUBLISH_CONFIRMATION:
        raise DeploymentError("The exact private-publication confirmation is required.")


def _require_existing_space(
    state: SpaceState,
    *,
    expected_revision: str | None,
    expected_private: bool,
    description: str,
) -> SpaceState:
    if not isinstance(state, SpaceState) or not state.exists:
        raise DeploymentError(f"The {description} is absent.")
    revision = _exact_revision(state.revision, f"{description} revision")
    if expected_revision is not None and revision != expected_revision:
        raise DeploymentError(f"The {description} revision moved.")
    if state.private is not expected_private:
        visibility = "private" if expected_private else "public"
        raise DeploymentError(f"The {description} must remain {visibility}.")
    if state.sdk != "gradio":
        raise DeploymentError(f"The {description} SDK must be Gradio.")
    if state.host is not None and (
        not isinstance(state.host, str) or not state.host.startswith("https://")
    ):
        raise DeploymentError(f"The {description} host is invalid.")
    return state


def _require_private_dataset(
    state: DatasetState,
    *,
    expected_revision: str,
) -> DatasetState:
    if not isinstance(state, DatasetState):
        raise DeploymentError("The private dataset state is unavailable.")
    revision = _exact_revision(state.revision, "private dataset revision")
    if revision != expected_revision:
        raise DeploymentError("The private dataset revision moved.")
    if state.private is not True:
        raise DeploymentError("The organizer ledger dataset must remain private.")
    return state


def _safe_json(response: HttpResponse, description: str) -> Mapping[str, object]:
    if (
        not isinstance(response, HttpResponse)
        or response.status_code != 200
        or not isinstance(response.body, bytes)
        or len(response.body) > _MAX_HTTP_BYTES
    ):
        raise DeploymentError(f"The {description} is unavailable.")
    try:
        value = json.loads(response.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeploymentError(f"The {description} is malformed.") from exc
    if not isinstance(value, Mapping):
        raise DeploymentError(f"The {description} is malformed.")
    return value


def _reconcile_private_dataset(
    backend: HubBackend,
    revision: str,
    runtime_token: str,
    snapshot_auditor: Callable[..., SnapshotAudit] | None,
) -> SnapshotAudit:
    try:
        paths = tuple(
            backend.list_dataset_files(
                PRIVATE_DATASET_REPO_ID,
                revision,
                runtime_token,
            )
        )
    except DeploymentError:
        raise
    except Exception as exc:
        raise DeploymentError("The private dataset inventory is unavailable.") from exc
    if "private/test_release.json" not in paths:
        return SnapshotAudit("disabled/no-release", 0, 0)
    auditor = snapshot_auditor or _default_snapshot_auditor
    try:
        result = auditor(
            PRIVATE_DATASET_REPO_ID,
            revision,
            runtime_token,
            api=backend,
        )
    except DeploymentError:
        raise
    except Exception as exc:
        raise DeploymentError(
            "The organizer snapshot failed exact-revision verification."
        ) from exc
    if (
        not isinstance(result, SnapshotAudit)
        or result.status != "verified"
        or type(result.account_count) is not int
        or type(result.attempt_count) is not int
        or result.account_count < 0
        or result.attempt_count < 0
    ):
        raise DeploymentError(
            "The organizer snapshot failed exact-revision verification."
        )
    return result


def _default_snapshot_auditor(
    repo_id: str,
    revision: str,
    token: str,
    *,
    api: object,
) -> SnapshotAudit:
    source_path = str(SOURCE_ROOT)
    added = source_path not in sys.path
    if added:
        sys.path.insert(0, source_path)
    try:
        module = importlib.import_module("organizer_data")
        module_path = Path(module.__file__).resolve().parent
        if module_path != SOURCE_ROOT.resolve():
            raise DeploymentError("The organizer snapshot verifier is unexpected.")
        snapshot = module.load_snapshot(repo_id, revision, token, api=api)
        report = module.verify_snapshot(snapshot)
        if report.valid is not True:
            raise DeploymentError(
                "The organizer snapshot failed exact-revision verification."
            )
        rows = module.organizer_rows(snapshot)
        if len(rows) != report.attempt_count:
            raise DeploymentError(
                "The organizer snapshot failed exact-revision verification."
            )
        return SnapshotAudit(
            status="verified",
            account_count=report.account_count,
            attempt_count=report.attempt_count,
        )
    finally:
        if added:
            sys.path.remove(source_path)


def _component_with_label(
    config: Mapping[str, object], label: str
) -> Mapping[str, object]:
    components = config.get("components")
    if not isinstance(components, list):
        raise DeploymentError("The public participant configuration is malformed.")
    matches = []
    for component in components:
        if not isinstance(component, Mapping):
            continue
        props = component.get("props")
        if isinstance(props, Mapping) and props.get("label") == label:
            matches.append(component)
    if len(matches) != 1:
        raise DeploymentError("The public participant configuration is malformed.")
    return matches[0]


def _verify_participant_disabled(
    backend: HubBackend,
    deploy_token: str,
) -> tuple[bool, bool]:
    try:
        state = _require_existing_space(
            backend.inspect_space(PARTICIPANT_SPACE_REPO_ID, deploy_token),
            expected_revision=None,
            expected_private=False,
            description="public participant Space",
        )
    except DeploymentError:
        raise
    except Exception as exc:
        raise DeploymentError("The public participant Space is unavailable.") from exc
    if not state.host:
        raise DeploymentError("The public participant Space host is unavailable.")
    config = _safe_json(
        backend.request("GET", state.host.rstrip("/") + "/config"),
        "public participant configuration",
    )
    split = _component_with_label(config, "Evaluation split")
    leaderboard = _component_with_label(config, "Leaderboard view")
    split_props = split.get("props")
    leaderboard_props = leaderboard.get("props")
    if (
        not isinstance(split_props, Mapping)
        or split_props.get("value") != "Validation (development)"
        or not isinstance(leaderboard_props, Mapping)
        or leaderboard_props.get("value") != "Validation leaderboard"
    ):
        raise DeploymentError("The public participant test surfaces are not disabled.")

    split_result = _safe_json(
        backend.request(
            "POST",
            state.host.rstrip("/") + "/api/select_split",
            json_body={"data": ["Test (final)"]},
        ),
        "public participant split response",
    )
    split_data = split_result.get("data")
    if (
        not isinstance(split_data, list)
        or len(split_data) != 4
        or not isinstance(split_data[2], Mapping)
        or split_data[2].get("interactive") is not False
        or not isinstance(split_data[0], Mapping)
        or "not open" not in str(split_data[0].get("value", "")).casefold()
    ):
        raise DeploymentError("The public participant test surfaces are not disabled.")

    leaderboard_id = leaderboard.get("id")
    dependencies = config.get("dependencies")
    candidates = []
    if isinstance(dependencies, list):
        for dependency in dependencies:
            if not isinstance(dependency, Mapping):
                continue
            if (
                dependency.get("inputs") == [leaderboard_id]
                and dependency.get("api_name") is False
                and [leaderboard_id, "change"] in dependency.get("targets", [])
            ):
                candidates.append(dependency)
    if len(candidates) != 1 or type(candidates[0].get("id")) is not int:
        raise DeploymentError("The public participant configuration is malformed.")
    final_result = _safe_json(
        backend.request(
            "POST",
            state.host.rstrip("/") + "/api/predict",
            json_body={
                "fn_index": candidates[0]["id"],
                "data": ["Final test leaderboard"],
            },
        ),
        "public participant final-leaderboard response",
    )
    final_data = final_result.get("data")
    if (
        not isinstance(final_data, list)
        or len(final_data) != 3
        or not isinstance(final_data[1], Mapping)
        or not isinstance(final_data[2], Mapping)
        or final_data[2].get("visible") is not False
        or "not available yet" not in str(final_data[1].get("value", "")).casefold()
        or "<table" in str(final_data[1].get("value", "")).casefold()
    ):
        raise DeploymentError("The public participant test surfaces are not disabled.")
    return True, True


def _require_secret_channel_clear(backend: HubBackend, deploy_token: str) -> None:
    try:
        variables = backend.get_space_variables(SPACE_REPO_ID, deploy_token)
    except DeploymentError:
        raise
    except Exception as exc:
        raise DeploymentError("Organizer Space variables are unavailable.") from exc
    if not isinstance(variables, Mapping) or any(
        name in variables for name in _RESERVED_SECRET_NAMES
    ):
        raise DeploymentError(
            "Organizer credentials must be secrets, never public variables."
        )


def _verify_exact_space_tree(
    backend: HubBackend,
    state: SpaceState,
    bundle: SourceBundle,
    deploy_token: str,
) -> None:
    revision = _exact_revision(state.revision, "organizer Space revision")
    try:
        paths = tuple(
            sorted(backend.list_space_files(SPACE_REPO_ID, revision, deploy_token))
        )
        expected_paths = tuple(sorted(BUNDLE_PATHS))
        if paths != expected_paths:
            raise DeploymentError("The deployed organizer Space tree is not exact.")
        files = backend.read_space_files(
            SPACE_REPO_ID,
            revision,
            expected_paths,
            deploy_token,
        )
    except DeploymentError:
        raise
    except Exception as exc:
        raise DeploymentError(
            "The deployed organizer Space tree is unavailable."
        ) from exc
    if set(files) != set(BUNDLE_PATHS) or any(
        files.get(path) != bundle.files[path].payload for path in BUNDLE_PATHS
    ):
        raise DeploymentError("The deployed organizer Space bytes do not reconcile.")


def _verify_organizer_runtime(
    backend: HubBackend,
    state: SpaceState,
    runtime_token: str,
    deploy_token: str,
) -> str:
    if not state.host:
        return "pending"
    host = state.host.rstrip("/")
    for path in ("/", "/config", "/info"):
        denied = backend.request("GET", host + path)
        if denied.status_code not in {401, 403, 404}:
            raise DeploymentError("Unauthenticated organizer access was not denied.")
    if state.runtime_stage != _READY_STAGE:
        return "pending"

    root = backend.request("GET", host + "/", token=runtime_token)
    if root.status_code != 200 or len(root.body) > _MAX_HTTP_BYTES:
        raise DeploymentError("Authenticated organizer access is unavailable.")
    config_response = backend.request("GET", host + "/config", token=runtime_token)
    info_response = backend.request("GET", host + "/info", token=runtime_token)
    config = _safe_json(config_response, "organizer Space configuration")
    info = _safe_json(info_response, "organizer Space API information")
    config_bytes = config_response.body
    forbidden_values = (
        runtime_token.encode("utf-8"),
        deploy_token.encode("utf-8"),
        PRIVATE_DATASET_REPO_ID.encode("utf-8"),
        b"ORGANIZER_READ_TOKEN",
        b"PRIVATE_REPO_ID",
        b"HF_WRITE_TOKEN",
    )
    if any(value and value in config_bytes for value in forbidden_values):
        raise DeploymentError("The organizer Space configuration exposes a secret.")
    dependencies = config.get("dependencies")
    if not isinstance(dependencies, list) or any(
        not isinstance(item, Mapping) or item.get("api_name") not in {False, None}
        for item in dependencies
    ):
        raise DeploymentError("The organizer Space exposes a named mutation endpoint.")
    named = info.get("named_endpoints")
    if not isinstance(named, Mapping) or named:
        raise DeploymentError("The organizer Space exposes a named mutation endpoint.")
    components = config.get("components")
    if not isinstance(components, list):
        raise DeploymentError("The organizer Space configuration is malformed.")
    for component in components:
        if not isinstance(component, Mapping) or component.get("type") != "button":
            continue
        props = component.get("props")
        value = props.get("value", "") if isinstance(props, Mapping) else ""
        if _MUTATION_CONTROL.search(str(value)):
            raise DeploymentError("The organizer Space exposes a mutation control.")
    return "verified"


def _inspect_expected_space(
    backend: HubBackend,
    request: DeploymentRequest,
    deploy_token: str,
) -> tuple[str, SpaceState]:
    try:
        state = backend.inspect_space(SPACE_REPO_ID, deploy_token)
    except DeploymentError:
        raise
    except Exception as exc:
        raise DeploymentError("The organizer Space state is unavailable.") from exc
    if request.expect_absent:
        if not isinstance(state, SpaceState) or state.exists:
            raise DeploymentError("The organizer Space was expected to be absent.")
        return "create", state
    return (
        "update",
        _require_existing_space(
            state,
            expected_revision=request.expected_space_parent,
            expected_private=True,
            description="organizer Space",
        ),
    )


def run_deployment(
    request: DeploymentRequest,
    *,
    deploy_token: str,
    runtime_token: str,
    source_backend: SourceBackend | None = None,
    hub_backend: HubBackend | None = None,
    snapshot_auditor: Callable[..., SnapshotAudit] | None = None,
) -> DeploymentResult:
    """Plan or publish one exact private organizer Space deployment."""

    validate_request(request)
    require_separate_tokens(deploy_token, runtime_token)
    source = source_backend or GitSourceBackend()
    hub = hub_backend or HuggingFaceBackend()
    bundle = capture_source_bundle(source, request.expected_source_revision)
    try:
        verify_deploy_identity(hub.whoami(deploy_token))
        verify_runtime_identity(
            hub.whoami(runtime_token),
            collaborators=request.collaborators,
        )
    except DeploymentError:
        raise
    except Exception as exc:
        raise DeploymentError("Hugging Face token verification failed.") from exc

    private_state = _require_private_dataset(
        hub.inspect_dataset(
            PRIVATE_DATASET_REPO_ID,
            request.expected_private_revision,
            runtime_token,
        ),
        expected_revision=request.expected_private_revision,
    )
    action, space_state = _inspect_expected_space(hub, request, deploy_token)
    if space_state.exists:
        _require_secret_channel_clear(hub, deploy_token)
    snapshot = _reconcile_private_dataset(
        hub,
        private_state.revision,
        runtime_token,
        snapshot_auditor,
    )
    participant_submissions_disabled, participant_final_disabled = (
        _verify_participant_disabled(hub, deploy_token)
    )

    if not request.publish:
        return DeploymentResult(
            published=False,
            action=action,
            source_revision=bundle.source_revision,
            bundle_tree_sha256=bundle.tree_sha256,
            space_revision=space_state.revision,
            private_dataset_revision=private_state.revision,
            organizer_reconciliation=snapshot.status,
            organizer_account_count=snapshot.account_count,
            organizer_attempt_count=snapshot.attempt_count,
            participant_test_submissions_disabled=participant_submissions_disabled,
            participant_final_leaderboard_disabled=participant_final_disabled,
            runtime_access="not-probed",
        )

    if action == "create":
        try:
            space_state = hub.create_private_space(SPACE_REPO_ID, deploy_token)
        except Exception as exc:
            raise DeploymentError("Private organizer Space creation failed.") from exc
        space_state = _require_existing_space(
            space_state,
            expected_revision=None,
            expected_private=True,
            description="organizer Space",
        )
        _require_secret_channel_clear(hub, deploy_token)
    else:
        _, space_state = _inspect_expected_space(hub, request, deploy_token)

    # Rebind source bytes immediately before the only repository mutation.
    fresh_bundle = capture_source_bundle(source, request.expected_source_revision)
    if fresh_bundle != bundle:
        raise DeploymentError("The organizer source revision changed before publish.")
    parent = _exact_revision(space_state.revision, "organizer Space parent revision")
    try:
        existing_paths = tuple(
            sorted(hub.list_space_files(SPACE_REPO_ID, parent, deploy_token))
        )
    except Exception as exc:
        raise DeploymentError("The organizer Space tree is unavailable.") from exc
    deletions = tuple(path for path in existing_paths if path not in BUNDLE_PATHS)
    additions = {path: bundle.files[path].payload for path in BUNDLE_PATHS}
    try:
        new_revision = hub.commit_space(
            SPACE_REPO_ID,
            parent,
            additions,
            deletions,
            deploy_token,
        )
    except DeploymentError:
        raise
    except Exception as exc:
        raise DeploymentError(
            "The exact-parent organizer Space commit failed."
        ) from exc
    _exact_revision(new_revision, "published organizer Space revision")

    try:
        hub.set_space_secret(
            SPACE_REPO_ID,
            "ORGANIZER_READ_TOKEN",
            runtime_token,
            deploy_token,
        )
        hub.set_space_secret(
            SPACE_REPO_ID,
            "PRIVATE_REPO_ID",
            PRIVATE_DATASET_REPO_ID,
            deploy_token,
        )
    except Exception as exc:
        raise DeploymentError("Organizer Space secret configuration failed.") from exc
    _require_secret_channel_clear(hub, deploy_token)

    final_state = _require_existing_space(
        hub.inspect_space(SPACE_REPO_ID, deploy_token),
        expected_revision=new_revision,
        expected_private=True,
        description="organizer Space",
    )
    _verify_exact_space_tree(hub, final_state, bundle, deploy_token)
    runtime_access = _verify_organizer_runtime(
        hub,
        final_state,
        runtime_token,
        deploy_token,
    )

    final_private_state = _require_private_dataset(
        hub.inspect_dataset(
            PRIVATE_DATASET_REPO_ID,
            request.expected_private_revision,
            runtime_token,
        ),
        expected_revision=request.expected_private_revision,
    )
    final_snapshot = _reconcile_private_dataset(
        hub,
        final_private_state.revision,
        runtime_token,
        snapshot_auditor,
    )
    if final_snapshot != snapshot:
        raise DeploymentError(
            "The private organizer snapshot changed during deployment."
        )
    participant_submissions_disabled, participant_final_disabled = (
        _verify_participant_disabled(hub, deploy_token)
    )
    return DeploymentResult(
        published=True,
        action=action,
        source_revision=bundle.source_revision,
        bundle_tree_sha256=bundle.tree_sha256,
        space_revision=final_state.revision,
        private_dataset_revision=final_private_state.revision,
        organizer_reconciliation=final_snapshot.status,
        organizer_account_count=final_snapshot.account_count,
        organizer_attempt_count=final_snapshot.attempt_count,
        participant_test_submissions_disabled=participant_submissions_disabled,
        participant_final_leaderboard_disabled=participant_final_disabled,
        runtime_access=runtime_access,
    )


def render_result(result: DeploymentResult) -> str:
    if not isinstance(result, DeploymentResult):
        raise DeploymentError("Organizer deployment result is invalid.")
    return json.dumps(
        {
            "action": result.action,
            "bundle_tree_sha256": result.bundle_tree_sha256,
            "organizer_accounts": result.organizer_account_count,
            "organizer_attempts": result.organizer_attempt_count,
            "organizer_reconciliation": result.organizer_reconciliation,
            "participant_final_leaderboard_disabled": result.participant_final_leaderboard_disabled,
            "participant_test_submissions_disabled": result.participant_test_submissions_disabled,
            "private_dataset_revision": result.private_dataset_revision,
            "published": result.published,
            "runtime_access": result.runtime_access,
            "source_revision": result.source_revision,
            "space_revision": result.space_revision,
            "target": SPACE_REPO_ID,
        },
        sort_keys=True,
    )


class GitSourceBackend:
    """Read a source tree without allowing Git state to drift into the bundle."""

    def inspect_source(self, source_root: Path) -> SourceState:
        root = Path(source_root).resolve()
        repository = _git(root, "rev-parse", "--show-toplevel").strip()
        expected_repo = str(REPOSITORY_ROOT.resolve())
        if repository != expected_repo:
            raise DeploymentError("The organizer source repository is unexpected.")
        revision = _git(root, "rev-parse", "HEAD").strip()
        _exact_revision(revision, "source revision")
        status = _git(
            root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
        return SourceState(root=root, revision=revision, clean=status == "")

    def read_revision_file(
        self, source_root: Path, revision: str, relative_path: str
    ) -> bytes:
        full_path = f"competition/hf-organizer-space/{relative_path}"
        result = subprocess.run(
            ["git", "show", f"{revision}:{full_path}"],
            cwd=source_root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            raise DeploymentError(
                "An organizer bundle file is not tracked at the pinned revision."
            )
        return result.stdout

    def read_worktree_file(self, source_root: Path, relative_path: str) -> bytes:
        path = Path(source_root) / relative_path
        try:
            if path.is_symlink() or not path.is_file():
                raise DeploymentError("An organizer bundle path is unsafe.")
            return path.read_bytes()
        except DeploymentError:
            raise
        except OSError as exc:
            raise DeploymentError("An organizer bundle file is unreadable.") from exc


class HuggingFaceBackend:
    """Documented Hugging Face API adapter with no implicit credential fallback."""

    def __init__(self, *, api_factory=None, download=None, session=None) -> None:
        if api_factory is None or download is None:
            try:
                from huggingface_hub import HfApi, hf_hub_download
            except ImportError as exc:
                raise DeploymentError(
                    "The pinned Hugging Face client is unavailable."
                ) from exc
            api_factory = api_factory or (lambda token: HfApi(token=token))
            download = download or hf_hub_download
        if session is None:
            try:
                import requests
            except ImportError as exc:
                raise DeploymentError("The HTTP client is unavailable.") from exc
            session = requests.Session()
        self._api_factory = api_factory
        self._download = download
        self._session = session

    def _api(self, token: str):
        return self._api_factory(token)

    def whoami(self, token: str) -> Mapping[str, object]:
        try:
            return self._api(token).whoami(token=token)
        except Exception as exc:
            raise DeploymentError("Hugging Face token verification failed.") from exc

    def inspect_space(self, repo_id: str, token: str) -> SpaceState:
        try:
            from huggingface_hub.utils import RepositoryNotFoundError

            info = self._api(token).space_info(repo_id, token=token)
        except RepositoryNotFoundError:
            return SpaceState.missing()
        except Exception as exc:
            raise DeploymentError("The organizer Space state is unavailable.") from exc
        runtime = getattr(info, "runtime", None)
        host = getattr(info, "host", None)
        if host is None:
            subdomain = getattr(info, "subdomain", None)
            if isinstance(subdomain, str) and subdomain:
                host = f"https://{subdomain}.hf.space"
        return SpaceState(
            exists=True,
            revision=getattr(info, "sha", None),
            private=getattr(info, "private", None),
            sdk=getattr(info, "sdk", None),
            host=host,
            runtime_stage=getattr(runtime, "stage", None),
        )

    def inspect_dataset(self, repo_id: str, revision: str, token: str) -> DatasetState:
        try:
            info = self._api(token).repo_info(
                repo_id,
                repo_type="dataset",
                revision="main",
                token=token,
            )
        except Exception as exc:
            raise DeploymentError("The private dataset state is unavailable.") from exc
        return DatasetState(
            revision=getattr(info, "sha", None),
            private=getattr(info, "private", None),
        )

    def list_dataset_files(
        self, repo_id: str, revision: str, token: str
    ) -> Sequence[str]:
        try:
            return tuple(
                self._api(token).list_repo_files(
                    repo_id,
                    repo_type="dataset",
                    revision=revision,
                    token=token,
                )
            )
        except Exception as exc:
            raise DeploymentError(
                "The private dataset inventory is unavailable."
            ) from exc

    def get_space_variables(self, repo_id: str, token: str) -> Mapping[str, object]:
        try:
            return self._api(token).get_space_variables(repo_id, token=token)
        except Exception as exc:
            raise DeploymentError("Organizer Space variables are unavailable.") from exc

    def create_private_space(self, repo_id: str, token: str) -> SpaceState:
        try:
            self._api(token).create_repo(
                repo_id=repo_id,
                repo_type="space",
                private=True,
                space_sdk="gradio",
                exist_ok=False,
                token=token,
            )
        except Exception as exc:
            raise DeploymentError("Private organizer Space creation failed.") from exc
        return self.inspect_space(repo_id, token)

    def list_space_files(
        self, repo_id: str, revision: str, token: str
    ) -> Sequence[str]:
        try:
            return tuple(
                self._api(token).list_repo_files(
                    repo_id,
                    repo_type="space",
                    revision=revision,
                    token=token,
                )
            )
        except Exception as exc:
            raise DeploymentError("The organizer Space tree is unavailable.") from exc

    def read_space_files(
        self,
        repo_id: str,
        revision: str,
        paths: Sequence[str],
        token: str,
    ) -> Mapping[str, bytes]:
        if tuple(sorted(paths)) != tuple(paths) or any(
            path not in BUNDLE_PATHS for path in paths
        ):
            raise DeploymentError("The organizer Space read inventory is unsafe.")
        result: dict[str, bytes] = {}
        try:
            with tempfile.TemporaryDirectory(prefix="docsem-organizer-space-") as name:
                directory = Path(name)
                directory.chmod(0o700)
                for path in paths:
                    downloaded = Path(
                        self._download(
                            repo_id=repo_id,
                            repo_type="space",
                            filename=path,
                            revision=revision,
                            token=token,
                            local_dir=name,
                        )
                    )
                    resolved = downloaded.resolve(strict=True)
                    if (
                        not resolved.is_file()
                        or resolved.stat().st_size > _MAX_BUNDLE_FILE_BYTES
                    ):
                        raise DeploymentError(
                            "A deployed organizer Space file is invalid."
                        )
                    result[path] = resolved.read_bytes()
        except DeploymentError:
            raise
        except Exception as exc:
            raise DeploymentError(
                "The deployed organizer Space bytes are unavailable."
            ) from exc
        return result

    def commit_space(
        self,
        repo_id: str,
        expected_parent: str,
        additions: Mapping[str, bytes],
        deletions: Sequence[str],
        token: str,
    ) -> str:
        if tuple(additions) != BUNDLE_PATHS or len(set(deletions)) != len(deletions):
            raise DeploymentError("The organizer Space commit inventory is invalid.")
        try:
            from huggingface_hub import CommitOperationAdd, CommitOperationDelete

            state = _require_existing_space(
                self.inspect_space(repo_id, token),
                expected_revision=expected_parent,
                expected_private=True,
                description="organizer Space",
            )
            if state.revision != expected_parent:
                raise DeploymentError("The organizer Space revision moved.")
            operations = [
                CommitOperationAdd(path_in_repo=path, path_or_fileobj=additions[path])
                for path in BUNDLE_PATHS
            ]
            operations.extend(
                CommitOperationDelete(path_in_repo=path) for path in sorted(deletions)
            )
            info = self._api(token).create_commit(
                repo_id=repo_id,
                repo_type="space",
                revision="main",
                parent_commit=expected_parent,
                create_pr=False,
                operations=operations,
                commit_message="Deploy private DocSem organizer dashboard",
                token=token,
            )
        except DeploymentError:
            raise
        except Exception as exc:
            raise DeploymentError(
                "The exact-parent organizer Space commit failed."
            ) from exc
        return _exact_revision(getattr(info, "oid", None), "organizer Space commit")

    def set_space_secret(self, repo_id: str, name: str, value: str, token: str) -> None:
        if (
            name not in _RESERVED_SECRET_NAMES
            or not isinstance(value, str)
            or not value
        ):
            raise DeploymentError("The organizer Space secret request is invalid.")
        try:
            self._api(token).add_space_secret(
                repo_id,
                name,
                value,
                token=token,
            )
        except Exception as exc:
            raise DeploymentError(
                "Organizer Space secret configuration failed."
            ) from exc

    def request(
        self,
        method: str,
        url: str,
        *,
        token: str | None = None,
        json_body: object | None = None,
    ) -> HttpResponse:
        if method not in {"GET", "POST"} or not url.startswith("https://"):
            raise DeploymentError("The Space verification request is invalid.")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        response = None
        try:
            response = self._session.request(
                method,
                url,
                headers=headers,
                json=json_body,
                timeout=20,
                allow_redirects=False,
                stream=True,
            )
            chunks = []
            size = 0
            iterator = (
                response.iter_content(chunk_size=64 * 1024)
                if hasattr(response, "iter_content")
                else (response.content,)
            )
            for chunk in iterator:
                if not isinstance(chunk, bytes):
                    raise DeploymentError("The Space verification response is invalid.")
                size += len(chunk)
                if size > _MAX_HTTP_BYTES:
                    raise DeploymentError(
                        "The Space verification response is too large."
                    )
                chunks.append(chunk)
            safe_headers = {
                key: value
                for key, value in getattr(response, "headers", {}).items()
                if key.casefold() in {"content-type", "location"}
            }
            return HttpResponse(
                status_code=int(response.status_code),
                body=b"".join(chunks),
                headers=safe_headers,
            )
        except DeploymentError:
            raise
        except Exception as exc:
            raise DeploymentError("The Space verification request failed.") from exc
        finally:
            if response is not None and hasattr(response, "close"):
                response.close()

    # The exact-snapshot verifier consumes an HfApi-shaped object.  These
    # wrappers preserve its pinned revision and explicit token arguments.
    def repo_info(self, *args, token: str, **kwargs):
        return self._api(token).repo_info(*args, token=token, **kwargs)

    def list_repo_commits(self, *args, token: str, **kwargs):
        return self._api(token).list_repo_commits(*args, token=token, **kwargs)

    def list_repo_files(self, *args, token: str, **kwargs):
        return self._api(token).list_repo_files(*args, token=token, **kwargs)

    def hf_hub_download(self, *args, token: str, **kwargs):
        return self._download(*args, token=token, **kwargs)


def _git(root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DeploymentError("The organizer source Git state is unavailable.") from exc
    return result.stdout


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--expected-private-revision", required=True)
    state = parser.add_mutually_exclusive_group(required=True)
    state.add_argument("--expected-space-parent")
    state.add_argument("--expect-absent", action="store_true")
    parser.add_argument("--visibility", choices=("private", "public"), required=True)
    parser.add_argument(
        "--collaborator", action="append", dest="collaborators", required=True
    )
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--confirm")
    return parser


def _request_from_argv(argv: Sequence[str] | None = None) -> DeploymentRequest:
    args = _parser().parse_args(argv)
    return DeploymentRequest(
        expected_source_revision=args.expected_source_revision,
        expected_private_revision=args.expected_private_revision,
        expected_space_parent=args.expected_space_parent,
        expect_absent=args.expect_absent,
        visibility=args.visibility,
        collaborators=tuple(args.collaborators),
        publish=args.publish,
        confirmation=args.confirm,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    environment: Mapping[str, object] | None = None,
    stdout=None,
    stderr=None,
) -> int:
    environment = os.environ if environment is None else environment
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr
    request = _request_from_argv(argv)
    try:
        validate_request(request)
        deploy_token = environment.get("DOCSEM_ORGANIZER_DEPLOY_TOKEN")
        runtime_token = environment.get("DOCSEM_ORGANIZER_READ_TOKEN")
        require_separate_tokens(deploy_token, runtime_token)
        result = run_deployment(
            request,
            deploy_token=deploy_token,
            runtime_token=runtime_token,
        )
    except DeploymentError as exc:
        print(f"ERROR: {exc}", file=stderr)
        return 2
    print(render_result(result), file=stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
