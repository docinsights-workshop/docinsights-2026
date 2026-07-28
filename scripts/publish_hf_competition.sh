#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
HF_CLI="${HF_CLI:-/Users/aamita/miniconda3/bin/huggingface-cli}"

PUBLIC_DATASET_NAME="${PUBLIC_DATASET_NAME:-docinsights-2026-shared-task-data}"
SUBMISSIONS_NAME="${SUBMISSIONS_NAME:-docinsights-2026-shared-task-submissions}"
SPACE_NAME="${SPACE_NAME:-docsem-docinsights}"

CURRENT_USER="$("$HF_CLI" whoami | sed -n '1p')"
if [[ "$CURRENT_USER" == "Not logged in" || -z "$CURRENT_USER" ]]; then
  echo "Not logged in to Hugging Face. Run: $HF_CLI login" >&2
  exit 1
fi

OWNER="${1:-$CURRENT_USER}"

create_repo() {
  local name="$1"
  local type="$2"
  shift 2
  local args=("$name" --type "$type" -y "$@")
  local output
  local status
  if [[ "$OWNER" != "$CURRENT_USER" ]]; then
    args+=(--organization "$OWNER")
  fi
  set +e
  output="$("$HF_CLI" repo create "${args[@]}" 2>&1)"
  status=$?
  set -e
  printf '%s\n' "$output"
  if [[ $status -ne 0 ]]; then
    if grep -qi "already exists" <<<"$output"; then
      echo "Repo $OWNER/$name already exists; continuing." >&2
      return 0
    fi
    echo "Could not create $type repo $OWNER/$name." >&2
    return "$status"
  fi
}

create_repo "$PUBLIC_DATASET_NAME" dataset
create_repo "$SPACE_NAME" space --space_sdk gradio

"$HF_CLI" upload "$OWNER/$PUBLIC_DATASET_NAME" "$ROOT/competition/hf-dataset" . \
  --repo-type dataset \
  --commit-message "Add DocInsights shared task public data"

"$HF_CLI" upload "$OWNER/$SUBMISSIONS_NAME" "$ROOT/competition/hf-submissions" . \
  --repo-type dataset \
  --private \
  --commit-message "Add DocInsights shared task private submission store"

"$HF_CLI" upload "$OWNER/$SPACE_NAME" "$ROOT/competition/hf-space" . \
  --repo-type space \
  --commit-message "Add DocInsights submission portal"

cat <<EOF

Published Hugging Face assets:
- Public data: https://huggingface.co/datasets/$OWNER/$PUBLIC_DATASET_NAME
- Private submissions: https://huggingface.co/datasets/$OWNER/$SUBMISSIONS_NAME
- Space: https://huggingface.co/spaces/$OWNER/$SPACE_NAME

Set these Space variables/secrets:
PUBLIC_DATASET_REPO=$OWNER/$PUBLIC_DATASET_NAME
GOLD_REPO_ID=$OWNER/$SUBMISSIONS_NAME
GOLD_FILE=private/val_labels.jsonl
SUBMISSIONS_REPO_ID=$OWNER/$SUBMISSIONS_NAME
HF_WRITE_TOKEN=<write token with access to $OWNER/$SUBMISSIONS_NAME>
EOF
