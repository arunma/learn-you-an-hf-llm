#!/usr/bin/env bash
#
# pull_artifacts.sh — Pull a finished training run's artifacts from a RunPod
#                     (or other rented-GPU) pod back to the local repo.
#
# Wipes the local checkpoints/ and tb_logs/ directories first (after
# interactive confirmation) so the new run replaces them cleanly. Pulls:
#
#   - checkpoints/final/  (model weights — required for inference)
#   - tb_logs/            (TensorBoard event files — for signal-1 analysis)
#   - train.log           (text record of the run)
#
# Usage:
#   ./runs/h100_tinystories/pull_artifacts.sh <pod_ip> <pod_port> [ssh_key]
#
# Example:
#   ./runs/h100_tinystories/pull_artifacts.sh 38.80.152.148 30948
#   ./runs/h100_tinystories/pull_artifacts.sh 38.80.152.148 30948 ~/.ssh/id_ed25519
#
# Defaults:
#   ssh_key       = ~/.ssh/id_ed25519_arunma
#   remote_repo   = /workspace/learn-you-an-hf-llm
#                   (override via REPO_REMOTE env var if you cloned elsewhere,
#                    e.g. REPO_REMOTE=/learn-you-an-hf-llm ./pull_artifacts.sh ...)
#
# Run from the repo root (where pyproject.toml lives).

set -euo pipefail

# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------
if [[ $# -lt 2 || $# -gt 3 ]]; then
    cat >&2 <<'USAGE'
Usage: pull_artifacts.sh <pod_ip> <pod_port> [ssh_key]
       ssh_key default: ~/.ssh/id_ed25519_arunma
       Override REPO_REMOTE env var if pod's clone isn't at /workspace/learn-you-an-hf-llm
USAGE
    exit 1
fi

POD_IP="$1"
POD_PORT="$2"
KEY="${3:-$HOME/.ssh/id_ed25519_arunma}"
REPO_REMOTE="${REPO_REMOTE:-/workspace/learn-you-an-hf-llm}"

# -----------------------------------------------------------------------------
# Verify we're at the repo root
# -----------------------------------------------------------------------------
if [[ ! -f "pyproject.toml" || ! -d "runs/h100_tinystories" ]]; then
    echo "Error: run this script from the repo root (where pyproject.toml lives)." >&2
    echo "       Current dir: $(pwd)" >&2
    exit 1
fi

if [[ ! -f "$KEY" ]]; then
    echo "Error: SSH key not found at $KEY" >&2
    exit 1
fi

LOCAL_RUN_DIR="runs/h100_tinystories"
REMOTE_RUN_DIR="$REPO_REMOTE/runs/h100_tinystories"
POD="root@$POD_IP"

echo "=== pull_artifacts.sh ==="
echo "Pod:           $POD"
echo "Port:          $POD_PORT"
echo "Key:           $KEY"
echo "Remote root:   $REPO_REMOTE"
echo "Local target:  $LOCAL_RUN_DIR"
echo ""

# -----------------------------------------------------------------------------
# Confirm before destructive cleanup
# -----------------------------------------------------------------------------
EXISTING=()
[[ -d "$LOCAL_RUN_DIR/checkpoints" ]] && EXISTING+=("$LOCAL_RUN_DIR/checkpoints")
[[ -d "$LOCAL_RUN_DIR/tb_logs"     ]] && EXISTING+=("$LOCAL_RUN_DIR/tb_logs")
[[ -f "$LOCAL_RUN_DIR/train.log"   ]] && EXISTING+=("$LOCAL_RUN_DIR/train.log")

if (( ${#EXISTING[@]} > 0 )); then
    echo "About to DELETE local artifacts (will be replaced by the pull):"
    for path in "${EXISTING[@]}"; do
        if [[ -d "$path" ]]; then
            du -sh "$path" 2>/dev/null || echo "  $path"
        else
            ls -lh "$path" | awk '{print "  "$NF" ("$5")"}'
        fi
    done
    echo ""
    read -r -p "Proceed? [y/N] " ans
    if [[ ! "$ans" =~ ^[Yy]$ ]]; then
        echo "Aborted." >&2
        exit 1
    fi
    rm -rf "$LOCAL_RUN_DIR/checkpoints" "$LOCAL_RUN_DIR/tb_logs"
    rm -f  "$LOCAL_RUN_DIR/train.log"
    echo "Wiped."
    echo ""
fi

mkdir -p "$LOCAL_RUN_DIR/checkpoints"

# -----------------------------------------------------------------------------
# Pull artifacts
# -----------------------------------------------------------------------------
SCP_OPTS=(-P "$POD_PORT" -i "$KEY" -o ConnectTimeout=10)

echo "=== 1/3 Pulling checkpoints/final/ (~117 MB for d8, ~470 MB for d12) ==="
scp "${SCP_OPTS[@]}" -r \
    "$POD:$REMOTE_RUN_DIR/checkpoints/final" \
    "$LOCAL_RUN_DIR/checkpoints/final"
echo ""

echo "=== 2/3 Pulling tb_logs/ ==="
scp "${SCP_OPTS[@]}" -r \
    "$POD:$REMOTE_RUN_DIR/tb_logs" \
    "$LOCAL_RUN_DIR/tb_logs"
echo ""

echo "=== 3/3 Pulling train.log ==="
scp "${SCP_OPTS[@]}" \
    "$POD:$REMOTE_RUN_DIR/train.log" \
    "$LOCAL_RUN_DIR/train.log"
echo ""

# -----------------------------------------------------------------------------
# Verify
# -----------------------------------------------------------------------------
echo "=== Verifying ==="
echo "checkpoints/final/:"
ls -lh "$LOCAL_RUN_DIR/checkpoints/final/" | tail -n +2
echo ""
echo "tb_logs/:"
ls "$LOCAL_RUN_DIR/tb_logs/" | sed 's/^/  /'
echo ""
echo "train.log:"
ls -lh "$LOCAL_RUN_DIR/train.log" | awk '{print "  size: "$5"  modified: "$6" "$7" "$8}'
echo ""

# Quick sanity: final loss from the log
echo "Final line of train.log:"
tail -n 3 "$LOCAL_RUN_DIR/train.log" | sed 's/^/  /'
echo ""

# -----------------------------------------------------------------------------
# Reminders
# -----------------------------------------------------------------------------
cat <<'EOF'
=== Done ===

Pod cleanup (do this now, both steps):
  1. RunPod dashboard → Pods → ⋯ → Terminate
  2. RunPod dashboard → Storage → delete the auto-created network volume

Next:
  python runs/h100_tinystories/infer.py
EOF
