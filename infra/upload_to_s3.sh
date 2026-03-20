#!/bin/bash
# Upload training data to S3 / download trained model from S3.
#
# Usage:
#   ./infra/upload_to_s3.sh upload        # Upload training data to S3
#   ./infra/upload_to_s3.sh download      # Download training data from S3 to local
#   ./infra/upload_to_s3.sh upload-model  # Upload trained model to S3
#   ./infra/upload_to_s3.sh download-model # Download trained model from S3
#
# Prerequisites:
#   - AWS CLI configured (aws configure) with appropriate IAM permissions
#   - S3 bucket exists: s3://omnis-sft

set -euo pipefail

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
S3_BUCKET="s3://omnis-sft"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Data files to sync
DATA_FILES=(
    "data/generated/training_data.jsonl"
    "data/tool_catalog.json"
    "data/environment_context.json"
    "data/seed_chains.json"
)

MODEL_DIR="training/output"

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
log()  { echo -e "\n\033[1;34m>>> $*\033[0m"; }
ok()   { echo -e "    \033[1;32m[OK]\033[0m $*"; }
fail() { echo -e "    \033[1;31m[FAIL]\033[0m $*"; exit 1; }
warn() { echo -e "    \033[1;33m[WARN]\033[0m $*"; }

print_size() {
    local path="$1"
    if [ -f "$path" ]; then
        local size
        size=$(ls -lh "$path" | awk '{print $5}')
        echo "    $path ($size)"
    elif [ -d "$path" ]; then
        local size
        size=$(du -sh "$path" 2>/dev/null | cut -f1)
        echo "    $path/ ($size)"
    fi
}

check_aws_cli() {
    if ! command -v aws &>/dev/null; then
        fail "AWS CLI not found. Install with: pip install awscli"
    fi
    # Quick credential check
    if ! aws sts get-caller-identity &>/dev/null; then
        fail "AWS credentials not configured. Run: aws configure"
    fi
    ok "AWS CLI configured ($(aws sts get-caller-identity --query 'Account' --output text))"
}

# --------------------------------------------------------------------------- #
# upload: Push training data files to S3
# --------------------------------------------------------------------------- #
do_upload() {
    log "Uploading training data to S3"
    check_aws_cli

    local uploaded=0
    local skipped=0

    for file in "${DATA_FILES[@]}"; do
        local local_path="$PROJECT_ROOT/$file"
        local s3_path="$S3_BUCKET/$file"

        if [ -f "$local_path" ]; then
            print_size "$local_path"
            aws s3 cp "$local_path" "$s3_path"
            ok "Uploaded $file"
            ((uploaded++))
        else
            warn "File not found: $local_path (skipping)"
            ((skipped++))
        fi
    done

    echo ""
    log "Upload Summary"
    echo "  Uploaded: $uploaded files"
    echo "  Skipped:  $skipped files"
    echo "  Destination: $S3_BUCKET/"

    if [ "$uploaded" -eq 0 ]; then
        fail "No files were uploaded. Have you generated training data yet?"
    fi
}

# --------------------------------------------------------------------------- #
# download: Pull training data from S3 to local
# --------------------------------------------------------------------------- #
do_download() {
    log "Downloading training data from S3"
    check_aws_cli

    for file in "${DATA_FILES[@]}"; do
        local local_path="$PROJECT_ROOT/$file"
        local s3_path="$S3_BUCKET/$file"

        # Ensure parent directory exists
        mkdir -p "$(dirname "$local_path")"

        echo "  Fetching $file ..."
        if aws s3 cp "$s3_path" "$local_path" 2>/dev/null; then
            print_size "$local_path"
            ok "Downloaded $file"
        else
            warn "Not found on S3: $s3_path (skipping)"
        fi
    done

    echo ""
    log "Download complete"
    echo "  Local data directory: $PROJECT_ROOT/data/"
}

# --------------------------------------------------------------------------- #
# upload-model: Push trained model to S3
# --------------------------------------------------------------------------- #
do_upload_model() {
    log "Uploading trained model to S3"
    check_aws_cli

    local local_model_dir="$PROJECT_ROOT/$MODEL_DIR"

    if [ ! -d "$local_model_dir" ]; then
        fail "Model directory not found: $local_model_dir — has training completed?"
    fi

    print_size "$local_model_dir"
    echo "  Syncing to $S3_BUCKET/$MODEL_DIR/ ..."
    aws s3 sync "$local_model_dir" "$S3_BUCKET/$MODEL_DIR/" \
        --exclude "*.log" \
        --exclude "__pycache__/*" \
        --exclude "*.pyc"

    ok "Model uploaded to $S3_BUCKET/$MODEL_DIR/"

    echo ""
    log "Upload-model summary"
    echo "  Source:      $local_model_dir"
    echo "  Destination: $S3_BUCKET/$MODEL_DIR/"
    echo "  S3 contents:"
    aws s3 ls "$S3_BUCKET/$MODEL_DIR/" --summarize --human-readable 2>/dev/null | tail -5
}

# --------------------------------------------------------------------------- #
# download-model: Pull trained model from S3
# --------------------------------------------------------------------------- #
do_download_model() {
    log "Downloading trained model from S3"
    check_aws_cli

    local local_model_dir="$PROJECT_ROOT/$MODEL_DIR"
    mkdir -p "$local_model_dir"

    echo "  Syncing from $S3_BUCKET/$MODEL_DIR/ ..."
    aws s3 sync "$S3_BUCKET/$MODEL_DIR/" "$local_model_dir/" \
        --exclude "*.log" \
        --exclude "__pycache__/*" \
        --exclude "*.pyc"

    ok "Model downloaded to $local_model_dir"
    print_size "$local_model_dir"

    echo ""
    log "Download-model summary"
    echo "  Source:      $S3_BUCKET/$MODEL_DIR/"
    echo "  Local path:  $local_model_dir"
}

# --------------------------------------------------------------------------- #
# Main dispatch
# --------------------------------------------------------------------------- #
usage() {
    echo "Usage: $0 {upload|download|upload-model|download-model}"
    echo ""
    echo "Commands:"
    echo "  upload         Upload training data files to S3"
    echo "  download       Download training data files from S3"
    echo "  upload-model   Upload trained model output to S3"
    echo "  download-model Download trained model from S3"
    exit 1
}

if [ $# -lt 1 ]; then
    usage
fi

case "$1" in
    upload)         do_upload ;;
    download)       do_download ;;
    upload-model)   do_upload_model ;;
    download-model) do_download_model ;;
    *)              echo "Unknown command: $1"; usage ;;
esac
