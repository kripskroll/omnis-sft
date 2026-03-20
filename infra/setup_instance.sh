#!/bin/bash
# Bootstrap an AWS g5.2xlarge instance for Omnis SFT training.
#
# Prerequisites:
#   - EC2 g5.2xlarge with Deep Learning AMI (Ubuntu 22.04)
#   - SSH access configured
#
# Usage:
#   ssh ubuntu@<instance-ip> 'bash -s' < infra/setup_instance.sh
#   # OR
#   scp infra/setup_instance.sh ubuntu@<instance-ip>:~ && ssh ubuntu@<instance-ip> './setup_instance.sh'

set -euo pipefail

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
log()  { echo -e "\n\033[1;34m>>> $*\033[0m"; }
ok()   { echo -e "    \033[1;32m[OK]\033[0m $*"; }
fail() { echo -e "    \033[1;31m[FAIL]\033[0m $*"; exit 1; }

REPO_URL="https://github.com/kripskroll/omnis-sft.git"
REPO_DIR="$HOME/omnis-sft"

# --------------------------------------------------------------------------- #
# 1. Verify NVIDIA GPU / CUDA drivers
# --------------------------------------------------------------------------- #
log "Checking NVIDIA GPU and CUDA drivers"

if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    ok "nvidia-smi found"
else
    fail "nvidia-smi not found. Are NVIDIA drivers installed? (expected on Deep Learning AMI)"
fi

if command -v nvcc &>/dev/null; then
    CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[\d.]+')
    ok "CUDA toolkit $CUDA_VERSION"
else
    echo "    [WARN] nvcc not on PATH — CUDA toolkit may still be available via PyTorch wheels"
fi

# --------------------------------------------------------------------------- #
# 2. Ensure Python 3.11+
# --------------------------------------------------------------------------- #
log "Checking Python version"

# Prefer python3.11 if available, else fall back to python3
PYTHON_CMD=""
for cmd in python3.11 python3.12 python3; do
    if command -v "$cmd" &>/dev/null; then
        PY_VER=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        PY_MAJOR=$("$cmd" -c "import sys; print(sys.version_info.major)")
        PY_MINOR=$("$cmd" -c "import sys; print(sys.version_info.minor)")
        if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 11 ]; then
            PYTHON_CMD="$cmd"
            ok "Found $cmd ($PY_VER)"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    log "No Python 3.11+ found — installing via deadsnakes PPA"
    sudo apt-get update -qq
    sudo apt-get install -y -qq software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3.11 python3.11-venv python3.11-dev
    PYTHON_CMD="python3.11"
    ok "Installed Python 3.11"
fi

# --------------------------------------------------------------------------- #
# 3. Install uv (Python package manager)
# --------------------------------------------------------------------------- #
log "Installing uv"

if command -v uv &>/dev/null; then
    ok "uv already installed ($(uv --version))"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add to current session PATH
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if command -v uv &>/dev/null; then
        ok "uv installed ($(uv --version))"
    else
        fail "uv installation failed"
    fi
fi

# Ensure uv is on PATH for future shells
for rc_file in "$HOME/.bashrc" "$HOME/.profile"; do
    if [ -f "$rc_file" ]; then
        if ! grep -q '.local/bin' "$rc_file" 2>/dev/null; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$rc_file"
        fi
    fi
done

# --------------------------------------------------------------------------- #
# 4. Clone the omnis-sft repository
# --------------------------------------------------------------------------- #
log "Setting up omnis-sft repository"

if [ -d "$REPO_DIR/.git" ]; then
    ok "Repository already cloned at $REPO_DIR"
    cd "$REPO_DIR"
    git pull --ff-only || echo "    [WARN] git pull failed — continuing with existing checkout"
else
    git clone "$REPO_URL" "$REPO_DIR"
    ok "Cloned $REPO_URL"
    cd "$REPO_DIR"
fi

# --------------------------------------------------------------------------- #
# 5. Set up Python virtual environment and install dependencies
# --------------------------------------------------------------------------- #
log "Setting up Python environment"

cd "$REPO_DIR"

if [ ! -d ".venv" ]; then
    uv venv --python "$PYTHON_CMD"
    ok "Created virtual environment"
else
    ok "Virtual environment already exists"
fi

# Activate venv
source .venv/bin/activate

log "Installing training dependencies"
uv pip install -e ".[training]"
ok "Dependencies installed"

# --------------------------------------------------------------------------- #
# 6. Verify critical imports
# --------------------------------------------------------------------------- #
log "Verifying Python packages"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'PyTorch {torch.__version__} — CUDA available: {torch.cuda.is_available()}')" \
    && ok "PyTorch + CUDA" \
    || fail "PyTorch CUDA verification failed"

python -c "import transformers; print(f'Transformers {transformers.__version__}')" \
    && ok "Transformers" \
    || fail "Transformers import failed"

python -c "import unsloth; print('Unsloth OK')" \
    && ok "Unsloth" \
    || echo "    [WARN] Unsloth not available — may need manual install"

python -c "import datasets; print(f'Datasets {datasets.__version__}')" \
    && ok "Datasets" \
    || echo "    [WARN] datasets not available"

# --------------------------------------------------------------------------- #
# 7. Print verification summary
# --------------------------------------------------------------------------- #
log "Setup Summary"
echo "============================================================"
echo "  GPU:            $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  GPU Memory:     $(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)"
echo "  Driver:         $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
echo "  Python:         $(python --version 2>&1)"
echo "  uv:             $(uv --version)"
echo "  PyTorch:        $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA (torch):   $(python -c 'import torch; print(torch.version.cuda)')"
echo "  Repo:           $REPO_DIR"
echo "  Venv:           $REPO_DIR/.venv"
echo "  Disk free:      $(df -h / | awk 'NR==2{print $4}')"
echo "  Disk used:      $(du -sh "$REPO_DIR" 2>/dev/null | cut -f1)"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  cd $REPO_DIR && source .venv/bin/activate"
echo "  # Upload training data via:  ./infra/upload_to_s3.sh download"
echo "  # Start training:            python training/train.py"
echo ""
ok "Instance setup complete!"
