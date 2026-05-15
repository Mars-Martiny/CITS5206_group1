#!/bin/bash

set -e

ENV_NAME="hs_classifier"
MARKER=".installed"
FORCE=false

for arg in "$@"; do
    case $arg in
        --force) FORCE=true ;;
    esac
done

echo "============================================"
echo "  Incident Classifier — Installer"
echo "============================================"

if [ -f "$MARKER" ] && [ "$FORCE" = false ]; then
    echo "[Install] Already installed. Run with --force to reinstall."
    exit 0
fi

# --- Python version check ---
echo "[Install] Checking Python version..."
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 12 ]; }; then
    echo "[Error] Python 3.12+ is required. Found: $PYTHON_VERSION"
    exit 1
fi
echo "[Install] Python $PYTHON_VERSION OK"

# --- GPU detection ---
echo "[Install] Detecting GPU..."
TORCH_INDEX=""
GPU_TYPE="cpu"

if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
    GPU_TYPE="nvidia"
    TORCH_INDEX="https://download.pytorch.org/whl/cu130"
    echo "[Install] NVIDIA GPU detected — will install CUDA 13.0 build"
elif command -v rocm-smi &>/dev/null && rocm-smi &>/dev/null 2>&1; then
    GPU_TYPE="amd"
    TORCH_INDEX="https://download.pytorch.org/whl/rocm7.0"
    echo "[Install] AMD GPU detected — will install ROCm 7.0 build"
elif ls /sys/class/drm/*/device/driver/module 2>/dev/null | grep -q amdgpu; then
    GPU_TYPE="amd"
    TORCH_INDEX="https://download.pytorch.org/whl/rocm7.0"
    echo "[Install] AMD GPU detected via drm — will install ROCm 7.0 build"
elif [ "$(uname -m)" = "arm64" ] && [ "$(uname -s)" = "Darwin" ]; then
    GPU_TYPE="apple"
    echo "[Install] Apple Silicon detected — MPS acceleration available"
else
    echo "[Install] No GPU detected — using CPU-only build"
fi

# --- Environment setup ---
USE_CONDA=false

if command -v conda &>/dev/null; then
    USE_CONDA=true
    echo "[Install] conda found — setting up environment '$ENV_NAME'"

    # Initialise conda for non-interactive shell
    CONDA_BASE=$(conda info --base 2>/dev/null)
    source "$CONDA_BASE/etc/profile.d/conda.sh"

    if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
        echo "[Install] Conda environment '$ENV_NAME' already exists — reusing"
    else
        echo "[Install] Creating conda environment '$ENV_NAME' with Python 3.12..."
        conda create -n "$ENV_NAME" python=3.12 -y
    fi

    conda activate "$ENV_NAME"
    echo "[Install] Conda environment '$ENV_NAME' activated"
else
    echo "[Install] conda not found — falling back to venv (.venv)"

    if [ ! -d ".venv" ]; then
        echo "[Install] Creating .venv..."
        python3 -m venv .venv
    fi

    source .venv/bin/activate
    echo "[Install] venv activated"
fi

# --- Base dependencies (torch handled separately for GPU index) ---
echo "[Install] Installing base dependencies..."
pip install --upgrade pip -q

# Strip torch/torchvision lines so we can install the right build below
TMPFILE=$(mktemp)
grep -v -E "^torch" requirements.txt > "$TMPFILE"
pip install -r "$TMPFILE"
rm "$TMPFILE"

# --- PyTorch (GPU-appropriate) ---
echo "[Install] Installing PyTorch ($GPU_TYPE)..."
if [ -n "$TORCH_INDEX" ]; then
    pip install torch torchvision --index-url "$TORCH_INDEX"
else
    pip install torch torchvision
fi

# --- spaCy language model ---
echo "[Install] Downloading spaCy model (en_core_web_sm)..."
python3 -m spacy download en_core_web_sm

# --- Write marker ---
{
    echo "env_type=$([ "$USE_CONDA" = true ] && echo conda || echo venv)"
    echo "env_name=$ENV_NAME"
    echo "gpu=$GPU_TYPE"
} > "$MARKER"

echo ""
echo "============================================"
echo "  Installation complete!"
echo "  GPU backend : $GPU_TYPE"
echo "  Environment : $([ "$USE_CONDA" = true ] && echo "conda ($ENV_NAME)" || echo "venv (.venv)")"
echo "  Run the app : ./run_ui.sh"
echo "============================================"
