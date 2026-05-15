#!/bin/bash

set -e

MARKER=".installed"

echo "[Docs] Building Sphinx documentation..."

if [ -f "$MARKER" ]; then
    ENV_TYPE=$(grep "^env_type=" "$MARKER" | cut -d= -f2)
    ENV_NAME=$(grep "^env_name=" "$MARKER" | cut -d= -f2)

    if [ "$ENV_TYPE" = "conda" ]; then
        CONDA_BASE=$(conda info --base 2>/dev/null)
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        conda activate "$ENV_NAME"
    else
        source .venv/bin/activate
    fi
else
    echo "[Docs] Warning: .installed not found — using current Python environment"
fi

cd docs
make html

echo ""
echo "[Docs] Build complete. Output: docs/build/html/index.html"
