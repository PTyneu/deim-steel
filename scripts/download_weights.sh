#!/usr/bin/env bash
# Download all weights for the experiment: DEIMv2 COCO checkpoint (weights/) + backbone (ckpts/: DINOv3 or ViT-Tiny).
#   bash scripts/download_weights.sh                 # model from experiment.yml
#   bash scripts/download_weights.sh x               # a given size: s | m | l | x | all
#   bash scripts/download_weights.sh x --no-backbone # only the detector checkpoint
set -euo pipefail
cd "$(dirname "$0")/.."
# interpreter: $PYTHON, else the repo venv made by `uv sync`, else python on PATH
if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe
elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
else PY=python; fi
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
"$PY" tools/steel/download_weights.py "$@"
