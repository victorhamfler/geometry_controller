#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -n "${VENV_PATH:-}" ]]; then
  _venv_path="${VENV_PATH}"
elif [[ -d "${REPO_ROOT}/.venv_mlcpu" ]]; then
  _venv_path="${REPO_ROOT}/.venv_mlcpu"
else
  _venv_path="${REPO_ROOT}/.venv"
fi
VENV_PATH="${_venv_path}"
PYTHON_BIN="${VENV_PATH}/bin/python"
PIP_BIN="${VENV_PATH}/bin/pip"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[setup] creating venv at ${VENV_PATH}"
  python3 -m venv "${VENV_PATH}"
fi

"${PYTHON_BIN}" -m pip install --upgrade pip >/dev/null

if ! "${PYTHON_BIN}" -c "import torch, sentence_transformers" >/dev/null 2>&1; then
  echo "[setup] installing CPU torch + sentence-transformers"
  "${PIP_BIN}" install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
  "${PIP_BIN}" install --no-cache-dir sentence-transformers
fi

exec "${PYTHON_BIN}" "${REPO_ROOT}/scripts/run_ml_split_regression.py" "$@"
