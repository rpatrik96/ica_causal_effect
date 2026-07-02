#!/bin/bash
# HTCondor job wrapper for sweep_runner.py.
# Usage: run_autoresearch_job.sh <idx> <script> <outdir> <args...>
# Resolves the project dir from its own location — no hardcoded cluster paths.
set -uo pipefail

IDX="$1"; SCRIPT="$2"; OUTDIR="$3"; shift 3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

mkdir -p "${OUTDIR}"
echo "[job ${IDX}] python ${SCRIPT} $*"
python "${SCRIPT}" "$@"
STATUS=$?

if [ ${STATUS} -eq 0 ]; then
    touch "${OUTDIR}/DONE"
fi
exit ${STATUS}
