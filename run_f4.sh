#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -n "${F4_PYTHON:-}" ]]; then
  py="$F4_PYTHON"
elif [[ -x .venv/bin/python ]]; then
  py=.venv/bin/python
else
  py=python
fi
exec "$py" -m stream4d.cli "$@"
