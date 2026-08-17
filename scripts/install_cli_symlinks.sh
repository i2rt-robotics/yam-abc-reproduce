#!/usr/bin/env bash
# Make the yam-abc-* CLI commands usable without activating the venv, by
# symlinking the venv entry points into /usr/local/bin (asks for sudo once).
#
#   Run once after install:  bash scripts/install_cli_symlinks.sh
set -euo pipefail

VENV_BIN="$(cd "$(dirname "$0")/.." && pwd)/.venv/bin"
if [ ! -d "$VENV_BIN" ]; then
  echo "ERROR: $VENV_BIN not found — create the venv and install the package first." >&2
  exit 1
fi

found=0
for tool in "$VENV_BIN"/yam-abc-*; do
  [ -x "$tool" ] || continue
  name="$(basename "$tool")"
  sudo ln -sf "$tool" "/usr/local/bin/$name"
  echo "linked /usr/local/bin/$name -> $tool"
  found=1
done

if [ "$found" -eq 0 ]; then
  echo "ERROR: no yam-abc-* entry points in $VENV_BIN — did the package install succeed?" >&2
  exit 1
fi
echo "OK: the yam-abc-* commands now work from any shell (no venv activation needed)."
