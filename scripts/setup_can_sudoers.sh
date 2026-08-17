#!/usr/bin/env bash
# One-time setup: install a scoped passwordless-sudo rule so the YAM-ABC-Reproduce backend
# can bring the CAN buses up without a password prompt. Matches the ip-link commands
# in i2rt's third_party/i2rt/scripts/reset_all_can.sh (run at go-live and via the
# GUI "Reset CAN" action).
#
#   Run once:  sudo bash scripts/setup_can_sudoers.sh
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "ERROR: must run as root.  Usage:  sudo bash $0" >&2
  exit 1
fi

USER_NAME="${SUDO_USER:-$(whoami)}"
IP="$(command -v ip || echo /usr/sbin/ip)"
SUDOERS_FILE="/etc/sudoers.d/yam-abc-can"

cat > "$SUDOERS_FILE" <<EOF
# Allow ${USER_NAME} to bring CAN interfaces up/down without a password (YAM-ABC-Reproduce).
# Installed by scripts/setup_can_sudoers.sh — scoped to can* interfaces only.
Cmnd_Alias YAM_ABC_CAN = ${IP} link set can* down, ${IP} link set can* up type can bitrate 1000000
${USER_NAME} ALL=(ALL) NOPASSWD: YAM_ABC_CAN
EOF

chmod 0440 "$SUDOERS_FILE"

if visudo -c -f "$SUDOERS_FILE" >/dev/null 2>&1; then
  echo "OK: installed ${SUDOERS_FILE} for user '${USER_NAME}'"
  echo "    Passwordless 'ip link set can* up/down' enabled; reset_can.sh now runs non-interactively."
else
  echo "ERROR: sudoers syntax check failed — removing bad file" >&2
  rm -f "$SUDOERS_FILE"
  exit 1
fi
