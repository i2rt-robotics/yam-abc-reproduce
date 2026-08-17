#!/usr/bin/env bash
# Start the long-lived ZMQ camera server for the YAM eval stack.
# Owns all 3 RealSense cameras so the eval client can pull obs on demand.
# Run from the molmoact2 repo root; leave it running across eval sessions.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-$HERE/configs/yam_left.yaml}"
exec python "$HERE/camera_server.py" --config "$CONFIG"
