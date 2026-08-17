"""Unified policy deployment for the YAM station.

One transport-agnostic client (this package) drives the robot from any policy
server that speaks the YAM-ABC-Reproduce observation/action contract (see ``contract``).
Each policy backend (openpi, molmoact, abc) is hosted by its own server process
in its own virtualenv on the GPU box; the client here never imports torch/jax
and is identical regardless of which policy it talks to.

Client side (runs in the YAM-ABC-Reproduce env, on the robot host):
    from yam_abc_reproduce.deploy.client import WebsocketPolicyClient
    from yam_abc_reproduce.deploy.loop import DeployLoop

Server side (each runs in its backend's venv, on the GPU box):
    yam_abc_reproduce/deploy/servers/openpi_server.py
    yam_abc_reproduce/deploy/servers/molmoact_server.py
    yam_abc_reproduce/deploy/servers/abc_server.py

See yam_abc_reproduce/deploy/README.md for the end-to-end deployment story.
"""

from __future__ import annotations

__all__ = ["contract", "client", "loop"]
