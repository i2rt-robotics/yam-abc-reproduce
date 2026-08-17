"""Minimal server side of the openpi websocket protocol.

``openpi_client.WebsocketClientPolicy`` (used by ``yam_abc_reproduce.deploy.client``)
expects, per connection:

  1. a single msgpack frame with the server metadata dict, sent on connect;
  2. then, per request: it sends a msgpack-packed observation dict and expects
     back a msgpack-packed response dict; if the server instead sends a *string*
     frame, the client raises it as an error.

openpi's own ``WebsocketPolicyServer`` does this with asyncio, but pulls in the full
(heavy) openpi package, which the molmoact/abc extras do not want. Reusing
``openpi_client.msgpack_numpy`` for encoding is what guarantees wire compatibility.

Needs ``websockets`` + ``openpi_client``. The `openpi` extra provides both; with
`molmoact2*` / `abc-policy` alone, put openpi-client on PYTHONPATH -- see docs/deploy.md.
"""

from __future__ import annotations

import http
import logging
import traceback
from collections.abc import Callable
from typing import Any

log = logging.getLogger("yam_abc_reproduce.deploy.server")


def serve(
    infer: Callable[[dict[str, Any]], dict[str, Any]],
    host: str,
    port: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Serve ``infer`` (obs dict -> response dict) forever over websocket.

    ``infer`` must return a dict containing at least ``{"actions": ndarray}``.
    Exceptions are sent to the client as a string frame (which it re-raises),
    then the connection is closed -- matching openpi's server semantics.
    """
    from openpi_client import msgpack_numpy
    from websockets.sync.server import serve as ws_serve

    packer = msgpack_numpy.Packer()
    meta = metadata or {}

    def handler(conn) -> None:
        log.info("client connected: %s", conn.remote_address)
        conn.send(packer.pack(meta))
        while True:
            try:
                raw = conn.recv()
            except Exception:  # noqa: BLE001 -- ConnectionClosed and friends
                log.info("client disconnected: %s", conn.remote_address)
                return
            try:
                obs = msgpack_numpy.unpackb(raw)
                response = infer(obs)
                conn.send(packer.pack(response))
            except Exception:  # noqa: BLE001
                tb = traceback.format_exc()
                log.exception("inference failed")
                try:
                    conn.send(tb)  # string frame -> client raises RuntimeError
                    conn.close()
                finally:
                    return

    def health(conn, request):
        """/healthz -> 200, as openpi's server does: keeps the GUI's liveness poll from
        provoking a handshake-failure traceback."""
        if request.path == "/healthz":
            return conn.respond(http.HTTPStatus.OK, "OK\n")
        return None  # anything else: continue into the normal websocket handshake

    log.info("serving on ws://%s:%d  (metadata=%s)", host, port, meta)
    with ws_serve(
        handler, host, port, compression=None, max_size=None, process_request=health
    ) as server:
        server.serve_forever()
