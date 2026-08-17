"""The policy-server connection must ignore any proxy set in the environment.

With a proxy exported (as the lab stations do), websockets routes even ws://127.0.0.1 at
the proxy, which fails as if the server were down -- a socks5:// value dies with
"connecting through a SOCKS proxy requires python-socks".
"""

import os
import urllib.request

from yam_abc_reproduce.deploy.client import _bypass_proxy_for


def test_bypass_exempts_the_policy_server(monkeypatch):
    monkeypatch.setenv("https_proxy", "socks5://proxy.example:10081")
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    # urllib does not exempt loopback on its own -- this is the trap.
    assert not urllib.request.proxy_bypass("127.0.0.1:8000")

    _bypass_proxy_for("127.0.0.1")

    assert urllib.request.proxy_bypass("127.0.0.1:8000")
    # Both spellings: either can win urllib's scan over os.environ.
    assert os.environ["no_proxy"] == "127.0.0.1"
    assert os.environ["NO_PROXY"] == "127.0.0.1"


def test_bypass_is_additive_and_idempotent(monkeypatch):
    """A remote GPU box must not clobber an operator's existing no_proxy entries."""
    monkeypatch.setenv("no_proxy", "example.com")
    monkeypatch.delenv("NO_PROXY", raising=False)

    _bypass_proxy_for("gpu-box.lan")
    _bypass_proxy_for("gpu-box.lan")

    assert os.environ["no_proxy"] == "example.com,gpu-box.lan"
