"""openpi (pi0 / pi0.5) policy server on the YAM-ABC-Reproduce wire protocol.

Needs the `openpi` group (uv sync --extra deploy --group openpi). Loads your trained
checkpoint via openpi's ``create_trained_policy`` and serves it with openpi's native
``WebsocketPolicyServer``, which already speaks the client's wire protocol -- so there
is no custom wire code here.

The only glue is a key remap: the client sends

    {"images": {<role>: HWC uint8}, "state": (S,), "prompt": str}

while your trained config's Input transform expects the feature keys your LeRobot
dataset used (see data/formats/lerobot_format.py and your openpi TrainConfig). Bridge
the two with --image-key-map; a config already reading `images`/`state`/`prompt`
(aloha-style) needs no flag.

Run (cd third_party/policy/openpi for its sources; python from the repo's .venv):
    python <yam_abc_reproduce>/yam_abc_reproduce/deploy/servers/openpi_server.py \
        --config <your_train_config_name> \
        --checkpoint checkpoints/<your_config>/<exp>/<step> \
        --prompt "default instruction" --port 8000

    # remap YAM-ABC-Reproduce camera roles to your model's image keys, e.g.:
    #   --image-key-map top=cam_high,left=cam_left_wrist,right=cam_right_wrist
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np

# Serving fetches the PaliGemma tokenizer from gs://big_vision via gcsfs, whose aiohttp
# session ignores http_proxy without trust_env=True. Must be at module scope: fsspec reads
# FSSPEC_GS when imported, which happens under the `from openpi...` imports in main().
os.environ.setdefault("FSSPEC_GS", '{"session_kwargs": {"trust_env": true}}')

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("yam_abc_reproduce.deploy.openpi")


def _parse_key_map(spec: str | None) -> dict[str, str]:
    if not spec:
        return {}
    out: dict[str, str] = {}
    for pair in spec.split(","):
        role, _, key = pair.partition("=")
        if not key:
            raise SystemExit(f"bad --image-key-map entry {pair!r}; use role=model_key")
        out[role.strip()] = key.strip()
    return out


class RemapPolicy:
    """Wrap an openpi Policy so it accepts the YAM-ABC-Reproduce observation dict.

    Renames image keys per ``image_key_map`` (role -> model key); passes
    ``state``/``prompt`` through. If your config expects images flattened as
    top-level ``observation/...`` keys instead of a nested ``images`` dict,
    set ``flatten_prefix`` and each image goes to
    ``f"{flatten_prefix}{model_key}"``.
    """

    def __init__(self, policy, image_key_map: dict[str, str], flatten_prefix: str | None,
                 state_key: str = "state"):
        self._policy = policy
        self._map = image_key_map
        self._flatten = flatten_prefix
        self._state_key = state_key
        self.metadata = getattr(policy, "metadata", {})

    def infer(self, obs: dict) -> dict:
        images = obs.get("images", {})
        remapped = {self._map.get(role, role): np.asarray(img) for role, img in images.items()}
        model_obs: dict = {self._state_key: np.asarray(obs["state"], dtype=np.float32)}
        if "prompt" in obs:
            model_obs["prompt"] = str(obs["prompt"])
        if self._flatten:
            for key, img in remapped.items():
                model_obs[f"{self._flatten}{key}"] = img
        else:
            model_obs["images"] = remapped
        return self._policy.infer(model_obs)

    def reset(self) -> None:
        if hasattr(self._policy, "reset"):
            self._policy.reset()


def main() -> None:
    p = argparse.ArgumentParser(description="openpi YAM-ABC-Reproduce websocket server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--config", required=True, help="openpi TrainConfig name (config.get_config)")
    p.add_argument("--checkpoint", required=True, help="checkpoint dir (local or gs://)")
    p.add_argument("--prompt", default=None, help="default prompt if obs lacks one")
    p.add_argument("--device", default=None, help="pytorch device for torch checkpoints")
    p.add_argument(
        "--image-key-map",
        default=None,
        help="comma list role=model_key, e.g. top=cam_high,left=cam_left_wrist",
    )
    p.add_argument(
        "--flatten-prefix",
        default=None,
        help="if set, images go to top-level keys f'{prefix}{model_key}' "
        "(e.g. 'observation/') instead of a nested 'images' dict",
    )
    p.add_argument(
        "--state-key",
        default="state",
        help="key the model's Input transform reads state from (e.g. "
        "'observation/state' for the YAM-ABC-Reproduce YAM config; default 'state')",
    )
    args = p.parse_args()

    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as _config

    train_config = _config.get_config(args.config)

    policy = policy_config.create_trained_policy(
        train_config,
        args.checkpoint,
        default_prompt=args.prompt,
        pytorch_device=args.device,
    )
    wrapped = RemapPolicy(policy, _parse_key_map(args.image_key_map), args.flatten_prefix, args.state_key)

    # Expose the raw state dim (7 per arm) in metadata for the client's dim check (YAM configs).
    meta = {"backend": "openpi", "config": args.config, **wrapped.metadata}
    num_arms = getattr(train_config.data, "num_arms", None)
    if num_arms is not None:
        meta["state_dim"] = 7 * num_arms

    # Warm-up inference: trigger the XLA JIT compile now rather than on the first live request.
    try:
        import time as _time
        n_arms = getattr(train_config.data, "num_arms", None) or 2
        dummy = {
            "images": {role: np.zeros((480, 640, 3), dtype=np.uint8)
                       for role in _parse_key_map(args.image_key_map)},
            "state": np.zeros(7 * n_arms, dtype=np.float32),
            "prompt": args.prompt or "warmup",
        }
        log.info("warm-up inference (JIT compile, may take 1-2 min)...")
        t0 = _time.time()
        wrapped.infer(dummy)
        log.info("warm-up done in %.1f s", _time.time() - t0)
    except Exception as e:  # noqa: BLE001
        log.warning("warm-up inference failed (continuing): %s", e)

    log.info("serving openpi config=%s on %s:%d", args.config, args.host, args.port)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=wrapped,
        host=args.host,
        port=args.port,
        metadata=meta,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
