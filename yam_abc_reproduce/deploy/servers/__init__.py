"""Policy server adapters -- one per backend, each run with only that backend installed.

These scripts are standalone: they do NOT import the ``yam_abc_reproduce`` package,
because the backends pin incompatible deps (torch 2.5.1 for molmoact, jax for openpi --
hence the conflicts declared in pyproject.toml). Each imports only its own backend plus
the shared ``_wire`` shim, which speaks the openpi wire protocol -- so the one client,
``yam_abc_reproduce.deploy.client``, talks to all of them.

Run each on the GPU box after `uv sync`-ing that backend's group, cd'd into its
submodule (where the backend's own sources live):

    # openpi (uv sync --extra deploy --group openpi; cd third_party/policy/openpi)
    python yam_abc_reproduce/deploy/servers/openpi_server.py --config <name> --checkpoint <dir> --port 8000

    # molmoact (uv sync --extra deploy --group molmoact2-train; cd third_party/policy/molmoact2)
    python yam_abc_reproduce/deploy/servers/molmoact_server.py --port 8202

    # abc (uv sync --extra deploy --group abc-policy; cd third_party/policy/abc)
    python yam_abc_reproduce/deploy/servers/abc_server.py --checkpoint <ckpt> --prompt "..." --port 8300
"""
