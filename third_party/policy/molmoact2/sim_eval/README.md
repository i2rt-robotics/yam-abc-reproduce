# sim_eval — MolmoAct2 Simulation Evaluation

Zero-shot evaluation of [MolmoAct2](https://huggingface.co/allenai/MolmoAct2-DROID) policies inside [ManiSkill](https://github.com/haosulab/ManiSkill) simulation.

## Directory layout

```
sim_eval/
├── run_eval.py          # CLI entry point
├── inference/
│   ├── client.py        # DroidClient / YAMClient (HTTP ↔ /act)
│   └── common.py        # Schemas, state/action adapters, obs helpers
├── robots/
│   ├── franka_droid.py  # Franka FR3 + Robotiq gripper (DROID)
│   └── bimanual_yam.py  # Bimanual YAM arms (YAM)
├── tasks/
│   ├── droid_tasks/
│   │   └── droid_put_everything_in_box.py
│   └── yam_tasks/
│       └── bimanual_put_everything_in_box.py
├── assets/              # Robot meshes / URDFs
└── scripts/
    └── download_assets.py
```

## Setup

**1. Install dependencies**

```bash
uv sync          # from repo root
```

**2. Download robot assets**

Assets (URDF meshes for Franka, MJCF files for YAM) are not committed to the repo.
Download them once:

```bash
uv run python sim_eval/scripts/download_assets.py
```

This pulls from `TreeePlanter/molmoact2-sim-eval-assets` on HuggingFace and places the
files under `sim_eval/assets/`.  Pass `--force` to re-download.

## Running evaluation

Start the inference server as described in the [main README](../README.md), then run the evaluator:

```bash
# YAM
uv run python -m sim_eval.run_eval \
    --policy-type remote-yam \
    --remote-url http://<host>:8202/act \
    -e BimanualYAMPutEverythingInBox-v1

# DROID
uv run python -m sim_eval.run_eval \
    --policy-type remote-droid \
    --remote-url http://<host>:8000/act \
    -e DroidPutEverythingInBox-v1
```

Results are written to `sim_eval/outputs/<timestamp>/results.json`.
Videos and per-episode camera frames are saved alongside.

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--policy-type` / `-p` | `remote-yam` | `remote-droid` or `remote-yam` |
| `--remote-url` | — | Full `/act` endpoint URL (required) |
| `-e` | — | One or more ManiSkill env IDs |
| `-n` | `10` | Episodes per task |
| `--max-episode-steps` | `800` | Step limit per episode |
| `--language-instruction` | per-env default | Language instruction override |
| `--n-action-steps` | full chunk | Actions to execute per server call |
| `--save-video` | `True` | Save rollout videos |

## Available environments

| Env ID | Robot | Task |
|--------|-------|------|
| `DroidPutEverythingInBox-v1` | Franka FR3 + Robotiq | Pick lego duplo + tennis ball → box |
| `BimanualYAMPutEverythingInBox-v1` | Bimanual YAM | Pick lego duplo + tennis ball → box |

## Adding a new task

1. Create `sim_eval/tasks/<embodiment>_tasks/my_task.py` — subclass `BaseEnv`, register
   with `@register_env`, import the robot from `...robots.<robot>` (side-effect: registers
   the agent with ManiSkill).
2. Add an import in the parent `__init__.py` so `from ..tasks import *` picks it up.
3. Add an entry in `DEFAULT_LANGUAGE_INSTRUCTIONS` in `run_eval.py`.
