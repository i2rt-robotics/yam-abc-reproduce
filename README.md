# YAM-ABC-Reproduce

A lightweight **teleop → collect → train → deploy** pipeline for the
[YAM](https://github.com/i2rt-robotics/i2rt) (i2rt) bimanual robot arm — the whole loop from
one small repo with a single GUI. Teleoperate to record demonstrations, convert to a
standard dataset, fine-tune a VLA policy, and run it on the real arms.

- **Collect** — leader→follower teleop at 30 Hz with synchronized multi-camera recording.
- **Review** — inspect episodes (video + joint/gripper curves) and prune bad ones in-browser.
- **Train** — policy backends wired in: **π<sub>0</sub> / π<sub>0.5</sub>** (openpi), **MolmoAct2**, and **ABC-DiT**.
- **Deploy** — one transport-only client + a policy server per backend; swap policies by changing `--host/--port`.

> **Scope: one station, one GPU.** The pipeline is validated end to end on a single-GPU station
> — one box with one NVIDIA card (a 32 GB RTX 5090) doing collect, review, convert, train and serve.
> Splitting the robot host from the policy server across two machines works and is documented
> in [deploy.md](docs/deploy.md). Multi-GPU π<sub>0</sub>/π<sub>0.5</sub> training (JAX FSDP) runs
> on an 8× RTX PRO 6000 Blackwell box; **multi-GPU molmoact2/abc is still untested**, and on the
> station itself there is one card, so expect one GPU job at a time — see
> [One GPU at a time](docs/training.md#one-gpu-at-a-time).
>
> On a station with more than one card, every GPU job pins itself: a trainer takes the `GPUs`
> cards with the most free VRAM at launch, a policy server takes the single freest, and each
> one echoes `[yam-abc] pinned to GPU(s) …` as its first log line. To restrict the GUI to a
> subset of the box, `export CUDA_VISIBLE_DEVICES=0,3` (or any visible indices) **before**
> starting `yam-abc-gui` — the GUI runs every trainer and policy server as a child process, so
> they inherit it, which is why it has to be set before the GUI rather than after. That list is
> a ceiling the picker never reaches outside of; the `GPUs` field only chooses how many of it a
> job gets. Do the same before a bare `yam-abc-deploy` or a hand-run policy server, which do no
> picking of their own. One caveat: the **GPU** badge shells out to `nvidia-smi`, which ignores
> `CUDA_VISIBLE_DEVICES` and always reports the *first physical* card — so when a job lands
> elsewhere, the badge describes a different GPU than the one it is on.

## How it works

```
                    ┌─────────────────────── YAM-ABC-Reproduce GUI (FastAPI) ───────────────────────┐
                    │   Collect          Review            Train              Deploy        │
                    └──────┬───────────────┬─────────────────┬───────────────────┬──────────┘
 YAM leader ─teleop─▶ ControlLoop ─▶ EpisodeRecorder    yam-abc-convert    yam-abc-deploy
   (passive GELLO)   30 Hz, e-stop   default format     ├─▶ LeRobot ─▶ π0/π0.5, MolmoAct2 │ obs dict
        │                 ▲              │              └─▶ ABC MCAP ─▶ ABC-DiT           ▼ (websocket)
     follower ◀───────────┘          CameraHub                                    policy server (GPU box)
```


## Install

Clone the repository. The policy backends (openpi / molmoact2 / abc) are vendored
directly under `third_party/policy/` (provenance in `third_party/policy/VERSIONS.md`);
the only submodule is `third_party/i2rt`, and there is never a `uv sync` to run
*inside* a backend. `i2rt`, `openpi` (+ `openpi-client`) and `molmoact2/experiments`
are installed editable from the checkout; `policy/abc` and the `policy/molmoact2`
root declare no `[build-system]` and cannot be installed, so their dependency-groups
(`abc-policy`, `molmoact2`) carry a hand-mirrored copy of their dependency lists and
their code runs in place. That submodule's remote is SSH, so you need a GitHub SSH key:
```bash
git clone --recurse-submodules git@github.com:i2rt-robotics/yam-abc-reproduce.git
# already cloned without the submodule?
git submodule update --init
```

Install — everything except a policy backend. The whole flow is driven by
[uv](https://docs.astral.sh/uv/), so install that first if you don't have it
(`curl -LsSf https://astral.sh/uv/install.sh | sh`); `uv.lock` needs a reasonably recent
release — tested on 0.9.x.
```bash
cd yam-abc-reproduce
sudo apt install build-essential python3-dev ffmpeg   # i2rt builds ruckig from source; ffmpeg for the ABC cache build
uv sync --all-extras
sudo bash scripts/setup_can_sudoers.sh   # bring CAN up without a sudo prompt at runtime
```

`uv sync` creates `.venv` (Python 3.12, pinned by `.python-version`), installs everything
from `uv.lock` — including `i2rt` from `third_party/`, which is a base dependency and so is
always installed — and installs this package editable so the `yam-abc-*` commands work.
There is no `uv pip install` step.

<a id="extras-table"></a>
### What to sync for what

Optional features are **extras** and always combine. The four policy backends are
**dependency-groups** and are mutually exclusive — pick at most one with `--group`.

| Use case | Command |
|---|---|
| Collect & Review (station) | `uv sync --extra camera --extra gui` |
| …plus converting episodes to LeRobot/ABC | `uv sync --extra camera --extra gui --extra convert` |
| Train, via the GUI (GPU box) | `uv sync --extra gui --extra convert --group <backend>` |
| Deploy — robot host (client only) | `uv sync --extra camera --extra gui --extra deploy` |
| Deploy — policy server (GPU box) | `uv sync --extra deploy --group <backend>` |
| One box that does everything | `uv sync --all-extras --group <backend>` |

`<backend>` is exactly one of `openpi` (π<sub>0</sub>/π<sub>0.5</sub>), `molmoact2-train`, or
`abc-policy` — see [docs/training.md](docs/training.md). The standalone MolmoAct2 *inference
server* is `molmoact2`, and it additionally collides with `lerobot`/`convert`
(`torchvision==0.20.1` vs `>=0.21`), so it needs
`uv sync --all-extras --no-extra lerobot --no-extra convert --group molmoact2`.

Three things worth knowing:

- **`uv sync` prunes.** The venv is made to match *exactly* what you list, so anything left
  off the command line is uninstalled — the `--group` included. `--all-extras` covers the
  extras half in one flag; the backend still has to be repeated each time.
- **Video is a base dependency, not an extra.** The recorder writes H.264 through PyAV
  (`av`), so it is installed on every box rather than hidden behind `camera`/`convert` — a
  station that only records still needs to encode. Where a GPU is present, encoding and
  decoding use **NVENC/NVDEC** automatically; see
  [Video encoding](docs/collect.md#video-encoding). System `ffmpeg` is no longer part of
  this path (Review used to shell out to it) and is now only needed for the ABC
  training-cache build.
- **`--no-default-groups`** drops the `dev` tools. On a robot host that only teleoperates and
  talks to a remote policy server, that plus the client extras is the lean install:
  ```bash
  uv sync --extra camera --extra gui --extra deploy --no-default-groups
  ```

Sanity-check a station (read-only; safe during teleop):

```bash
yam-abc-doctor        # CAN naming/udev + passwordless sudo, per-bus traffic fingerprints vs
                      # config, RealSense serials, free VRAM (GPU 0), ffmpeg, and which
                      # policy backend the venv holds.  --no-listen skips the 1s-per-bus
                      # passive CAN listen; --json for machine-readable output
```

Expect two WARN rows under `sys.backend.*`: only one policy backend fits the venv at a time,
so the two you haven't synced are always reported missing — that's by design, not breakage.
Exit code is 1 if any row FAILs.

> The policy backends (openpi / molmoact2 / abc) are vendored directly in
> `third_party/policy/` (see VERSIONS.md there for provenance); the only
> submodule is `third_party/i2rt`.

Optional — use the `yam-abc-*` commands without activating the venv:

```bash
bash scripts/install_cli_symlinks.sh   # symlinks them into /usr/local/bin (sudo once)
```

## Hardware Setup
Hardware wiring (CAN buses, udev names, station/camera config, leader calibration) is in
**[docs/hardware.md](docs/hardware.md)**.

## Quickstart

All `yam-abc-*` commands assume the repo venv is active — `cd yam-abc-reproduce && source .venv/bin/activate` first. Or skip activation and prefix with `uv run` (e.g. `uv run yam-abc-gui`).

```bash
yam-abc-gui          # http://localhost:8042
```

Opens the GUI. **Collect** and **Review** work
end-to-end — teleoperate, record a few episodes, and inspect them. The **Train** and
**Deploy** tabs are browsable here; running them for real needs that backend's dependency-group installed (`uv sync --all-extras --group <backend>`). Full walkthrough: [collect.md](docs/collect.md) → [training.md](docs/training.md) →
[deploy.md](docs/deploy.md).

## Documentation

| Guide | What it covers |
|---|---|
| [docs/hardware.md](docs/hardware.md) | CAN buses, station/camera config, leader calibration, verifying teleop |
| [docs/collect.md](docs/collect.md) | Collect & Review — teleop recording, plus the Collect/Review tab controls |
| [docs/training.md](docs/training.md) | Convert & train, per-backend (π<sub>0</sub> / π<sub>0.5</sub> / MolmoAct2 / ABC), plus the Train tab controls |
| [docs/deploy.md](docs/deploy.md) | Run a trained policy — Deploy tab controls, robot host + GPU-box server, obs/action contract |

## CLI reference

| Command | What it does |
|---|---|
| `yam-abc-gui` | Launch the web GUI (Collect / Review / Train / Deploy) |
| `yam-abc-cameras` | List connected cameras + serials for `configs/cameras.yaml` |
| `yam-abc-teleop --record "<task>" --seconds 15` | Record one episode headless |
| `yam-abc-convert <episodes> --to lerobot\|abc --repo-id <name>` | Convert recorded data for training |
| `yam-abc-viz --repo-id <name> --episode-index 0` | Inspect a LeRobot dataset (Rerun) — **currently broken** with the locked LeRobot; use the Review tab, see [training.md](docs/training.md) |
| `yam-abc-deploy --host <ip> --port <p> --prompt "..."` | Run a policy rollout from the terminal |
| `yam-abc-doctor [--config <yaml>] [--no-listen] [--json]` | Read-only station self-check (CAN / cameras / GPU / deps); exit 1 if any row FAILs |

`yam-abc-teleop`, `yam-abc-gui` and `yam-abc-deploy` accept `--mock` (mock robot + cameras).
`yam-abc-cameras`, `yam-abc-convert`, `yam-abc-viz` and `yam-abc-doctor` do not.

## Project layout

```
yam_abc_reproduce/robot/     RobotInterface + yam_adapter/passive_gello (i2rt boundary) + can_bus + mock
yam_abc_reproduce/camera/    CameraDriver + realsense/decxin drivers + mock + discovery
yam_abc_reproduce/data/      schema, recorder, formats/{default,lerobot,abc}, visualize
yam_abc_reproduce/teleop/    ControlLoop (read → act → command → record, e-stop)
yam_abc_reproduce/deploy/    transport-only client + loop + servers/{openpi,molmoact,abc}
yam_abc_reproduce/gui/       FastAPI server, session, jobs, camera_hub, builders, static frontend
scripts/            calibrate_gello, CAN setup (sudoers/udev), CLI symlinks, CAN diagnostics, ABC data-prep
configs/            station_yam.yaml, cameras.yaml
third_party/        i2rt (submodule) + vendored policy/{openpi,molmoact2,abc}
docs/               setup / training / deploy guides
```

## Safety

The Deploy tab's **E-STOP** (and `Ctrl-C` for terminal runs) halts the control loop and
any deploy process immediately. Keep a hardware e-stop within reach, and start every
rollout with the arms in a safe pose. See the safety notes in each guide.

## Common Issues
1. i2rt pulls in `ruckig`, which ships no prebuilt wheel and compiles C++ at install time, so
`uv sync` needs system build tools — on *every* install, since i2rt is a base dependency.
ffmpeg is separate — recording and Review use PyAV, but abc's `export_mcap.py` shells out to
system `ffmpeg`/`ffprobe` when building the training cache. `linux-headers` is for the CAN
driver, not the build:
```bash
sudo apt install build-essential python3-dev linux-headers-$(uname -r) ffmpeg
uv sync --all-extras
sudo bash scripts/setup_can_sudoers.sh   # bring CAN up without a sudo prompt at runtime
```
The ruckig build also needs `scikit-build-core<0.10` (newer versions reject its deprecated
`cmake.targets`). That is pinned declaratively via `[tool.uv] build-constraint-dependencies`
in `pyproject.toml`, so uv compiles it automatically and there is nothing to install by hand.
Only if you install i2rt with plain pip from somewhere else do you still need the manual
dance (`pip install "scikit-build-core<0.10"` first, then `--no-build-isolation`).

2. `uv sync` fails with a missing `third_party/...` path. uv validates every path dependency on
each sync, even when the extra or group using it isn't selected, and there are four (`i2rt`,
`policy/openpi`, `policy/openpi/packages/openpi-client`, `policy/molmoact2/experiments`). Three
of those are vendored, so they are present in any clone; the one that can genuinely be missing
is `third_party/i2rt`, the only submodule — run `git submodule update --init`.

3. `error: Groups ... are incompatible with the conflicts` — you passed more than one policy
backend `--group`. Sync one at a time; see [What to sync for what](#extras-table). The variant
`error: Extra 'lerobot' and group 'molmoact2' are incompatible ...` is the MolmoAct2 inference
server colliding with LeRobot — add `--no-extra lerobot --no-extra convert`.

## Acknowledgements

YAM-ABC-Reproduce stands on excellent open-source work — thanks to their authors and maintainers:

- [i2rt](https://github.com/i2rt-robotics/i2rt) — YAM robot driver, CAN tooling, and hardware interface.
- [ABC](https://github.com/amazon-far/abc)  — the ABC policy backend.
- [openpi](https://github.com/Physical-Intelligence/openpi) — the π<sub>0</sub> / π<sub>0.5</sub> policies and the websocket serving stack.
- [MolmoAct](https://github.com/allenai/MolmoAct) — the MolmoAct2 VLA backend.

- [LeRobot](https://github.com/huggingface/lerobot) — dataset format and utilities.

## License

Apache-2.0. i2rt is a submodule and the policy backends are vendored under `third_party/`, each
under its own license.
