# Deploy

Run a trained policy on the real YAM station. Deployment is two roles, which can live on one
machine or two:

- **Robot host** — the YAM-ABC-Reproduce GUI/CLI. Reads cameras + joint state, sends an obs dict,
  commands the arms. Transport-only (no torch, no jax); identical for every backend.
- **GPU box** — one policy **server per backend** (openpi / MolmoAct2 / ABC), each needing its
  own backend **dependency-group** installed (`--group openpi | molmoact2-train | abc-policy`;
  only one fits a venv at a time — and note the `abc` *extra* is the MCAP writer, not the ABC
  policy). All embodiment-specific transforms (image resize/reorder, normalization, action
  space) live server-side.

Both roles can be — and on a single-GPU station usually are — the same machine: the GUI
launches the server from this same checkout's `.venv` and the client connects over
`127.0.0.1`.

They talk over openpi's websocket protocol — the de-facto VLA-serving wire format, with a
battle-tested client (reconnect + msgpack). openpi serves it natively; MolmoAct/ABC get
small adapters. Swapping policies is just a different `--host/--port`.

```
YAM-ABC-Reproduce robot host                    GPU box — one server at a time
┌──────────────────────────────┐   websocket   ┌────────────────────────────────────────┐
│ yam-abc-deploy / Deploy tab  │   obs dict    │ ONE of (the venv holds one backend):    │
│  cameras + joint state       │ ────────────► │   openpi_server.py   :8000  pi0         │
│  → obs → client.get_action   │ ◄──────────── │                      :8001  pi0.5       │
│  → robot.command_joint_pos   │   {actions}   │   molmoact_server.py :8202              │
│                              │               │   abc_server.py      :8300              │
└──────────────────────────────┘               │  embodiment transforms live HERE        │
                                               └────────────────────────────────────────┘
```

On the station both columns are the same box, and every server runs from this checkout's one
`.venv` — there are no per-backend venvs.

## Deploy tab (GUI)

Two sections — **Server** (start a policy server) and **Client** (run the robot against it).
Same machine: Start Server → wait for `server ready` → Load & Run (host `127.0.0.1`). Remote
GPU box: start the server there, set the host to its IP → Load & Run.

> **Only the Client half is remotable.** **Start Server** always launches on the box running
> the GUI, and the **checkpoint** dropdown lists checkpoints found on *that* box. So for a
> remote GPU box you start the server there by hand (§1) and type its checkpoint path in —
> the dropdown will be showing this machine's paths, not the server's.

**Server**

| Field / control | What it is |
|---|---|
| **policy** | Backend to serve: `pi0`, `pi0.5`, `molmoact2`, `abc`. |
| **checkpoint** | Checkpoint path — pick from the dropdown (scanned per backend) or type one. |
| **config (pi0 / pi0.5)** | openpi TrainConfig name (e.g. `pi0_yam_lora`); editable only for π<sub>0</sub> / π<sub>0.5</sub>. |
| **server badge** | `no server` (grey) → `server loading…` (amber) → `server ready :PORT` (green) → `server error` (red). Load & Run stays disabled until it's ready. |
| **Start Server / Stop Server** | Launch / stop the policy server locally on the port below. |

**Client**

| Field / control | What it is |
|---|---|
| **server host** | Where the server is: `127.0.0.1` (local) or a remote GPU box's IP. |
| **port** | Server port (server binds it, client connects to it). |
| **task instruction** | The prompt sent to the policy. |
| **Load & Run** | Connect and start the rollout. Self-sufficient: from cold it brings the CAN buses up, builds the arms, homes them, then eases into the first action. |
| **Stop** | Stop the rollout. |
| **E-STOP** | (Server header, top-right) Immediate stop of the running rollout. |
| **record rollout** | Log the rollout as episodes (reviewable in the Review tab). |
| **RTC (abc)** | Real-time chunking mode (abc backend). |
| **rollout save path** | Where recorded rollouts are written. |
| **home pose (optional)** | 14 comma-separated joints; ramp to this pose before the policy takes over (keeps the first observation in-distribution). Pre-filled from `deploy_home_pose` in the station YAML — clear the field to skip homing. |
| **clamp rad/s** | Safety cap on arm-joint **speed** in rad/s (truncate + warn), so it is control-rate invariant. `0` disables. Default `1.5`. Typing the old per-step value `0.05` here asks for 0.05 rad/s — 30× slower than intended, and the arm will barely move. |

### Deploy uses the follower buses only

The policy replaces the teaching arm, so a rollout opens **one CAN bus per arm** — `can_left`
and `can_right` on a bimanual station — not the four that teleop needs. Consequences:

- **The GELLO leaders needn't be plugged in or powered** to run a policy.
- **From cold** (straight to the Deploy tab), Load & Run brings the buses up, builds only the
  followers, homes them, then starts the policy. It never engages teleop on the way in, so the
  arms don't ramp toward the leaders first.
- **If teleop was already live**, Load & Run hands the arms over: it stops mirroring and closes
  the two leader buses while the followers stay energized (they never go limp). Teleop cannot
  resume afterwards — **Stop**, then **Reset Session**, then **Start Teleop** to rebuild the
  leaders. Until you do, the Start Teleop and Start Recording buttons are greyed out.
- A **motorized** `yam_lead_*` leader is the exception: it keeps its bus, because closing it
  would drop gravity compensation and the arm would sag. A warning says so in the log.

The same applies to `yam-abc-deploy`.

The sections below cover the same setup from the command line, plus the per-backend server flags.

## The obs/action contract

```python
# client -> server
obs = {
  "images": {<camera_role>: (H, W, 3) uint8 RGB, ...},   # roles: top / left / right / wrist
  "state":  (S,) float32,   # per-arm [joints..., gripper], concatenated in `units` order
  "prompt": str,
  # --rtc / the GUI's "RTC (abc)" checkbox add two more keys:
  "action_prefix": (P, S) float32,   # rows the server must freeze as this chunk's prefix
  "prefix_length": int,              # P
}
# server -> client
{"actions": (H, S) float32}   # H-step chunk, same per-arm layout as `state`
```

The state/action layout comes from the **same `units` list** used at record time
(`yam_abc_reproduce/data/schema.py`), so training data and live inference agree by construction.
For a 2-arm YAM: `[left joints(6), left gripper(1), right joints(6), right gripper(1)]` =
14-D, gripper normalized `[0,1]`.

## 1. GPU box — start the server

Each server runs from the repo's own venv with that backend's dependency-group installed
(`uv sync --all-extras --group openpi | --group molmoact2-train | --group abc-policy` — see
[training.md](training.md)), and needs the lightweight `openpi-client` for the wire protocol.
openpi depends on it directly, so `--group openpi` already covers it. molmoact2 and abc are
separate upstream forks that don't declare it, so add `--extra deploy` or pick it up via
`PYTHONPATH` — the latter is what the GUI does
(`yam_abc_reproduce/gui/builders.py:27-29`).

Trim that sync to `uv sync --extra deploy --group <backend>` only on a box that does nothing
but serve: `uv sync` prunes, so on a box that also runs the GUI it uninstalls the
`gui`/`camera` extras and `yam-abc-gui` stops booting.

Each command below `cd`s into that backend's vendored tree under `third_party/policy/`,
where that backend's own sources live, and puts what it needs on `PYTHONPATH`: a script run
by path gets its *own* directory on `sys.path`, not the working directory, so `cd` alone does
not make `abc_minimal` importable.

**openpi (π<sub>0</sub> / π<sub>0.5</sub>):**
```bash
cd third_party/policy/openpi                     # sources; the venv is the repo's own
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
../../../.venv/bin/python ../../../yam_abc_reproduce/deploy/servers/openpi_server.py \
    --config pi0_yam_lora \
    --checkpoint checkpoints/<config>/<exp>/<step> \
    --prompt "pick up the red block" --port 8000 \
    --flatten-prefix observation/ \
    --image-key-map top=image,left=left_wrist,right=right_wrist \
    --state-key observation/state
# The last three flags map YAM-ABC-Reproduce's contract obs onto the yam config's input keys.
# They are REQUIRED (the GUI passes them too) — without them the first frame is a
# KeyError: 'observation/image'. For pi0.5 use --config pi05_yam_lora.
# If your config uses different image keys, change the --image-key-map right-hand sides.
```

**MolmoAct2:**
```bash
cd third_party/policy/molmoact2                  # sources; served from --group molmoact2-train
PYTHONPATH=$PWD/../openpi/packages/openpi-client/src \
    ../../../.venv/bin/python ../../../yam_abc_reproduce/deploy/servers/molmoact_server.py --port 8202
```

**ABC:**
```bash
cd third_party/policy/abc                        # sources; served from --group abc-policy
PYTHONPATH=$PWD:$PWD/../openpi/packages/openpi-client/src \
    ../../../.venv/bin/python ../../../yam_abc_reproduce/deploy/servers/abc_server.py \
    --checkpoint /path/to/ckpt.pt --prompt "put the bottle in the bin" --port 8300
```

## 2. Robot host — run the rollout

Commands run from the repo root with the venv active
(`cd yam-abc-reproduce && source .venv/bin/activate`). The client side — `websockets`,
`msgpack`, and the `openpi-client` msgpack wire client from the vendored openpi tree — comes
from the `deploy` extra, so it is already installed if that extra was in your `uv sync`.
Otherwise:

```bash
uv sync --extra camera --extra gui --extra deploy   # `uv sync` PRUNES — a bare
                                                   # `--extra deploy` uninstalls the
                                                   # GUI and the camera drivers
```

Sync *before* starting `yam-abc-gui` — a package added while the GUI is already running
isn't importable until you restart it, so a running GUI will fail with
`ModuleNotFoundError: openpi_client`.

**Deploy tab:** in the **Server** section enter the **checkpoint path** (the path on the
*server* machine — for openpi it's `checkpoints/<config>/<exp>/<step>`) and click **Start
Server**. Once the badge reads `server ready :PORT`, set **host**, **port** and **task
instruction** in the **Client** section and click **Load & Run** — the arms ease to a start
pose, then the policy drives them. (There is no single "Start" button; `Load & Run` stays
disabled until the server badge is ready.)

You do **not** need to Start Teleop first: Load & Run brings the follower buses up itself. If
you did start teleop, it releases the leaders on the way in and teleop then needs **Reset
Session** to come back.

molmoact2 may leave the checkpoint blank, which serves the stock
`allenai/MolmoAct2-BimanualYAM`. To serve **your own** molmoact2 finetune, first convert the
OLMo-core checkpoint to HF format with
`third_party/policy/molmoact2/experiments/olmo/hf_model/convert_molmoact2_to_hf.py` and pass
the converted directory — the norm tag is auto-detected from its `norm_stats.json`.

**CLI:**
```bash
yam-abc-deploy --station configs/station_yam.yaml \
    --host <GPU_BOX_IP> --port 8300 --prompt "put the bottle in the bin" --seconds 60
```

Useful flags:

- `--open-loop-horizon N` — rows executed per predicted chunk before re-querying
  (default 15, matches ABC's `execute_chunk_dim`). Smaller = more reactive + more server
  load; larger = smoother + more lag.
- `--rtc` — real-time chunking (async, prefix-conditioned), with `--rtc-prefix-length`,
  `--rtc-action-horizon`, `--rtc-lead-steps`.
- `--ramp-seconds` — ease-in to the first action (default 1.0), so the arm never snaps.
- `--home-pose` — comma/space-separated joint vector to ramp to before the policy takes over
  (`[joints…, gripper]` per arm, in `robots` order — 14 values for a bimanual YAM). Defaults to
  the station's `deploy_home_pose`; pass `--home-pose ""` to skip homing. The CLI twin of the
  GUI's **home pose**, except that the CLI falls back to the station default while the GUI only
  pre-fills the field — so an empty GUI field really means "don't home".
- `--max-joint-speed R` — safety clamp on arm-joint speed in rad/s (default 1.5; truncates
  each step and warns, `<=0` disables). This is the CLI twin of the GUI's **clamp rad/s**. If
  the arm crawls or the log spams `action clamped to ±… rad/step`, this is why.
- `--resize HxW` — shrink images before sending.
- `--mock` — dry-run against a mock robot + cameras.

**The 1–2 min compile happens at server start, not on the first rollout.** openpi and
molmoact2 run a warm-up inference *before* they bind the port, so the badge sits on
`server loading…` (and **Load & Run** stays disabled) for that whole window. ABC does no
warm-up. The loop still logs `first inference: server is JIT-compiling…` on its first query;
a long pause there means the server is still loading, not that the rollout hung.

**Proxies.** You never need to set `no_proxy` for the policy server: both `yam-abc-deploy`
and the GUI client append the target host to `no_proxy`/`NO_PROXY` themselves before
connecting. Without that, `websockets` dials the station's `http_proxy`/`socks5://` proxy even
for `127.0.0.1` and the connection fails as if the server were down. The corollary matters for
a remote GPU box: it must be reachable by a **direct** route, since the client deliberately
bypasses the proxy. (`ImportError: connecting through a SOCKS proxy requires python-socks`
means you are on a build without the bypass.)

## Action space (read this)

The YAM-ABC-Reproduce client commands **absolute joint positions** (`command_joint_pos`, matching
how teleop is recorded). Make sure your policy's output transform emits absolute joint
targets — **not deltas or velocities**. A mismatched action space is the #1 cause of a policy that "runs" but does
nothing sane.

**Prompt must match training.** The deploy `--prompt` has to equal the task the policy
was trained on — the same task name you set when collecting (underscores become spaces),
e.g. `insert the wireless bluetooth earbuds into the charging case`. A different or empty
prompt is out of distribution and the policy will behave erratically.

For ABC specifically: start the arms near the task's demonstrated home pose. Pin it once as
`deploy_home_pose` in the station YAML (top-level, next to `control_hz`) and every rollout
ramps there before the policy runs — read the values off the ARM monitor's `pos` row with the
arms held at the pose your demonstrations start from, or from an episode's first frame.

## Safety

- The loop **ramps** from the current pose to the first action over `--ramp-seconds`.
- GUI **E-STOP** / `Ctrl-C` triggers `estop()` — stop commanding and hard-stop every arm. This
  holds in a deploy-only session too, including after **Stop**, when there is no loop left.
- Start with the arms in a safe pose and a hand near the e-stop for the first rollouts of
  any new checkpoint.
- **Load & Run moves the arms before the policy does.** From cold, an unpinned gripper makes
  i2rt re-measure its travel by driving the jaws into both stops (pin `gripper_limits` in the
  station YAML to skip it — see [hardware.md](hardware.md)); then the arms ramp to the home
  pose. Both happen before the first inference.
- **Reset Session relaxes the arms to zero torque** — they go **limp** and will drop if
  unsupported. It is the only way back to teleop from a deploy session, so support the arms
  before clicking it.
- **If the policy server crashes or drops mid-rollout**, the loop stops commanding and the
  arm **holds at its last commanded pose** (motors stay energized — it is *not* homed or
  released). The GUI shows `stopped · error: …` with that note. To recover: **Reset Session**
  (or **Power Off Arms**) to release the arm — E-STOP *holds* the last pose rather than
  releasing it, and it also discards the in-progress rollout recording and stops the local
  policy-server job. Otherwise **Stop**, then **Start Server** + **Load & Run** again once the
  server is back.
