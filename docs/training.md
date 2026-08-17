# Convert & train

Turn recorded demonstrations into a trained policy. Collect and review episodes first —
see [collect.md](collect.md). Assumes the station/venv are set up ([hardware.md](hardware.md));
run everything from the repo root with the venv active.

## 1. Convert to a training dataset (Train tab → Convert)

The backend decides the format:

| Backend | Format | Command |
|---|---|---|
| π<sub>0</sub> / π<sub>0.5</sub>, `molmoact2` | LeRobot | `yam-abc-convert data/episodes/<task> --to lerobot --repo-id <name>` |
| `abc` | ABC release MCAP | `yam-abc-convert data/episodes/<task> --to abc --repo-id <name>` |

The **Convert** panel runs this for you — no terminal needed:

| Field | What it is |
|---|---|
| **task** | Folder under `data/episodes` to convert. |
| **format** | `lerobot` (π<sub>0</sub> / π<sub>0.5</sub> / molmoact2) or `abc` (ABC-DiT). |
| **repo-id** | Output dataset name (defaults to the task name). |
| **Convert** | Runs `yam-abc-convert`. |

Sanity-check a converted dataset in the **Review** tab. `yam-abc-viz` is meant to open
LeRobot's own Rerun viewer for one episode:

```bash
yam-abc-viz --repo-id <name> --root data/lerobot/<name> --episode-index 0
```

but it is **currently broken** with the locked LeRobot: it shells out to
`lerobot.scripts.visualize_dataset`, a module lerobot >=0.5 no longer ships (it is now
`lerobot_dataset_viz`), so the command exits with `No module named
lerobot.scripts.visualize_dataset`. Use the Review tab until that is fixed. Note also that
converted datasets land under `data/lerobot/` — there is no `./lerobot_ds`.

## 2. Train (Train tab → Train policy)

**One-time: install the backend.** A fresh YAM-ABC-Reproduce venv has no policy backend — no
jax at all, and only whatever torch `lerobot` happens to pull in, which the backend sync then
*replaces* with its own pinned build. So add the backend you'll use. Each is a
**dependency-group** of the repo's own `pyproject.toml`, installed from the vendored tree under
`third_party/policy/` — there is nothing to sync inside `third_party/`:

```bash
uv sync --all-extras --group openpi
uv sync --all-extras --group molmoact2-train
uv sync --all-extras --group abc-policy
```

Two rules make those command lines look the way they do:

- **`uv sync` prunes.** It makes the venv match *exactly* what you list, so anything left off
  is uninstalled. Sync a backend on its own and you lose `fastapi`/`uvicorn` and the GUI stops
  booting — `--all-extras` is the one flag that keeps every extra, but the `--group` still has
  to be repeated each time. (Narrowing pays off only on a box that is not the station. To
  *serve* and nothing else: `uv sync --extra deploy --group <backend>` — the `deploy` extra
  supplies `websockets` + `openpi-client` for the molmoact2/abc servers, while `--group openpi`
  already ships `openpi-client` itself and the GUI instead puts it on `PYTHONPATH`. To *train*
  on a headless GPU box you still want `--extra gui --extra convert --group <backend>`, since
  the Train tab is the launcher.)
- **One backend at a time.** They pin incompatible torch/jax builds and are declared as
  conflicts in `pyproject.toml`, so uv rejects two `--group`s and each line above *replaces*
  the previously installed backend. Modelling them as groups rather than extras is what keeps
  `uv sync --all-extras` meaningful: it installs every feature and no backend.

The GUI launches every trainer and policy server from this one venv (`gui/builders.py`), so
only one backend can be trained or served at a time. Switching backends is a re-sync, not a
reinstall from scratch: `uv sync --all-extras --group <other-backend>` replaces the installed
backend in place and keeps every extra. (Serving two backends *simultaneously* is what needs a
second machine or a second checkout.) If the backend isn't installed you get a
`... is not installed` error naming the exact command to run, instead of an ImportError
several minutes into training.

Two consequences of sharing one venv:

- `rerun-sdk` is de-pinned to `>=0.32.2` in `[tool.uv] override-dependencies`, so i2rt can
  coexist with the `lerobot==0.5.1` openpi pins. This affects only lerobot's own Rerun-based
  dataset viewer, not training or serving.
- openpi depends on `opencv-python` while `camera`/`gui` use `opencv-python-headless`. Both
  ship the same `cv2` module, so `--group openpi` together with either extra puts two providers
  of one import in the venv and install order decides which wins. If `import cv2` starts
  failing after an openpi sync, reinstall the provider you want to win
  (`uv sync --all-extras --group openpi --reinstall-package opencv-python-headless`). On a box
  that needs neither `camera` nor `gui`, the clash cannot arise in the first place.

Pick the **policy** backend, fill the fields, then **Launch Train** — the loss curve and
logs stream live, with a status pill, progress + step counter, and the launched command
shown above the plot. Everything runs from this repo's single `.venv` — whichever backend
group is currently synced.

**Fields common to every backend:**

| Field | Meaning |
|---|---|
| dataset / task | The LeRobot repo-id to train on (for `abc` this field is vestigial — it uses the prepared cache, see below). |
| run name | Names the output run/checkpoint directory. |
| batch / GPU | Per-GPU batch size. |
| GPUs | Number of GPUs **in this machine** — single-node only (`torchrun --standalone` for molmoact2/abc, JAX FSDP `--fsdp-devices` for π<sub>0</sub>/π<sub>0.5</sub>; there is no multi-node path). The job is confined to exactly this many cards via `CUDA_VISIBLE_DEVICES`, picked by free VRAM at launch; asking for more than are available is refused before the job starts. **Leave at `1`** — see [One GPU at a time](#one-gpu-at-a-time). `batch / GPU` × `GPUs` is the global batch. |
| steps | Total training steps. |
| save every | Checkpoint interval (steps). |

**Backend-specific fields:**

| Field | Backends | Meaning |
|---|---|---|
| train mode | all | Default first: `fft` / `lora` (π<sub>0</sub> / π<sub>0.5</sub>), `lora` / `fft` (abc), `action_expert_only` / `lora` / `fft` (molmoact2). |
| action chunk / horizon | all | Action-sequence length the policy predicts per step. |
| arms (1 or 2) | π<sub>0</sub> / π<sub>0.5</sub> | 2 = bimanual 14-D, 1 = single-arm 7-D. |
| log every | π<sub>0</sub> / π<sub>0.5</sub> / abc | Logging interval (steps). |
| lr (peak) / lr | π<sub>0</sub> / π<sub>0.5</sub> / abc | Peak learning rate. |
| lr llm / vit / connector / action_expert | molmoact2 | Per-component learning rates. |
| seed | all | RNG seed. |
| base / pretrained ckpt (opt) | all | Starting weights. Blank = the backend default for π<sub>0</sub> / π<sub>0.5</sub> (the `gs://openpi-assets` base) and molmoact2 (`allenai/MolmoAct2-BimanualYAM`) — but for **abc** blank means **train from random init**, so set a path (e.g. the ABC DiT checkpoint) to fine-tune. |
| resume ckpt (opt) | π<sub>0</sub> / π<sub>0.5</sub> / molmoact2 | Resume from an existing checkpoint. |

**Per-backend defaults** (tuned for a single 32 GB RTX 5090):

| Field | π<sub>0</sub> / π<sub>0.5</sub> | molmoact2 | abc |
|---|---|---|---|
| batch / GPU | 8 | 1 | 16 |
| GPUs | 1 | 1 | 1 |
| steps | 30000 | 50000 | 75000 |
| action chunk | 50 | 30 | 30 |
| train mode | fft | action_expert_only | lora |
| lr | 2.5e-5 | per-component | 1e-4 |

Checkpoints are written under each backend's output dir (shown in the logs).

<a id="one-gpu-at-a-time"></a>
### One GPU at a time

This pipeline is only validated on a **single-GPU station** — one box, one NVIDIA card (a
32 GB RTX 5090), doing collect, review, convert, train and serve. Three consequences:

- **`GPUs` stays `1`.** Multi-GPU is wired up (`--fsdp-devices` for π<sub>0</sub>/π<sub>0.5</sub>,
  `torchrun --nproc-per-node` for molmoact2/abc) but has never been exercised here, and it is
  single-node regardless — nothing sets `MASTER_ADDR`/`rdzv`, so "more GPUs" always means more
  cards in *this* machine. The defaults above are what fits one card: molmoact2 only trains
  comfortably in `action_expert_only` (frozen VLM, ~8 GB peak); `lora` needs ~30.7 GB and OOMs
  with no headroom, and `fft` needs more still. A value above `1` now genuinely hands the job
  that many cards — it used to only resize the batch and the mesh while every card stayed open
  — so treat it as an unvalidated path, not an inert one.
- **The card is picked for you; `CUDA_VISIBLE_DEVICES` says which ones are on offer.** Each
  job exports `CUDA_VISIBLE_DEVICES` for the `GPUs` cards with the most free VRAM at launch
  (plus `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so those indices mean the same thing to CUDA as they
  do to `nvidia-smi`), and echoes `[yam-abc] pinned to GPU(s) …` as its first log line. Export
  `CUDA_VISIBLE_DEVICES=2,5` *before* starting `yam-abc-gui` to restrict it to a subset — jobs
  inherit the GUI's environment, so setting it afterwards has no effect on a running GUI, and
  that list is a ceiling the picker never reaches outside of. With `2,5` set, `GPUs: 1` runs
  on whichever of the two is freer and `GPUs: 4` is refused with
  `only 2 GPU(s) are available to this GUI`. Policy servers have no `GPUs` field and always
  take one card, the freest. Note that `nvidia-smi` ignores the variable, so the **GPU** badge
  keeps reporting the first physical card even when your job is on another one.
- **One GPU job per card.** Both **Launch Train** and **Start Server** are refused with
  `no GPU has 8000 MiB free (most free: <N> MiB)` when *no* card has **8000 MiB free**. Since
  the openpi trainer *and* the openpi server each reserve ~90% of the card they are pinned to
  (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`), on a single-GPU station a policy server left running
  still blocks the next training launch: press **Stop Server** first, or click the **GPU**
  badge and kill the squatting pid. One caveat on a multi-GPU box — two jobs started within a
  few seconds of each other can both pick the same card, because the first has not allocated
  yet. Wait for `pinned to GPU(s) …` and the memory to show up in `nvidia-smi` before starting
  the second.

### π<sub>0</sub> / π<sub>0.5</sub> (openpi)

Trains against openpi's `scripts/train.py` on a LeRobot dataset. The dataset feature keys must
match the config's Input transform. π<sub>0</sub> and π<sub>0.5</sub> run through the same
stack with identical knobs — pick the backend in the GUI (`pi0` / `pi0.5`), which maps to the
openpi configs `pi0_yam` / `pi0_yam_lora` and `pi05_yam` / `pi05_yam_lora` respectively.

`--repo-id` is simply **which dataset** — it's used for the convert output and passed to
training as `--data.repo-id`. You do **not** need to keep it consistent for normalization:
the YAM config pins a fixed `asset_id` (`yam`), so `compute_norm_stats` writes and the server
loads norm-stats under `assets/yam` regardless of the dataset name.

### molmoact2

Needs the `molmoact2-train` group — its torch carries sm_120 kernels (CUDA ≥ 12.8), unlike
the `molmoact2` group, which mirrors the molmoact2 tree root's cu121 server env and so cannot run
on an RTX 5090 at all. On a single 32 GB card only
`action_expert_only` (frozen VLM, ~8 GB peak) fits comfortably; `lora`/`fft` need more
VRAM. `base checkpoint` is a local HF-format dir (or `allenai/MolmoAct2-BimanualYAM`).

### abc (ABC-DiT)

ABC does **not** read the raw dataset at train time — it reads a prepared cache at
`data/abc_cache/`. The GUI's `task`/`dataset` field is **vestigial**: the launcher hard-codes
`--cache-root=data/abc_cache --mixture-preset=yam_abc`, which reads `{train_real,val_real}/` +
`norm_stats.json`. (`yam_abc` is the real preset name — the only other accepted values are the
legacy alias `freemani` and upstream's `bottles`, which also demands sim data.)

So the cache is the one prerequisite the Train form cannot show you — which is why the tab
carries a note about it, and why **Launch Train** is refused outright when it is missing,
naming the three paths and the steps below. Run `train.py` by hand without it and you get the
unfiltered version instead, from `validate_train_config`, after `torchrun` has already started
every rank and buried under a `ChildFailedError`:

```
ValueError: Invalid training config:
  - missing required paths: data/abc_cache/norm_stats.json, data/abc_cache/train_real, data/abc_cache/val_real
```

Build that cache before training. Step 2 re-encodes every camera stream, so it needs system
**ffmpeg + ffprobe** (`sudo apt install ffmpeg`):

```bash
# 1. Lay the episodes out as  <root>/{train,val}/<task>/episode_NNNNNN/episode.mcap .
#    For externally-collected raw YAM episodes (episode_*.npy.mp4/ dirs holding
#    left.mcap + right.mcap + per-camera mp4), this repo ships the converter:
python scripts/yam_episodes_to_abc_mcap.py --src <raw_dir> --out <root> \
    --task put_the_bottle_into_the_bin --val 2
#    <task> is NOT cosmetic: export_mcap.py records it as the episode's task_name and the
#    trainer turns it into the CLIP prompt (underscores -> spaces), so name it exactly as the
#    instruction you will pass at deploy time.
#    Episodes recorded by THIS repo go through `yam-abc-convert --to abc`, which writes a flat
#    data/abc/<repo-id>/episode_NNNNNN.mcap — re-lay it out as above, or export_mcap.py's
#    <task>/episode_*/episode.mcap glob matches nothing and reports 0 episodes.

# 2. Build the training cache. Run from the vendored abc tree because the script path below is
#    relative; any sync carrying the `abc` extra will do (--group abc-policy is only needed
#    for train.py itself):
cd third_party/policy/abc
../../../.venv/bin/python export_mcap.py <root>/train  ../../../data/abc_cache/train_real 4
../../../.venv/bin/python export_mcap.py <root>/val    ../../../data/abc_cache/val_real   2

# 3. Compute normalization stats FROM THIS DATASET. ABC ships a pre-baked norm_stats.json
#    tied to its own data; reuse it on yours and any dimension it held constant (std -> 0)
#    blows the normalized values — and the loss — up to millions:
cd ../../..
python scripts/compute_abc_norm_stats.py --cache data/abc_cache --train-dir train_real

# 4. Only when training from scratch (blank `pretrained ckpt`): put the DINOv3 ViT-B/16
#    weights in the cache as data/abc_cache/dinov3_vitb16_pretrain_lvd1689m.pth (mind Meta's
#    licence). Without the file the run prints "using random DINOv3" and trains an untrained
#    vision backbone. Fine-tuning from a DiT checkpoint takes its vision weights from that
#    checkpoint instead, and skips this.
```
