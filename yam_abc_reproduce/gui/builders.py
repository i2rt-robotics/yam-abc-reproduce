"""Per-backend command builders for Job-backed work (train / deploy).

The Train tab submits one canonical ``params`` dict (gui/train_schema.py has the fields
per backend); here it becomes the *real* launch command for that policy's repo. All
trainers emit ``@metric {"step":..,"loss":..}`` lines (openpi/molmoact2/abc were patched)
so the GUI draws a live loss curve.

The Deploy tab submits a similar dict, and ``build_deploy_command`` launches that
backend's policy SERVER. Every command runs from this repo's own ``.venv`` -- the backends
are dependency-groups of our pyproject.toml, so nothing reaches into a per-submodule venv.
The client runs in-session (see gui/session.py start_deploy).
"""

from __future__ import annotations

import shlex
from pathlib import Path

from . import gpus as _gpus  # aliased: the builders' own `gpus` is the requested card count

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OPENPI = _REPO_ROOT / "third_party" / "policy" / "openpi"
_MOLMOACT = _REPO_ROOT / "third_party" / "policy" / "molmoact2"
_ABC = _REPO_ROOT / "third_party" / "policy" / "abc"
_LEROBOT_HOME = _REPO_ROOT / "data" / "lerobot"      # openpi/molmoact2 resolve repo_id here
_ABC_CACHE = _REPO_ROOT / "data" / "abc_cache"       # abc reads its prepared cache here
_SERVERS = _REPO_ROOT / "yam_abc_reproduce" / "deploy" / "servers"   # per-backend server scripts
# openpi-client (msgpack wire) ships in the `deploy` extra, which a GPU box running only
# --group molmoact2-train / abc-policy need not have; pull it in via PYTHONPATH instead.
_OPENPI_CLIENT = _OPENPI / "packages" / "openpi-client" / "src"
_DEPLOY_PORT = {"pi0": 8000, "pi05": 8001, "molmoact2": 8202, "abc": 8300}

# Everything launches from this repo's venv -- no per-submodule venv paths. Each backend's
# stack is a dependency-group of our own pyproject.toml, installed editable straight out of
# third_party/, so `uv sync --all-extras --group <backend>` is the only setup step.
_VENV = _REPO_ROOT / ".venv"
_PY = _VENV / "bin" / "python"
_TORCHRUN = _VENV / "bin" / "torchrun"
# Make openpi's gs:// downloads (pi0/pi05 base params, PaliGemma tokenizer) honour a proxy:
# they go through gcsfs, whose aiohttp session ignores http_proxy without trust_env=True,
# and so hang instead of erroring. FSSPEC_<PROTOCOL> is fsspec's way in from outside the
# code -- but only this single-underscore form, which it JSON-parses into the GCSFileSystem
# kwargs. FSSPEC_GS_SESSION_KWARGS arrives as a raw string and dies in ClientSession(**str).
_FSSPEC_GS_TRUST_ENV = """FSSPEC_GS='{"session_kwargs": {"trust_env": true}}'"""
# The group that installs each backend, and a distribution only that group brings in --
# checked in site-packages to preflight the venv really holds the stack, since the backends
# are mutually exclusive and a sync for one silently uninstalls another.
_BACKEND_GROUP = {"pi0": "openpi", "pi05": "openpi", "molmoact2": "molmoact2-train", "abc": "abc-policy"}
_BACKEND_MARKER = {"pi0": "openpi-", "pi05": "openpi-", "molmoact2": "ai2_molmo-", "abc": "warp_lang-"}


def _require_backend_venv(backend: str) -> None:
    marker = _BACKEND_MARKER.get(backend)
    if marker is None:
        return
    if not any(_VENV.glob(f"lib/python*/site-packages/{marker}*.dist-info")):
        cmd = f"uv sync --all-extras --group {_BACKEND_GROUP[backend]}"
        raise ValueError(
            f"{backend} is not installed (no {marker}* in {_VENV}). Install it with:  {cmd}  "
            f"— keep `--all-extras` on that line, because `uv sync` uninstalls everything you "
            f"leave out, this GUI's own extras included. The policy backends are mutually "
            f"exclusive, so this also replaces any other backend in the venv."
        )


def _bash(script: str) -> tuple[list[str], str]:
    return ["bash", "-lc", script], script


def _int(v, d):
    try:
        return int(v)
    except (TypeError, ValueError):
        return d


# --- pi0 / pi0.5 (openpi, JAX) -----------------------------------------------
# Both run through the same openpi trainer; only the config name differs
# (pi0_yam* vs pi05_yam*), so they share one builder body.
def _train_openpi(p: dict, cfg: str) -> tuple[list[str], str]:
    ds = p.get("dataset", "put_the_bottle_into_the_bin")
    # Preflight that the LeRobot dataset exists under data/lerobot (where the GUI points
    # HF_LEROBOT_HOME), reporting what's available if not.
    if ds and not (_LEROBOT_HOME / ds).exists():
        have = sorted(d.name for d in _LEROBOT_HOME.glob("*") if d.is_dir()) if _LEROBOT_HOME.exists() else []
        raise ValueError(
            f"dataset {ds!r} not found under {_LEROBOT_HOME} — run yam-abc-convert first. "
            f"Available: {have or '(none)'}"
        )
    run = p.get("run_name", "yam_pp")
    gpus = _int(p.get("gpus"), 1)
    bpg = _int(p.get("batch_per_gpu"), 16)
    steps = _int(p.get("steps"), 30000)
    resume = " --resume" if p.get("resume") else ""
    # openpi rejects --resume together with --overwrite (TrainConfig.__post_init__), and
    # without either one it aborts on an existing exp-name dir (initialize_checkpoint_dir),
    # so exactly one of them has to be passed.
    overwrite = "" if resume else " --overwrite"
    sc = (p.get("start_checkpoint") or "").strip()
    base = f" --weight-loader.params-path={shlex.quote(sc)}" if sc else ""
    # Arm count (2 = bimanual 14-D default, 1 = single-arm 7-D): must match between the
    # norm-stats and train steps, so pass it to both.
    arms = _int(p.get("arms"), 2)
    # Call our venv's python outright rather than `uv run`: inside the submodule `uv run`
    # would resolve openpi's own project and build third_party/policy/openpi/.venv, which is
    # exactly the per-submodule env this repo no longer keeps.
    # PATH does still need the venv's bin, for one thing we cannot address by absolute path:
    # openpi's maybe_download() locates gsutil (the `openpi` group installs it) with
    # shutil.which(), and the job only inherits whatever PATH launched the GUI.
    # CUDA_VISIBLE_DEVICES has to live in this shared export rather than on the train.py line
    # below: compute_norm_stats.py initialises JAX too, and would otherwise still open every
    # card on the box. Pinned to `gpus` cards, the mesh becomes (1, fsdp_devices) -- pure FSDP
    # over exactly what was asked for, instead of (device_count, 1) over the whole machine.
    devices = _gpus.pick(gpus)
    script = (
        f"cd {_OPENPI} && export CUDA_VISIBLE_DEVICES={devices} {_gpus.DEVICE_ORDER} "
        f"HF_LEROBOT_HOME={_LEROBOT_HOME} HF_HUB_OFFLINE=1 "
        f"{_FSSPEC_GS_TRUST_ENV} PATH={_VENV}/bin:$PATH && "
        f"echo '[yam-abc] pinned to GPU(s) {devices}; computing norm stats for {ds} "
        f"(a few min; tqdm may not stream)...' && "
        f"{_PY} scripts/compute_norm_stats.py --config-name {cfg} --repo-id {ds} --num-arms {arms} && "
        f"echo '[yam-abc] launching train.py — JAX compile + weight load first; first loss line can take 5-15 min...' && "
        f"XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 {_PY} scripts/train.py {cfg} "
        f"--exp-name={run} --data.repo-id={ds} --data.num-arms={arms} "
        f"--num-train-steps={steps} --batch-size={bpg * gpus} --fsdp-devices={gpus} "
        f"--model.action-horizon={_int(p.get('chunk_size'), 50)} "
        f"--lr-schedule.peak-lr={p.get('lr', '2.5e-5')} "
        f"--log-interval={_int(p.get('log_freq'), 100)} --save-interval={_int(p.get('save_freq'), 5000)} "
        f"--seed={_int(p.get('seed'), 0)}{resume}{base} --no-wandb-enabled{overwrite}"
    )
    return _bash(script)


def _train_pi0(p: dict) -> tuple[list[str], str]:
    return _train_openpi(p, "pi0_yam_lora" if p.get("train_mode") == "lora" else "pi0_yam")


def _train_pi05(p: dict) -> tuple[list[str], str]:
    return _train_openpi(p, "pi05_yam_lora" if p.get("train_mode") == "lora" else "pi05_yam")


# --- molmoact2 (PyTorch, torchrun) ------------------------------------------
def _train_molmoact2(p: dict) -> tuple[list[str], str]:
    ds = p.get("dataset", "put_the_bottle_into_the_bin")
    run = p.get("run_name", "yam_pp")
    gpus = _int(p.get("gpus"), 1)
    bpg = _int(p.get("batch_per_gpu"), 2)
    ckpt = p.get("start_checkpoint") or "allenai/MolmoAct2-BimanualYAM"
    mode = p.get("train_mode", "fft")
    # train_mode -> ft flags
    ft_vlm = "false" if mode == "action_expert_only" else "true"
    ft_embed = "none" if mode == "action_expert_only" else "lm_head"
    lora = " --lora_enable=true" if mode == "lora" else ""
    lrs = (
        f"--llm_learning_rate={p.get('lr_llm', '1e-5')} "
        f"--vit_learning_rate={p.get('lr_vit', '5e-6')} "
        f"--connector_learning_rate={p.get('lr_connector', '5e-6')} "
        f"--action_expert_learning_rate={p.get('lr_action_expert', '5e-5')}"
    )
    resume = " --resume" if p.get("resume") else ""
    exp = _MOLMOACT / "experiments"       # training sources live here (olmo)
    # Environment learned from a verified single-GPU smoke run on the RTX 5090:
    #  - needs a torch with sm_120 kernels, i.e. CUDA >= 12.8: `uv sync --extra
    #    molmoact2-train` (torch 2.11.0, CUDA 13 runtime), NOT the molmoact2 submodule
    #    root's cu121 server env (the `molmoact2` extra), which has no sm_120
    #  - LD_LIBRARY_PATH must include the venv's nvidia/*/lib so torchcodec finds libnppicc
    #  - MOLMO_DATA_DIR is read unconditionally by olmo's data loader
    #  - WANDB_* satisfy omegaconf ${oc.env:...} interpolation even in disabled mode
    #  - expandable_segments lets the 4B model + batch 1 fit in 32 GB (peak ~31 GB)
    #  - CUDA_VISIBLE_DEVICES confines the run to `gpus` cards; torchrun's LOCAL_RANK indexes
    #    into the *visible* set, so cuda:0 is whichever card comes first there
    # A generic single-dataset mixture ("yam_abc_yam") reads the repo_id + action
    # horizon from env, so the GUI drives the dataset without editing data_mixtures.py.
    devices = _gpus.pick(gpus)
    script = (
        f"cd {exp} && "
        f"export CUDA_VISIBLE_DEVICES={devices} {_gpus.DEVICE_ORDER} "
        f"HF_LEROBOT_HOME={_LEROBOT_HOME} MOLMO_DATA_DIR={_REPO_ROOT}/data/molmo_data "
        f"YAM_ABC_LEROBOT_REPO={ds} YAM_ABC_ACTION_HORIZON={_int(p.get('chunk_size'), 30)} "
        f"WANDB_MODE=disabled WANDB_API_KEY= WANDB_PROJECT=yam_abc_reproduce WANDB_ENTITY=yam_abc_reproduce "
        f"PYTHONPATH={exp} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
        f"export LD_LIBRARY_PATH=\"$(ls -d {_VENV}/lib/python*/site-packages/nvidia/*/lib | tr '\\n' ':')$LD_LIBRARY_PATH\" && "
        f"mkdir -p {_REPO_ROOT}/data/molmo_data && "
        f"echo '[yam-abc] pinned to GPU(s) {devices}' && "
        f"{_TORCHRUN} --standalone --nproc-per-node={gpus} "
        f"launch_scripts/train_lerobot.py {ckpt} yam_abc_yam "
        f"--wandb.name={run} --wandb.project=yam_abc_reproduce --wandb.entity=yam_abc_reproduce "
        f"--save_folder=checkpoints/finetune/{run} "
        f"--max_duration={_int(p.get('steps'), 50000)} "
        f"--device_batch_size={bpg} --global_batch_size={bpg * gpus} "
        f"--save_interval={_int(p.get('save_freq'), 10000)} "
        f"--num_workers=4 --pin_memory=true "
        f"--packing=false --dynamic_seq_len=true "
        f"--ft_vlm={ft_vlm} --ft_action_expert=true --ft_embedding={ft_embed}{lora} "
        f"--seed={_int(p.get('seed'), 0)}{resume} {lrs}"
    )
    return _bash(script)


# --- abc (ABC-DiT, torchrun) -------------------------------------------------
# The same three paths train.py's validate_train_config requires for the yam_abc preset.
_ABC_CACHE_REQUIRED = ("norm_stats.json", "train_real", "val_real")


def _require_abc_cache() -> None:
    """Refuse an abc launch when the prepared cache is missing.

    train.py checks this itself, but only inside validate_train_config -- i.e. after torchrun
    has already spun up every rank, so the real error arrives buried under a ChildFailedError.
    The trap it usually catches: `yam-abc-convert --to abc` writes a FLAT
    data/abc/<repo-id>/episode_NNNNNN.mcap, while abc's export_mcap.py globs
    <root>/<task>/episode_*/episode.mcap -- aim it at the flat dir and it reports "0 episodes"
    and leaves the cache empty.
    """
    missing = [str(_ABC_CACHE / n) for n in _ABC_CACHE_REQUIRED if not (_ABC_CACHE / n).exists()]
    if not missing:
        return
    raise ValueError(
        "abc has no prepared training cache -- missing: " + ", ".join(missing) + ". abc never "
        "reads the raw episodes, so build the cache first (needs system ffmpeg + ffprobe). "
        "<task> is not cosmetic: it becomes the CLIP prompt (underscores -> spaces), so name it "
        "as the instruction you will deploy with.  "
        "1) lay the episodes out as <root>/{train,val}/<task>/episode_NNNNNN/episode.mcap  "
        f"2) cd {_ABC} && {_PY} export_mcap.py <root>/train {_ABC_CACHE}/train_real 8  "
        f"(then <root>/val -> {_ABC_CACHE}/val_real)  "
        f"3) cd {_REPO_ROOT} && {_PY} scripts/compute_abc_norm_stats.py "
        f"--cache {_ABC_CACHE} --train-dir train_real "
        "-- recompute these, the pre-baked stats are tied to upstream's own data.  "
        "Full recipe: docs/training.md, 'abc (ABC-DiT)'."
    )


def _train_abc(p: dict) -> tuple[list[str], str]:
    run = p.get("run_name", "yam_pp")
    gpus = _int(p.get("gpus"), 1)
    # tyro bool flags are --lora/--no-lora and --load-pretrained (not =true).
    lora = " --lora" if p.get("train_mode") == "lora" else " --no-lora"
    # Fine-tune from a pretrained DiT if a checkpoint is given: --pretrained-ckpt
    # sets the path (default would be cache/abc_dit_xl_200k_model.pt). Anchor a
    # relative path to the repo root (the command cd's into the abc repo first).
    sc = p.get("start_checkpoint") or ""
    if sc and not Path(sc).is_absolute() and (_REPO_ROOT / sc).exists():
        sc = str((_REPO_ROOT / sc).resolve())
    pretrained = f" --load-pretrained --pretrained-ckpt {shlex.quote(sc)}" if sc else ""
    # abc reads its prepared cache (yam-abc-convert --to abc -> episode_*.mcap, then
    # abc's export_mcap.py -> $ABC_CACHE/{train,val}_real + norm_stats.json). The
    # "yam_abc" mixture preset (config.py) points at those real-only dirs; without it
    # the default "bottles" preset also requires train_sim/val_sim and validate fails.
    # PYTHONUNBUFFERED=1 so train_loop's "step N loss ..." lines reach the GUI live.
    # Runs the venv's torchrun directly (torch==2.11.0+cu128 supports the RTX 5090 sm_120).
    # CUDA_VISIBLE_DEVICES confines the run to `gpus` cards (LOCAL_RANK indexes the visible
    # set); --batch-size is per-GPU here, so it is deliberately not scaled by the count.
    devices = _gpus.pick(gpus)
    script = (
        f"cd {_ABC} && export CUDA_VISIBLE_DEVICES={devices} {_gpus.DEVICE_ORDER} "
        f"ABC_CACHE={_ABC_CACHE} PYTHONUNBUFFERED=1 && "
        f"echo '[yam-abc] pinned to GPU(s) {devices}' && "
        f"{_TORCHRUN} --standalone --nproc-per-node={gpus} train.py "
        f"--cache-root={_ABC_CACHE} --mixture-preset=yam_abc "
        f"--batch-size={_int(p.get('batch_per_gpu'), 90)} --train-steps={_int(p.get('steps'), 75000)} "
        f"--ckpt-every={_int(p.get('save_freq'), 5000)} --log-every={_int(p.get('log_freq'), 50)} "
        f"--optim.learning-rate={p.get('lr', '1e-4')} --model.chunk-length={_int(p.get('chunk_size'), 30)} "
        f"--seed={_int(p.get('seed'), 123)}{lora}{pretrained} --run-name={run}"
    )
    return _bash(script)


_TRAIN = {"pi0": _train_pi0, "pi05": _train_pi05, "molmoact2": _train_molmoact2, "abc": _train_abc}


def build_train_command(params: dict) -> tuple[list[str], str]:
    backend = params.get("backend", "pi0")
    fn = _TRAIN.get(backend)
    if fn is None:
        raise ValueError(f"unknown train backend {backend!r}")
    _require_backend_venv(backend)
    if backend == "abc":
        _require_abc_cache()
    return fn(params)


# --- deploy: launch the policy SERVER from this repo's venv ------------------
# The venv holds whichever backend group was synced (jax for openpi, an sm_120-capable
# torch for molmoact2/abc); each server still cd's into its submodule for its sources.
# Non-default flags/envs below were verified end-to-end on the RTX 5090 (see
# yam_abc_reproduce/deploy). The client runs in-session, not here.
# Every server is pinned to one card, the freest at launch. There is no GPUs field on the
# Deploy tab and a server only ever needs one, but without the pin JAX opens all of them and
# XLA_PYTHON_CLIENT_MEM_FRACTION reserves 90% of each -- which is what usually blocks the next
# training launch. The servers' own --device flags (cuda:0 / cuda) then name the visible card.
def _deploy_openpi(ckpt, prompt, port, p, default_cfg):
    cfg = p.get("config") or default_cfg
    # {pi0,pi05}_yam's YamInputs reads observation/image|left_wrist|right_wrist +
    # observation/state; create_trained_policy skips the config repack at inference, so the
    # server must map the contract's camera roles + state onto those keys.
    ikm = p.get("image_key_map", "top=image,left=left_wrist,right=right_wrist")
    dev = _gpus.pick(1)
    return (
        f"cd {_OPENPI} && echo '[yam-abc] pinned to GPU(s) {dev}' && "
        f"CUDA_VISIBLE_DEVICES={dev} {_gpus.DEVICE_ORDER} "
        f"XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 "
        f"{_PY} {_SERVERS}/openpi_server.py "
        f"--config {shlex.quote(cfg)} --checkpoint {shlex.quote(ckpt)} "
        f"--port {port} --prompt {shlex.quote(prompt)} "
        f"--flatten-prefix observation/ --image-key-map {shlex.quote(ikm)} "
        f"--state-key observation/state"
    )


def _train_deploy_pi0(ckpt, prompt, port, p):
    return _deploy_openpi(ckpt, prompt, port, p, "pi0_yam_lora")


def _train_deploy_pi05(ckpt, prompt, port, p):
    return _deploy_openpi(ckpt, prompt, port, p, "pi05_yam_lora")


def _train_deploy_molmoact2(ckpt, prompt, port, p):
    # Served from the molmoact2 *training* group (molmoact2-train): its torch has sm_120
    # kernels, the submodule root's torch 2.5.1+cu121 does not. ckpt = a local HF-format dir
    # (OLMo-core finetune converted via convert_molmoact2_to_hf). norm-tag auto-detected
    # from norm_stats.json.
    ck = f"--checkpoint {shlex.quote(ckpt)} " if ckpt else ""
    dev = _gpus.pick(1)
    return (
        f"cd {_MOLMOACT} && echo '[yam-abc] pinned to GPU(s) {dev}' && "
        f"CUDA_VISIBLE_DEVICES={dev} {_gpus.DEVICE_ORDER} "
        f"PYTHONPATH={_OPENPI_CLIENT} "
        f"{_PY} {_SERVERS}/molmoact_server.py "
        f"{ck}--molmoact-repo {_MOLMOACT} --port {port} --device cuda:0 --dtype bfloat16"
    )


def _train_deploy_abc(ckpt, prompt, port, p):
    # abc's torch 2.11+cu128 env. A LoRA checkpoint is auto-merged by the
    # server before load. ABC_CACHE holds norm_stats/DINOv3/CLIP.
    # ABC_DEBUG_DUMP: server writes the first live obs (per-camera PNGs + state.npy)
    # here so it can be diffed against a known-good reference to catch an obs shift.
    dump = _REPO_ROOT / "data" / "abc_debug"
    dev = _gpus.pick(1)
    return (
        f"cd {_ABC} && echo '[yam-abc] pinned to GPU(s) {dev}' && "
        f"CUDA_VISIBLE_DEVICES={dev} {_gpus.DEVICE_ORDER} "
        f"ABC_CACHE={_ABC_CACHE} PYTHONPATH={_ABC}:{_OPENPI_CLIENT} "
        f"ABC_DEBUG_DUMP={dump} "
        f"{_PY} {_SERVERS}/abc_server.py "
        f"--checkpoint {shlex.quote(ckpt)} --prompt {shlex.quote(prompt)} "
        f"--port {port} --device cuda"
    )


_DEPLOY = {"pi0": _train_deploy_pi0, "pi05": _train_deploy_pi05,
           "molmoact2": _train_deploy_molmoact2, "abc": _train_deploy_abc}


def build_deploy_command(params: dict) -> tuple[list[str], str]:
    """Launch ``backend``'s policy SERVER from this repo's venv, same machine. For a
    remote GPU box, run the same script there and point the client at its host:port."""
    backend = params.get("backend", "pi0")
    fn = _DEPLOY.get(backend)
    if fn is None:
        raise ValueError(f"unknown deploy backend {backend!r}")
    _require_backend_venv(backend)
    ckpt = params.get("checkpoint", "")
    # The command cd's into the backend repo before launching, so a relative
    # checkpoint (e.g. "data/molmoact2_hf/...") would resolve against the submodule,
    # not the YAM-ABC-Reproduce root. Anchor relative paths to the repo root; leave absolute
    # paths and HF repo ids (no slash / not an existing path) untouched.
    if ckpt and not Path(ckpt).is_absolute() and (_REPO_ROOT / ckpt).exists():
        ckpt = str((_REPO_ROOT / ckpt).resolve())
    prompt = params.get("prompt", "")
    port = _int(params.get("port"), _DEPLOY_PORT.get(backend, 8000))
    return _bash(fn(ckpt, prompt, port, params))


# --- convert: run yam-abc-convert as a job so its progress streams to the GUI --------
def build_convert_command(params: dict) -> tuple[list[str], str]:
    """Run yam-abc-convert (episodes -> LeRobot/ABC dataset) as a job, so the GUI streams
    its progress like any other — no terminal needed."""
    task = (params.get("task") or params.get("dataset") or "").strip()
    src = params.get("src") or (f"data/episodes/{task}" if task else "")
    if not src:
        raise ValueError("convert needs a task (data/episodes/<task>) or an explicit src path")
    fmt = params.get("to", "lerobot")
    repo_id = (params.get("repo_id") or task or "yam_abc_reproduce/dataset").strip()
    script = (
        f"cd {_REPO_ROOT} && {_VENV}/bin/yam-abc-convert "
        f"{shlex.quote(src)} --to {shlex.quote(fmt)} --repo-id {shlex.quote(repo_id)}"
    )
    return _bash(script)


def build_command(kind: str, params: dict) -> tuple[list[str], str]:
    if kind == "train":
        return build_train_command(params)
    if kind == "deploy":
        return build_deploy_command(params)
    if kind == "convert":
        return build_convert_command(params)
    raise ValueError(f"unknown job kind {kind!r}")
