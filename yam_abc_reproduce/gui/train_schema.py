"""Declarative Train-tab field schema, per policy backend.

The GUI fetches the fields for the selected backend (`GET /api/train/fields`)
and renders them flat; ``builders.py`` maps the submitted params to each
backend's real launch command. Keep this minimal — only knobs changed per run.

Each field: name, label, type (text|number|select), default, [options], [help].
"""

from __future__ import annotations

BACKENDS = ["pi0", "pi05", "molmoact2", "abc"]


def _f(name, label, ftype="text", default="", options=None, help=""):
    d = {"name": name, "label": label, "type": ftype, "default": default}
    if options:
        d["options"] = options
    if help:
        d["help"] = help
    return d


# Shared fields reused across backends (same meaning; per-backend flag mapping
# lives in builders.py).
def _common(dataset_label, dataset_default, batch_default, gpus_default, steps_default, save_default):
    return [
        _f("dataset", dataset_label, "text", dataset_default),
        _f("run_name", "run name", "text", "yam_test"),
        _f("batch_per_gpu", "batch / GPU", "number", batch_default),
        _f("gpus", "GPUs", "number", gpus_default),
        _f("steps", "steps", "number", steps_default),
        _f("save_freq", "save every", "number", save_default),
    ]


FIELDS: dict[str, list[dict]] = {
    "pi0": [
        *_common("dataset (lerobot repo-id)", "", 8, 1, 30000, 5000),
        _f("train_mode", "train mode", "select", "fft", ["fft", "lora"]),
        _f("log_freq", "log every", "number", 100),
        _f("chunk_size", "action chunk", "number", 50),
        _f("arms", "arms (1 or 2)", "number", 2),  # 2 = bimanual 14-D, 1 = single-arm 7-D
        _f("lr", "lr (peak)", "text", "2.5e-5"),
        _f("seed", "seed", "number", 0),
        # Base weights: blank = the config default (gs://openpi-assets pi0_base); set a
        # local path (e.g. NFS mirror) to override -> --weight-loader.params-path.
        _f("start_checkpoint", "base ckpt (opt)", "text", ""),
        _f("resume", "resume ckpt (opt)", "text", ""),
    ],
    # pi0.5: same openpi stack as pi0 (config pi05_yam / pi05_yam_lora); identical knobs.
    "pi05": [
        *_common("dataset (lerobot repo-id)", "", 8, 1, 30000, 5000),
        _f("train_mode", "train mode", "select", "fft", ["fft", "lora"]),
        _f("log_freq", "log every", "number", 100),
        _f("chunk_size", "action chunk", "number", 50),
        _f("arms", "arms (1 or 2)", "number", 2),  # 2 = bimanual 14-D, 1 = single-arm 7-D
        _f("lr", "lr (peak)", "text", "2.5e-5"),
        _f("seed", "seed", "number", 0),
        _f("start_checkpoint", "base ckpt (opt)", "text", ""),
        _f("resume", "resume ckpt (opt)", "text", ""),
    ],
    "molmoact2": [
        # batch 1 / 1 GPU: the 4B model + activations peak ~31 GB, fits a single 32 GB
        # RTX 5090 only at device_batch_size=1 (needs expandable_segments, set in builders).
        *_common("dataset (lerobot repo-id)", "", 1, 1, 50000, 10000),
        # local snapshot (avoids a 21 GB re-download); repo id "allenai/MolmoAct2-BimanualYAM" also works.
        _f("start_checkpoint", "base checkpoint", "text", ""),
        # On a single 32 GB card (~30.7 GB free after the desktop) only action_expert_only
        # (frozen VLM, ~8 GB peak) fits comfortably; lora needs ~30.7 GB and OOMs with zero
        # headroom, fft needs even more. Use lora/fft only with more VRAM (bigger/multi-GPU).
        _f("train_mode", "train mode", "select", "action_expert_only", ["action_expert_only", "lora", "fft"]),
        _f("chunk_size", "action horizon", "number", 30),
        # molmoact2 has four component learning rates (see experiments/README).
        _f("lr_llm", "lr llm", "text", "1e-5"),
        _f("lr_vit", "lr vit", "text", "5e-6"),
        _f("lr_connector", "lr connector", "text", "5e-6"),
        _f("lr_action_expert", "lr action_expert", "text", "5e-5"),
        _f("seed", "seed", "number", 0),
        _f("resume", "resume ckpt (opt)", "text", ""),
    ],
    "abc": [
        # single 5090: batch 16 / 1 GPU (verified). Upstream's 90/8 is their cluster default.
        # NOTE: the "task" field is vestigial — the abc builder hardcodes
        # --mixture-preset=yam_abc (config.py), so this value is not used. What abc actually
        # trains on is the prepared cache named in NOTES below.
        *_common("task", "", 16, 1, 75000, 5000),
        _f("start_checkpoint", "pretrained ckpt (opt)", "text", ""),
        _f("train_mode", "train mode", "select", "lora", ["lora", "fft"]),
        _f("log_freq", "log every", "number", 50),
        _f("chunk_size", "action chunk", "number", 30),
        _f("lr", "lr", "text", "1e-4"),
        _f("seed", "seed", "number", 0),
    ],
}


# Per-backend prerequisite, rendered under the Train form as `note` (same key the Deploy tab
# uses). Only abc has one: it is the single backend that does not train off the dataset the
# form names, so without this the form looks complete and the launch dies inside torchrun.
NOTES: dict[str, str] = {
    "abc": (
        "abc trains off a prepared cache at data/abc_cache/, not the raw episodes — the task "
        "field above is ignored. Build it once before launching (needs ffmpeg + ffprobe): "
        "1) lay the episodes out as <root>/{train,val}/<task>/episode_NNNNNN/episode.mcap — "
        "<task> becomes the CLIP prompt, so name it as the instruction you will deploy with; "
        "2) third_party/policy/abc/export_mcap.py <root>/train data/abc_cache/train_real 8 "
        "(then <root>/val → val_real); "
        "3) scripts/compute_abc_norm_stats.py --cache data/abc_cache --train-dir train_real. "
        "See docs/training.md, “abc (ABC-DiT)”."
    ),
}


def fields_for(backend: str) -> list[dict]:
    return FIELDS.get(backend, FIELDS["pi0"])


def note_for(backend: str) -> str:
    return NOTES.get(backend, "")
