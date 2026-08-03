"""The GUI builds shell command lines for the CLIs by flag name, so a renamed or retyped
flag breaks a launch that no other test exercises: test_gui_smoke asserts on the command
*strings*, never feeding them to the parser they target. These tests close that loop by
parsing each generated argv with the very dataclass the CLI parses it with.
"""

import importlib.util
import shlex
import sys
from pathlib import Path

import pytest
import tyro

SERVERS = Path(__file__).resolve().parents[1] / "yam_abc_reproduce" / "deploy" / "servers"

# Deploy backend -> the server module its command line launches.
BACKEND_SERVER = {"pi0": "openpi_server", "pi05": "openpi_server",
                  "molmoact2": "molmoact_server", "abc": "abc_server"}


def _load_server(name: str):
    """Import a server module by path. They are scripts, not package members (each one
    sys.path-inserts its own directory for _wire), and each needs its backend's deps at
    import time -- so skip when the group that provides them is not synced here."""
    spec = importlib.util.spec_from_file_location(name, SERVERS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves a field's annotations through sys.modules[cls.__module__], so the
    # module has to be registered before its ServerArgs can be introspected.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except ImportError as e:
        pytest.skip(f"{name} needs deps this venv lacks: {e}")
    return module


def _server_argv(script: str, module_name: str) -> list[str]:
    """The flags the generated command passes to the server, i.e. everything after its path."""
    tokens = shlex.split(script)
    for i, tok in enumerate(tokens):
        if tok.endswith(f"{module_name}.py"):
            return tokens[i + 1:]
    raise AssertionError(f"{module_name}.py not found in: {script}")


@pytest.fixture
def builders(monkeypatch):
    """builders with the two things that would otherwise need real hardware stubbed out."""
    from yam_abc_reproduce.gui import builders, gpus

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(gpus, "inventory", list)
    monkeypatch.setattr(builders, "_require_backend_venv", lambda backend: None)
    return builders


@pytest.mark.parametrize("backend", sorted(BACKEND_SERVER))
def test_deploy_command_parses_against_its_server_cli(builders, backend, tmp_path):
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("")
    _, script = builders.build_deploy_command(
        {"backend": backend, "checkpoint": str(ckpt), "prompt": "pick up the red block",
         "port": 8123})

    module_name = BACKEND_SERVER[backend]
    argv = _server_argv(script, module_name)
    args = tyro.cli(_load_server(module_name).ServerArgs, args=argv, console_outputs=False)

    assert args.port == 8123
    assert getattr(args, "checkpoint", str(ckpt)) == str(ckpt)


def test_convert_command_parses_against_the_convert_cli(builders):
    from yam_abc_reproduce.cli import ConvertArgs

    _, script = builders.build_convert_command({"task": "pick_and_place", "to": "abc"})
    argv = shlex.split(script)
    argv = argv[argv.index("yam-abc-convert" if "yam-abc-convert" in argv else
                           next(t for t in argv if t.endswith("yam-abc-convert"))) + 1:]

    args = tyro.cli(ConvertArgs, args=argv, console_outputs=False)
    assert args.src == "data/episodes/pick_and_place"
    assert args.to == "abc"
    assert args.repo_id == "pick_and_place"
