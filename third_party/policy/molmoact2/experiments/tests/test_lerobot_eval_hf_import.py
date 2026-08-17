from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"


def test_lerobot_eval_import_does_not_require_olmo_for_hf_eval(tmp_path):
    code = """
import importlib.abc
import sys


class BlockOlmo(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "olmo" or fullname.startswith("olmo."):
            raise ModuleNotFoundError("blocked olmo import")
        return None


sys.meta_path.insert(0, BlockOlmo())
from lerobot.scripts.lerobot_eval import main
print(main.__name__)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(LEROBOT_SRC)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "main"
