"""No Git LFS anywhere in this repo -- vendored `.gitattributes` files included.

This repo has no LFS store and no clone of it should need one, so a tracked path that
carries `filter=lfs` is a clone-breaking trap rather than a feature. Vendoring the
molmoact2 snapshot brought in lerobot's `.gitattributes` (LFS filters for `*.mp4`,
`*.safetensors`, `*.stl`, `*.bag`, `tests/artifacts/cameras/*.png`, ...) together with
five 130-byte pointer stubs under `experiments/lerobot/tests/artifacts/cameras/` whose
objects were never pushed here. The result cost a full afternoon of "the repo won't
clone": with git-lfs installed, checkout dies in the smudge filter ("object does not
exist on the server") and `git clone` exits non-zero halfway through; without git-lfs
the clone succeeds and silently leaves text stubs where the binaries should be.

Two independent halves have to stay gone -- the filter declarations and the pointers --
so there is a test for each. Neither touches the network; both are pure checkout reads.
"""

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

LFS_POINTER_MAGIC = b"version https://git-lfs.github.com/spec/v1"
# Pointers are ~130 bytes. The ceiling keeps the scan off the vendored backends' real
# assets (~500 MB of meshes and videos) instead of opening every one of them.
POINTER_SIZE_CEILING = 1024


def tracked_files():
    """Repo-relative paths git tracks. Skips when run from a non-git checkout (sdist)."""
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    listing = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "-z"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [path for path in listing.split("\0") if path]


def test_no_gitattributes_declares_an_lfs_filter():
    """A `filter=lfs` line anywhere is what turns a future vendored blob into a pointer."""
    offenders = []
    for relpath in tracked_files():
        if Path(relpath).name != ".gitattributes":
            continue
        text = (REPO_ROOT / relpath).read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if "filter=lfs" in line and not line.lstrip().startswith("#"):
                offenders.append(f"{relpath}:{lineno}: {line.strip()}")
    assert not offenders, (
        "LFS filters declared in a repo with no LFS store -- replace the "
        "`filter=lfs diff=lfs merge=lfs -text` attributes with `binary`:\n  "
        + "\n  ".join(offenders)
    )


def test_no_tracked_file_is_an_lfs_pointer():
    """Pointer stubs are the visible damage: 130 bytes of text where a binary belongs."""
    offenders = []
    for relpath in tracked_files():
        path = REPO_ROOT / relpath
        # Gitlinks (third_party/i2rt) are directories, and a sparse checkout may be
        # missing a file entirely -- neither can be a pointer.
        if not path.is_file() or path.stat().st_size > POINTER_SIZE_CEILING:
            continue
        with path.open("rb") as handle:
            if handle.read(len(LFS_POINTER_MAGIC)) == LFS_POINTER_MAGIC:
                offenders.append(relpath)
    assert not offenders, (
        "Git LFS pointer stubs committed without their objects -- fetch the real "
        "content from upstream or drop the files:\n  " + "\n  ".join(offenders)
    )
