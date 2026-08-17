# Vendored policy backends

Snapshots vendored from the `yam-abc` branches of the private staging forks
(now archived under github.com/Seagull-Y after serving their purpose) — plain
directories, not submodules, per the "current version works; re-vendor on a
major upstream release" maintenance policy. Nested submodules of the
upstreams (molmoact2's top-level lerobot/YAM/EVA_DROID, openpi's
aloha/libero) are examples/reference code the pipeline never imports and are
intentionally not included (molmoact2 training uses its own in-tree
`experiments/lerobot`).

| backend | vendored commit | upstream |
|---|---|---|
| openpi | 5f3fbb3f2429bc6b1342b1c6502a74113eb28747 | Physical-Intelligence/openpi |
| molmoact2 | 78b323996d805bf581cbc8359a795cdf758fb304 | allenai/MolmoAct |
| abc | 282e153055b422d9753400d765b48085cc842746 | amazon-far/abc |

Also intentionally excluded, all non-functional here — but by three different
mechanisms, which matters when re-vendoring:

- IDE settings (`.vscode/`, `.idea/`) — genuinely ignored by the upstreams' own
  `.gitignore`, so a plain copy already omits them.
- Demo videos (`abc/assets/*.mp4`) — dropped by *this* repo's root `.gitignore`
  (`*.mp4`, unanchored, so it reaches into every vendored tree), not by abc's.
- lerobot camera fixtures (`experiments/lerobot/tests/artifacts/cameras/`) —
  nothing ignores these anywhere: upstream lerobot explicitly un-ignores them
  (`!tests/artifacts` in its `.gitignore`) and tracks them **in Git LFS**. A
  plain copy therefore *does* bring them, as pointers — see the next section.
  The rest of `tests/artifacts/` (three `save_*_to_safetensors.py` generators)
  is small, plain text, and still vendored.

## No Git LFS in a vendored snapshot

The first vendoring pass copied those five camera fixtures from a checkout whose
LFS content had never been fetched, so what landed here were the ~130-byte Git
LFS *pointers* — together with lerobot's `.gitattributes` LFS filters. This repo
has no LFS store, so the pointers named objects no server holds: cloning with
git-lfs installed failed in the smudge filter, and cloning without it produced
text stubs where a PNG and a RealSense bag belonged. The pointers are deleted
(their only consumers are vendored lerobot's own `tests/cameras/test_opencv.py`
and `test_realsense.py`, which this repo never runs) and that `.gitattributes`
now marks the same patterns `binary` instead of `filter=lfs ...` -- the one
local edit inside a vendored tree, kept because re-vendoring must not
reintroduce the trap.

So: when re-vendoring, copy from a checkout whose LFS content is either fully
fetched or fully excluded, never a mix — and remember that `git status` in the
source checkout looks clean either way, because an unfetched pointer *is* the
committed content there. `tests/test_no_git_lfs.py` fails the suite if either
half comes back.
