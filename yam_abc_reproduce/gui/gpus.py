"""Which cards a GPU job gets.

The Train tab's ``GPUs`` field used to only *size* a run -- it fed ``--fsdp-devices`` /
``--nproc-per-node`` and the global batch, while every card on the box stayed visible. openpi
then built its mesh as ``(jax.device_count() // fsdp_devices, fsdp_devices)``, so "GPUs: 1" on
an 8-card host still meant 8-way data parallelism across everyone else's GPUs.

So the launch commands pin themselves: ``pick(n)`` returns the ``CUDA_VISIBLE_DEVICES`` value
that confines a job to the ``n`` cards with the most free VRAM right now. Two rules:

- An inherited ``CUDA_VISIBLE_DEVICES`` (the operator exports one before starting the GUI --
  see the README) says *which* cards are on offer. It is a ceiling, never replaced.
- ``nvidia-smi`` is advisory. A robot-host-only station has none, so a missing or broken
  ``nvidia-smi`` falls back to the first ``n`` on offer rather than failing the launch.

Callers export DEVICE_ORDER alongside the picked value; see its comment for why.
"""

from __future__ import annotations

import os
import subprocess

# nvidia-smi enumerates by PCI bus order, but CUDA's default CUDA_DEVICE_ORDER is
# FASTEST_FIRST -- so an index read out of nvidia-smi is not guaranteed to name the same card
# to CUDA. Pinning by index is only sound with the two orders forced to agree. Identical cards
# tie-break by bus order anyway, so this is a no-op on a homogeneous box.
DEVICE_ORDER = "CUDA_DEVICE_ORDER=PCI_BUS_ID"


def _offered() -> list[str]:
    """Entries of a CUDA_VISIBLE_DEVICES inherited from whoever started the GUI.

    Stripped, because ``CUDA_VISIBLE_DEVICES="0, 1"`` is legal and an unstripped copy would
    word-split inside the launch script's ``export``. Set-but-empty means *no* card is
    visible, which is not something to train on -- treat it as unset.
    """
    return [d.strip() for d in (os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",") if d.strip()]


def inventory() -> list[tuple[str, int | None]]:
    """``[(device, free MiB)]`` for the cards this GUI may use, in enumeration order.

    A free-VRAM reading of ``None`` means nvidia-smi did not price that card -- it prints
    ``[N/A]`` / ``[Unknown Error]`` for one it cannot read, and a MIG instance never shows up
    in the index/uuid columns at all. That is deliberately distinct from ``0``, which means
    the card is genuinely full: the admission guard must not refuse a launch over a reading
    it never got. Empty when nvidia-smi is absent or told us nothing at all.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3).stdout
    except Exception:
        return []
    # Keyed by both spellings, since CUDA_VISIBLE_DEVICES may legally hold GPU-<uuid> rather
    # than an index. One card nvidia-smi cannot price must not cost us the other seven, so a
    # bad row drops on its own instead of aborting the read.
    free: dict[str, int] = {}
    order: list[str] = []
    for line in out.strip().splitlines():
        row = [c.strip() for c in line.split(",")]
        if len(row) != 3:
            continue
        idx, uuid, mib = row
        order.append(idx)
        try:
            free[idx] = free[uuid] = int(mib)
        except ValueError:
            continue
    # An entry we could not price stays in the list unpriced -- the operator offered it, so
    # it stays on offer; pick() just ranks it below anything with a real reading.
    return [(d, free.get(d)) for d in (_offered() or order)]


def pick(n: int) -> str:
    """The ``CUDA_VISIBLE_DEVICES`` value confining a job to the ``n`` freest cards.

    Rendered in enumeration order (``0,3``, not ``3,0``) so the command the GUI shows the
    operator reads naturally. Entries are stripped, so the result never needs shell quoting.
    """
    if n < 1:
        raise ValueError(
            f"GPUs={n}: a job needs at least 1 GPU -- an empty CUDA_VISIBLE_DEVICES would hide "
            f"every card. Set the Train tab's GPUs field to 1."
        )
    cards = inventory()
    order = [d for d, _ in cards] or _offered() or [str(i) for i in range(n)]
    if n > len(order):
        raise ValueError(
            f"GPUs={n} but only {len(order)} GPU(s) are available to this GUI "
            f"(CUDA_VISIBLE_DEVICES={','.join(order)}). Set GPUs to {len(order)} or less, or "
            f"restart yam-abc-gui with a wider CUDA_VISIBLE_DEVICES -- jobs inherit it at "
            f"launch, so changing it now has no effect on the running GUI."
        )
    if cards:
        # sorted() is stable, so an idle box (every card equally free) keeps enumeration
        # order and "GPUs: 1" deterministically picks the first card. An unpriced card ranks
        # below every priced one rather than being dropped.
        freest = {d for d, _ in sorted(cards, key=lambda c: -c[1] if c[1] is not None else 1)[:n]}
        order = [d for d in order if d in freest]
    return ",".join(order[:n])


def most_free_mib() -> int | None:
    """Free VRAM on the emptiest card this GUI may use, or None if nothing could be read.

    The admission guard's question is "can *any* card take this job", not "is card 0 free" --
    pick() will put the job on whichever one this reports. None means "no reading", and the
    guard skips rather than refusing; cards we could not price are left out of the max
    instead of counting as full.
    """
    return max((f for _, f in inventory() if f is not None), default=None)
