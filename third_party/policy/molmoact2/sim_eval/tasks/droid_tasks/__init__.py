"""DROID bimanual tasks. Importing each module registers its env with ManiSkill."""

from .droid_put_everything_in_box import DroidPutEverythingInBoxEnv

__all__ = [
    "DroidPutEverythingInBoxEnv",
]
