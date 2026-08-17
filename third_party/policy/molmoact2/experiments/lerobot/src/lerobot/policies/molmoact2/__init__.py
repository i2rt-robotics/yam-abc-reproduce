from .configuration_molmoact2 import MolmoAct2Config
from .processor_molmoact2 import make_molmoact2_pre_post_processors

__all__ = ["MolmoAct2Config", "MolmoAct2Policy", "make_molmoact2_pre_post_processors"]


def __getattr__(name: str):
    if name == "MolmoAct2Policy":
        from .modeling_molmoact2 import MolmoAct2Policy

        return MolmoAct2Policy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
