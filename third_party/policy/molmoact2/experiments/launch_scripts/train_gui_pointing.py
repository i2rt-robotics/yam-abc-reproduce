import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(prog="Train a GUI pointing model")
    parser.add_argument("checkpoint", nargs="?", help="Path to checkpoint to start from")
    parser.parse_known_args()
    raise NotImplementedError(
        "GUI pointing training depends on MolmoPoint, which is not included in this MolmoAct2 release."
    )


if __name__ == "__main__":
    main()
