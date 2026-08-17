import os
import tempfile
from os.path import join

from olmo.util import select_checkpoint


def select_or_none(checkpoint, prefer_unsharded=False):
    try:
        return select_checkpoint(checkpoint, prefer_unsharded)
    except FileNotFoundError:
        return None


def test_select_checkpoint():
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            assert not select_checkpoint(tmp_dir)
        except FileNotFoundError:
            pass

        os.mkdir(join(tmp_dir, "131"))
        os.mkdir(join(tmp_dir, "unsharded-thing"))
        try:
            assert not select_checkpoint(tmp_dir)
        except FileNotFoundError:
            pass

        os.mkdir(join(tmp_dir, "step2"))
        assert select_checkpoint(tmp_dir) == join(tmp_dir, "step2")

        with open(join(tmp_dir, "step2", "model.pt"), "w") as f:
            f.write("")
        assert select_checkpoint(join(tmp_dir, "step2")) == join(tmp_dir, "step2")

        os.mkdir(join(tmp_dir, "step10"))
        assert select_checkpoint(tmp_dir) == join(tmp_dir, "step10")

        os.mkdir(join(tmp_dir, "step9-unsharded"))
        assert select_checkpoint(tmp_dir) == join(tmp_dir, "step10")

        os.mkdir(join(tmp_dir, "step11-unsharded"))
        assert select_checkpoint(tmp_dir) == join(tmp_dir, "step11-unsharded")

        os.mkdir(join(tmp_dir, "step11"))
        assert select_checkpoint(tmp_dir) == join(tmp_dir, "step11")
        assert select_checkpoint(tmp_dir, prefer_unsharded=True) == join(tmp_dir, "step11-unsharded")
