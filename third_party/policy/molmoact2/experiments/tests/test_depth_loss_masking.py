import torch

from olmo.train.trainer import Trainer


def test_apply_depth_loss_masking_is_always_noop():
    trainer = object.__new__(Trainer)
    input_ids = torch.tensor([[5, 11, 21, 22, 23, 12, 99]], dtype=torch.long)
    flat_loss_masks = torch.ones(input_ids.numel(), dtype=torch.float32)
    flat_labels = torch.arange(input_ids.numel(), dtype=torch.long)

    trainer._apply_depth_loss_masking(
        batch={
            "depth_updated_mask": torch.tensor([[True, False, True]], dtype=torch.bool),
        },
        input_ids=input_ids,
        subsegment_ids=None,
        flat_loss_masks=flat_loss_masks,
        flat_labels=flat_labels,
    )

    assert torch.equal(flat_loss_masks, torch.ones_like(flat_loss_masks))
    assert torch.equal(flat_labels, torch.arange(input_ids.numel(), dtype=torch.long))
