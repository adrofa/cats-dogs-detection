from torch.nn.modules.loss import _Loss
import torch.nn as nn


class WeightedBCEWithLogitsLoss(_Loss):
    """BCEWithLogitsLoss for a class and bounding boxes predictions.

    Args:
        bboxes_share (float): between 0 and 1, share of the bboxes loss in the total loss.
        reduction (str): specifies the reduction to apply to the output
            'mean' - the sum of the output will be divided by the number of elements in the output;
            others are not implemented yet.
    """

    def __init__(self, bboxes_share=0.5, reduction="mean"):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.bboxes_share = bboxes_share
        self.cls_share = 1 - self.bboxes_share
        self.loss = nn.BCEWithLogitsLoss(weight=None, size_average=None,
                                         reduce=None, reduction="none", pos_weight=None)
        self.reduction = reduction

    def forward(self, cls_pred, bboxes_pred, cls_true, bboxes_true):
        cls_loss = self.loss(cls_pred, cls_true)
        bboxes_loss = self.loss(bboxes_pred, bboxes_true)
        loss = cls_loss * self.cls_share + bboxes_loss.mean(axis=1).unsqueeze(1) * self.bboxes_share
        if self.reduction == "mean":
            return loss.mean()
        else:
            raise Exception(f"Reduction '{self.reduction}' is unknown!")


def get_criterion(version):
    """Returns criterion function by its version."""

    if version == "v1":
        return WeightedBCEWithLogitsLoss(bboxes_share=0.5, reduction="mean")

    else:
        raise Exception(f"Criterion version '{version}' is unknown!")
