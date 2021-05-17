import torch
from torch.nn.modules.loss import _Loss
import torch.nn as nn


def IoU(bboxes_pred, bboxes):
    xmin_pred, ymin_pred, xmax_pred, ymax_pred = bboxes_pred.T
    xmin_true, ymin_true, xmax_true, ymax_true = bboxes.T

    xmin_intersection = torch.stack((xmin_pred, xmin_true)).max(axis=0)[0]
    ymin_intersection = torch.stack((ymin_pred, ymin_true)).max(axis=0)[0]
    xmax_intersection = torch.stack((xmax_pred, xmax_true)).min(axis=0)[0]
    ymax_intersection = torch.stack((ymax_pred, ymax_true)).min(axis=0)[0]

    intersection_area = (xmax_intersection - xmin_intersection) * (ymax_intersection - ymin_intersection)
    intersection_area = intersection_area * (intersection_area > 0)

    true_area = (xmax_true - xmin_true) * (ymax_true - ymin_true)
    pred_area = (xmax_pred - xmin_pred) * (ymax_pred - ymin_pred)
    union_area = true_area + pred_area - intersection_area

    iou = intersection_area / union_area

    return iou


class CombinedLoss(_Loss):
    """Combination of losses for class and bounding boxes predictions.

    Args:
        cls_loss (torch._Loss): loss function for class predictions (reduction should be 'none').
        bboxes_loss (torch._Loss): loss function for bounding boxes predictions
            (reduction should be 'none').
        bboxes_mult (float): multiplier for the bboxes loss.
        reduction (str): specifies the reduction to apply to the output
            'mean' - the sum of the output will be divided by the number of elements in the output;
            others are not implemented yet.
    """

    def __init__(self, cls_loss, bboxes_loss, bboxes_mult=4, reduction="mean"):
        super(CombinedLoss, self).__init__()
        self.bboxes_mult = bboxes_mult
        self.cls_loss = cls_loss
        self.bboxes_loss = bboxes_loss
        self.reduction = reduction

    def forward(self, cls_pred, bboxes_pred, cls_true, bboxes_true):
        """Computes loss.

        Args:
            cls_pred (torch.tensor): class predictions of size Nx1
            bboxes_pred (torch.tensor): bounding boxes predictions of size Nx4
            cls_true (torch.tensor): class ground truth of size Nx1
            bboxes_true (torch.tensor): bounding boxes ground truth of size Nx4
        """
        cls_loss = self.cls_loss(cls_pred, cls_true)
        bboxes_loss = self.bboxes_loss(bboxes_pred, bboxes_true).mean(axis=1).unsqueeze(1)
        loss = cls_loss + bboxes_loss * self.bboxes_mult
        if self.reduction == "mean":
            return loss.mean(), cls_loss.mean(), bboxes_loss.mean()
        else:
            raise Exception(f"Reduction '{self.reduction}' is unknown!")


def get_criterion(version):
    """Returns criterion function by its version.
    If prefix is "dct_*" - criterion for detector;
    if prefix is "clf_*" - criterion for classifier.
    """

    if version == "v1":
        criterion = CombinedLoss(
            cls_loss=nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None,
                                          reduction='none', pos_weight=None),
            bboxes_loss=nn.SmoothL1Loss(size_average=None, reduce=None,
                                        reduction='none', beta=1.0),
            bboxes_mult=75,
            reduction="mean"
        )

    else:
        raise Exception(f"Criterion version '{version}' is unknown!")

    return criterion
