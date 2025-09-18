import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as Fun


class FFTLoss(nn.Module):
    def __init__(self, loss_weight=0.1, reduction='mean'):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        pred_fft = torch.fft.rfft2(pred, norm='backward')
        target_fft = torch.fft.rfft2(target, norm='backward')
        loss = self.loss_weight * Fun.l1_loss(pred_fft, target_fft, reduction=self.reduction)
        return loss


class CharbonnierLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12) -> None:
        super(CharbonnierLoss, self).__init__()

        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}')

        self.__loss_weight = loss_weight
        self.__reduction = reduction
        self.__eps = eps

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        :param pred:
        :param target:
        :return:
        """
        loss = self.__loss_weight * torch.sqrt((pred - target) ** 2 + self.__eps).mean()
        return loss


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * Fun.l1_loss(
            pred, target, weight, reduction=self.reduction)
