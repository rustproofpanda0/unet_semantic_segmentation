import torch


class DiceLoss(torch.nn.Module):
    def __init__(self):
           super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        return 1. - ((torch.sum(2 * pred * target) + 1e-6) / (torch.sum(pred) + torch.sum(target) + 1e-6))
