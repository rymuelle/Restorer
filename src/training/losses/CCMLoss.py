import torch
import torch.nn as nn

class CCMLoss(nn.Module):
    def __init__(self, criterion=nn.L1Loss()):
        super().__init__()
        self.criterion = criterion

    def forward(self, pred, gt, ccm):
        pred = torch.einsum('b c o, b c h w -> b o h w', ccm, pred)
        gt = torch.einsum('b c o, b c h w -> b o h w', ccm, gt)
        return self.criterion(pred, gt)