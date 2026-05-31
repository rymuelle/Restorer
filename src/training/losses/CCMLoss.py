import torch
import torch.nn as nn



class CCMLoss(nn.Module):
    def __init__(self, criterion=nn.L1Loss(), gamma = 0, lpips=False, apply_ccm=True):
        super().__init__()
        self.criterion = criterion
        self.gamma = gamma
        self.lpips_scale = lpips
        if self.lpips_scale > 0:
            import lpips
            self.lpips = lpips.LPIPS(net='vgg')
        self.apply_ccm = apply_ccm

    def forward(self, pred, gt, ccm):
        if self.apply_ccm:
            pred = torch.einsum('b c o, b c h w -> b o h w', ccm, pred)
            gt = torch.einsum('b c o, b c h w -> b o h w', ccm, gt)
        if self.gamma > 0:
            gt = gt.clip(0, 1) ** (1 / self.gamma)
            pred = pred.clip(0, 1) ** (1 / self.gamma)
        loss = self.criterion(pred, gt)
        if self.lpips_scale > 0:
            loss += self.lpips_scale * self.lpips(pred, gt).mean()
        return loss 
    

