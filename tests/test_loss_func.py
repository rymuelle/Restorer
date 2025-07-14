import torch
from Restorer.CombinedPerceptualLoss import CombinedPerceptualLoss


def test_loss():
    loss = CombinedPerceptualLoss(
        l1=1,
        mse=1,
        ssim=1,
        vgg=1,
        style=1,
        frequency=1,
        chroma=1,
        luma=1,
        luma_mse=1,
        lpips=1,
        sharpness_loss=1,
    )
    torch.manual_seed(0)
    x = torch.rand(1, 3, 256, 256)
    y = torch.rand(1, 3, 256, 256)

    assert loss(x, y).detach().numpy() == 175.96115
    print("Loss test passed")
