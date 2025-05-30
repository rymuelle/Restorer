import torch
from ConvNeXtUNet.convnextv2_unet import convnextv2unet_atto

model = convnextv2unet_atto()

test_data = torch.rand(1, 3, 32, 32)

output = model(test_data)