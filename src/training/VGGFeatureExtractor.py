import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_msssim
import math

def custom_weight_init(m):
    if isinstance(m, nn.Conv2d):
        k = m.kernel_size[0]  # assuming square kernels
        c_in = m.in_channels
        n_l = (k ** 2) * c_in
        std = math.sqrt(2.0 / n_l)
        nn.init.normal_(m.weight, mean=0.0, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Linear):
        n_l = m.in_features
        std = math.sqrt(2.0 / n_l)
        nn.init.normal_(m.weight, mean=0.0, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class VGGFeatureExtractor(nn.Module):
    """
    VGG-like network for perceptual loss with runtime hot-swappable activations.
    Returns features from selected layers.
    """
    def __init__(self, config=None, feature_layers=None, activation=None):
        super().__init__()
        if config is None:
            config = [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)]
        
        if feature_layers is None:
            feature_layers = [3, 8, 15, 22, 29]

        if activation is None:
            activation = lambda: nn.ReLU(inplace=False)
        
        layers = []
        in_channels = 3
        for num_convs, out_channels in config:
            for _ in range(num_convs):
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
                layers.append(activation())
                in_channels = out_channels
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.features = nn.Sequential(*layers)
        self.feature_layers = feature_layers
        self.activation_factory = activation  # store for later swapping
        self.apply(custom_weight_init)

    def set_activation(self, activation_cls, **kwargs):
        """
        Replace all activation layers with new activation.
        activation_cls: activation class (e.g., nn.GELU, nn.LeakyReLU)
        kwargs: any keyword args for the activation class
        """
        for i, layer in enumerate(self.features):
            if isinstance(layer, nn.ReLU) or \
               isinstance(layer, nn.LeakyReLU) or \
               isinstance(layer, nn.GELU) or \
               isinstance(layer, nn.SiLU) or \
               isinstance(layer, nn.Identity) or \
               isinstance(layer, nn.Tanh):
                self.features[i] = activation_cls(**kwargs)
        # Update the stored factory for future reference
        self.activation_factory = lambda: activation_cls(**kwargs)

    def forward(self, x):
        outputs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.feature_layers:
                outputs.append(x)
        return outputs




