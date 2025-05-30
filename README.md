# ConvNeXt-V2-UNet
A U-Net architecture built with a modified ConvNeXt-V2 block, designed for enhanced flexibility by allowing different input and output channel configurations. This library currently focuses on standard U-Net applications; sparse versions suitable for autoencoding tasks are not yet implemented. 

![denoising_example](figures/example_usage.pdf)

## Installation instructions

You can install this package locally using pip.

1. Clone the Repository:
First, get a copy of the project:
Bash

``` bash
git clone https://github.com/your-username/ConvNeXt-V2-UNet.git
cd ConvNeXt-V2-UNet
```

2. Install with Pip:
For Development (Editable Mode):
``` bash
pip install -e .
```

For Standard Local Installation:
``` bash
pip install .
```
## Example:

```examples/simple_denoising.ipynb``` contains a simple demonstration of the unet for denoising. 

## Licensing

This project is licensed under the MIT free use license. 


## Acknowledgments

This code is adapted from work by Meta:

Woo, S., Debnath, S., Hu, R., Chen, X., Liu, Z., Kweon, I. S., & Xie, S. (2023). ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders. arXiv preprint arXiv:2301.00808.

https://github.com/facebookresearch/ConvNeXt-V2