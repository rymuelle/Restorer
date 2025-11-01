import pandas as pd
import os
from torch.utils.data import Dataset
import imageio
from colour_demosaicing import (
    mosaicing_CFA_Bayer
)
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from RawHandler.RawHandler import RawHandler
from src.training.utils import cfa_to_sparse

def random_crop_dim(shape, crop_size, buffer, validation=False):
        """
        Calculates random (or centered) crop dimensions, ensuring even coordinates.
        """
        h, w = shape
        
        # Ensure crop size is even
        crop_size = (crop_size // 2) * 2
        
        if not validation:
            top = np.random.randint(0 + buffer, h - crop_size - buffer)
            left = np.random.randint(0 + buffer, w - crop_size - buffer)
        else:
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2

        # Ensure top-left corner is even for correct Bayer pattern alignment
        if top % 2 != 0: 
            top -= 1
        if left % 2 != 0: 
            left -= 1
            
        # Handle potential boundary issues after adjustment
        top = max(0, top)
        left = max(0, left)
        if top + crop_size > h:
            top = h - crop_size
        if left + crop_size > w:
            left = w - crop_size
            
        bottom = top + crop_size
        right = left + crop_size
        return (left, right, top, bottom)

# def cfa_to_sparse(cfa_image, pattern='RGGB'):
#     """
#     Converts a 2D CFA image (H, W) to a 3D sparse RGB image (H, W, 3).
#     """
#     H, W = cfa_image.shape
#     sparse_rgb = np.zeros((H, W, 3), dtype=cfa_image.dtype)
    
#     # 
#     if pattern == 'RGGB':
#         # R
#         sparse_rgb[0::2, 0::2, 0] = cfa_image[0::2, 0::2]
#         # G (top-right)
#         sparse_rgb[0::2, 1::2, 1] = cfa_image[0::2, 1::2]
#         # G (bottom-left)
#         sparse_rgb[1::2, 0::2, 1] = cfa_image[1::2, 0::2]
#         # B
#         sparse_rgb[1::2, 1::2, 2] = cfa_image[1::2, 1::2]
#     elif pattern == 'GRBG':
#         sparse_rgb[0::2, 0::2, 1] = cfa_image[0::2, 0::2]
#         sparse_rgb[0::2, 1::2, 0] = cfa_image[0::2, 1::2]
#         sparse_rgb[1::2, 0::2, 2] = cfa_image[1::2, 0::2]
#         sparse_rgb[1::2, 1::2, 1] = cfa_image[1::2, 1::2]
#     elif pattern == 'GBRG':
#         sparse_rgb[0::2, 0::2, 1] = cfa_image[0::2, 0::2]
#         sparse_rgb[0::2, 1::2, 2] = cfa_image[0::2, 1::2]
#         sparse_rgb[1::2, 0::2, 0] = cfa_image[1::2, 0::2]
#         sparse_rgb[1::2, 1::2, 1] = cfa_image[1::2, 1::2]
#     elif pattern == 'BGGR':
#         sparse_rgb[0::2, 0::2, 2] = cfa_image[0::2, 0::2]
#         sparse_rgb[0::2, 1::2, 1] = cfa_image[0::2, 1::2]
#         sparse_rgb[1::2, 0::2, 1] = cfa_image[1::2, 0::2]
#         sparse_rgb[1::2, 1::2, 0] = cfa_image[1::2, 1::2]
#     else:
#         raise NotImplementedError(f"Pattern {pattern} not implemented")
        
#     return sparse_rgb

def cfa_to_rggb_stack(cfa_image, pattern='RGGB'):
    """
    Converts a (H, W) CFA image to an (4, H/2, W/2) RGGB stack.
    """
    assert cfa_image.ndim == 2, "Input must be (H, W)"
    H, W = cfa_image.shape
    assert H % 2 == 0 and W % 2 == 0, "Height and width must be even"
    
    if pattern == 'RGGB':
        R = cfa_image[0::2, 0::2]
        G1 = cfa_image[0::2, 1::2] # G at top-right
        G2 = cfa_image[1::2, 0::2] # G at bottom-left
        B = cfa_image[1::2, 1::2]
    elif pattern == 'GRBG':
        G1 = cfa_image[0::2, 0::2]
        R = cfa_image[0::2, 1::2]
        B = cfa_image[1::2, 0::2]
        G2 = cfa_image[1::2, 1::2]
    elif pattern == 'GBRG':
        G1 = cfa_image[0::2, 0::2]
        B = cfa_image[0::2, 1::2]
        R = cfa_image[1::2, 0::2]
        G2 = cfa_image[1::2, 1::2]
    elif pattern == 'BGGR':
        B = cfa_image[0::2, 0::2]
        G1 = cfa_image[0::2, 1::2]
        G2 = cfa_image[1::2, 0::2]
        R = cfa_image[1::2, 1::2]
    else:
        raise NotImplementedError(f"Pattern {pattern} not implemented")
        
    # Stack R, G1, G2, B
    rggb_stack = np.stack([R, G1, G2, B], axis=0)
    return rggb_stack

def pixel_unshuffle(x, r):
    C, H, W = x.shape
    x = (
        x.reshape(C, H // r, r, W // r, r)
        .transpose(0, 2, 4, 1, 3)
        .reshape(C * r**2, H // r, W // r)
    )
    return x

class DemosaicingDataset(Dataset):
    """
    Dataset for learned demosaicing.
    
    Workflow:
    1. Load High-Res Noisy DNG.
    2. Crop a large patch, get its RGB representation (e.g., from RawHandler).
    3. Downsample this RGB patch (area) to create the Ground Truth image.
    4. Apply Bayer mosaicing to the Ground Truth to create the network input.
    5. Provide input in sparse (3-ch) and RGGB (4-ch) formats.
    """
    def __init__(self, path, csv, colorspace, 
                 output_crop_size=256, 
                 downsample_factor=2,
                 buffer=10, 
                 validation=False,
                 bayer_pattern='RGGB'):
        super().__init__()
        self.df = pd.read_csv(csv)
        self.path = Path(path)
        self.output_crop_size = (output_crop_size // 2) * 2 # Ensure even
        self.downsample_factor = downsample_factor
        self.input_crop_size = self.output_crop_size * self.downsample_factor
        self.buffer = buffer
        self.coordinate_iso = 6400.0 # Normalization constant for ISO
        self.validation = validation
        self.dtype = np.float32 # Use float32 for tensors
        self.colorspace = colorspace
        self.bayer_pattern = bayer_pattern

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        try:
            name = Path(f"{row.bayer_path}").name
            name = str(self.path / name.replace('_bayer.jpg', '.dng'))
            noisy_rh = RawHandler(name)
        except Exception as e:
            print(f"Error loading {name}: {e}")
            return self.__getitem__((idx + 1) % len(self)) # Skip bad file

        try:
            dims = random_crop_dim(
                noisy_rh.raw.shape, 
                self.input_crop_size, 
                self.buffer, 
                validation=self.validation
            )
            
            # Get (3, H_in, W_in) RGB patch from RawHandler
            noisy_rgb_full = noisy_rh.as_rgb(
                dims=dims, 
                colorspace=self.colorspace
            ).astype(self.dtype)

            # 3. Downsample to create Ground Truth
            # Convert to (1, 3, H_in, W_in) tensor for interpolation
            noisy_tensor = torch.from_numpy(noisy_rgb_full).unsqueeze(0)

            gt_tensor = F.interpolate(
                noisy_tensor, 
                scale_factor=1.0/self.downsample_factor, 
                mode='area', 
                recompute_scale_factor=False # Use scale_factor directly
            )

            # Back to (3, H_out, W_out) and clip
            gt_tensor = gt_tensor.squeeze(0).clip(0, 1)

            # 4. Apply Bayer mosaicing to create network input
            # Convert GT to (H_out, W_out, 3) numpy array
            gt_numpy = gt_tensor.permute(1, 2, 0).numpy()

            # Create (H_out, W_out) 2D CFA
            input_cfa_hw = mosaicing_CFA_Bayer(
                gt_numpy, 
                pattern=self.bayer_pattern
            )

            # 5. Provide input in sparse and RGGB formats
            
            # Create (H_out, W_out, 3) sparse array
            input_sparse_chw = cfa_to_sparse(
                input_cfa_hw, 
                pattern=self.bayer_pattern
            )[0]
            # Convert to (3, H_out, W_out) tensor
            input_sparse_chw = torch.from_numpy(
                input_sparse_chw
            ).to(torch.float32)


            # Create (4, H_out/2, W_out/2) RGGB stack
            input_cfa_hw_expanded = np.expand_dims(input_cfa_hw, 0)
            input_rggb_stack = pixel_unshuffle(input_cfa_hw_expanded, 2)
            # input_rggb_stack = cfa_to_rggb_stack(
            #     input_cfa_hw, 
            #     pattern=self.bayer_pattern
            # )
            # Convert to (4, H_out/2, W_out/2) tensor
            input_rggb_tensor = torch.from_numpy(input_rggb_stack).to(torch.float32)

            # 6. Conditioning tensor
            conditioning = torch.tensor(
                [row.iso / self.coordinate_iso]
            ).to(torch.float32)

            output = {
                "ground_truth": gt_tensor,        # (3, H_out, W_out)
                "cfa_sparse": input_sparse_chw,  # (3, H_out, W_out)
                "cfa_rggb": input_rggb_tensor,   # (4, H_out/2, W_out/2)
                "conditioning": conditioning     # (1,)
            }
            return output

        except Exception as e:
            print(f"Error processing {name} (idx {idx}): {e}")
            return self.__getitem__((idx + 1) % len(self)) # Skip bad file
