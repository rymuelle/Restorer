import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2

def random_crop_dim(shape, crop_size, buffer, validation=False):
        h, w = shape
        if not validation:
            top = np.random.randint(0 + buffer, h - crop_size - buffer)
            left = np.random.randint(0 + buffer, w - crop_size - buffer)
        else:
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2

        if top % 2 != 0: top = top - 1
        if left % 2 != 0: left = left - 1
        bottom = top + crop_size
        right = left + crop_size
        return (left, right, top, bottom)

def apply_alignment(img, warp_params, interpolation=cv2.INTER_LANCZOS4):
    """
    Applies a previously estimated affine warp to an image.
    warp_params: dict with keys m00..m12 or a 2x3 numpy array.
    """
    if isinstance(warp_params, dict):
        M = np.array([
            [warp_params["m00"], warp_params["m01"], -warp_params["m02"]],
            [warp_params["m10"], warp_params["m11"], -warp_params["m12"]],
        ], dtype=np.float32)
    else:
        M = np.array(warp_params, dtype=np.float32)

    h, w = img.shape[:2]
    aligned = cv2.warpAffine(
        img.astype(np.float32),
        M,
        (w, h),
        flags=interpolation + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT
    )
    return aligned

def sparse_representation_and_mask(cfa, pattern):
    H, W = cfa.shape
    ph, pw = pattern.shape
    # If two green channels, set both to 1
    pattern[pattern == 3] = 1
    # Create the output arrays
    sparse = np.zeros((3, H, W), dtype=cfa.dtype)
    mask = np.zeros((3, H, W), dtype=np.uint8)

    # Tile the pattern to match the CFA shape
    full_pattern = np.tile(pattern, (H // ph + 1, W // pw + 1))
    full_pattern = full_pattern[:H, :W]

    # Vectorized assignment for each channel (R, G, B)
    for ch in range(3):
        ch_mask = full_pattern == ch
        mask[ch] = ch_mask
        sparse[ch] = cfa * ch_mask
    return sparse, mask


def compute_mask_and_sparse(
        raw_img, pattern, dims=None, safe_crop=0
    ):
        if dims is not None:
            h1, h2, w1, w2 = dims
            if safe_crop:
                h1 -= h1 % safe_crop
                h2 -= h2 % safe_crop
                w1 -= w1 % safe_crop
                w2 -= w2 % safe_crop
            raw_img = raw_img[h1:h2, w1:w2]
            # Roll the pattern to align with crop
            pattern = np.roll(
                pattern, shift=(-h1, -w1), axis=(0, 1)
            )

        # Compute sparse representation on the (potentially smaller) image
        sparse, mask = sparse_representation_and_mask(raw_img, pattern)

        return sparse, mask


class JDDDataset(Dataset):
    def __init__(self, csv, crop_size=256, buffer=10, validation=False, augment=False):
        super().__init__()
        self.csv = pd.read_csv(csv)
        self.crop_size = crop_size
        self.validation = validation 
        self.buffer = buffer
        self.augment = augment
    
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, i):
        row = self.csv.iloc[i]
        gt_path = row["gt_out"]
        deg_path = row["deg_out"]
        deg_pattern_path = row["deg_out_ccm"].replace('ccm', 'pattern')
        # gt_ccm = np.load(row["gt_out_ccm"])
        deg_ccm = np.load(row["deg_out_ccm"])

        gt_image = np.load(gt_path, mmap_mode="r")
        deg_image = np.load(deg_path, mmap_mode="r")
        deg_pattern = np.load(deg_pattern_path, mmap_mode="r")

        H, W, C = gt_image.shape
        dims = random_crop_dim((W, H), self.crop_size, self.buffer, validation=self.validation)
        # _deg = np.array(deg_image[dims[0]:dims[1], dims[2]:dims[3]])

        sparse, mask = compute_mask_and_sparse(deg_image, deg_pattern, dims=dims)

        _gt_patch = np.array(gt_image[dims[0]-self.buffer:dims[1]+self.buffer,
                                     dims[2]-self.buffer:dims[3]+self.buffer])
        
    
        aligned = apply_alignment(_gt_patch, row.to_dict())[self.buffer:-self.buffer, self.buffer:-self.buffer]
        a = np.array([[row['ra'],row['ga'],row['ba']]])
        b = np.array([[row['rb'],row['gb'],row['bb']]])
        aligned = a + b * aligned
        aligned = aligned.transpose(2, 0, 1)
        # Au
        if self.augment:
            brightness_range = (0.8, 1/aligned.max()*1.2)
            sparse, aligned = augment_pair(sparse, aligned, brightness_range=brightness_range)
        _deg = np.concat([sparse, mask], axis=0)


        output = {
            "aligned": torch.from_numpy(aligned).to(torch.float32).clamp_(0.0, 1.0),
            "deg": torch.from_numpy(_deg).to(torch.float32).clamp_(0.0, 1.0),
            "iso": torch.tensor([row.iso], dtype=torch.float32),
            "ccm": torch.from_numpy(deg_ccm).to(torch.float32)
        }

        compute_mono_noise = True
        if compute_mono_noise:
            mono_noise = (output['deg'][:3] - (output['aligned'] * output['deg'][3:])).sum(axis=0, keepdim=True)
            output['mono_noise'] = mono_noise
            cfa_gt = (output['aligned'] * output['deg'][3:]).sum(axis=0, keepdim=True)
            mono_noise_proportion = output['mono_noise']/(cfa_gt+1e-6)
            output['mono_noise_proportion'] =  mono_noise_proportion
        return output


import numpy as np

def augment_pair(img1, img2, brightness_range=(0.8, 1.2)):
    """
    Applies identical random rotations and flips to both arrays to keep them aligned,
    and applies independent or joint exposure variations.
    
    Parameters:
        img1, img2 (np.ndarray): Arrays of shape (c, h, w)
        brightness_range (tuple): Min and max multiplier for exposure
        
    Returns:
        tuple: Augmented img1 and img2
    """
    assert img1.shape == img2.shape, "Arrays must have the same shape"
    assert len(img1.shape) == 3, "Input arrays must be of shape (c, h, w)"
    
    k_rot = np.random.randint(0, 4)
    if k_rot > 0:
        img1 = np.rot90(img1, k=k_rot, axes=(1, 2))
        img2 = np.rot90(img2, k=k_rot, axes=(1, 2))
        
    if np.random.rand() > 0.5:
        img1 = np.flip(img1, axis=2) # axis 2 is width
        img2 = np.flip(img2, axis=2)
        
    if np.random.rand() > 0.5:
        img1 = np.flip(img1, axis=1) # axis 1 is height
        img2 = np.flip(img2, axis=1)
        

    scale = np.random.uniform(*brightness_range)

    img1 = np.clip(img1 * scale, 0.0, 1.0)
    img2 = np.clip(img2 * scale, 0.0, 1.0)
    
    return img1, img2