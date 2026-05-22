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

        bottom = top + crop_size
        right = left + crop_size
        return (left, right, top, bottom)

patterns = [np.array([[0., 2., 1., 2., 0., 1.],
        [1., 1., 0., 1., 1., 2.],
        [1., 1., 2., 1., 1., 0.],
        [2., 0., 1., 0., 2., 1.],
        [1., 1., 2., 1., 1., 0.],
        [1., 1., 0., 1., 1., 2.]], dtype=np.float16),
 np.array([[0., 1.],
        [3., 2.]], dtype=np.float16),
 np.array([[3., 2.],
        [0., 1.]], dtype=np.float16),
 np.array([[1., 1., 0., 1., 1., 2.],
        [1., 1., 2., 1., 1., 0.],
        [2., 0., 1., 0., 2., 1.],
        [1., 1., 2., 1., 1., 0.],
        [1., 1., 0., 1., 1., 2.],
        [0., 2., 1., 2., 0., 1.]], dtype=np.float16),
 np.array([[2., 3.],
        [1., 0.]], dtype=np.float16)]


def compute_mask(shape, pattern):
    H, W = shape
    ph, pw = pattern.shape
    # If two green channels, set both to 1
    _pattern = pattern.copy()
    _pattern[_pattern == 3] = 1
    # Create the output arrays
    mask = np.zeros((3, H, W), dtype=np.uint8)

    # Tile the pattern to match the CFA shape
    full_pattern = np.tile(_pattern, (H // ph + 1, W // pw + 1))
    full_pattern = full_pattern[:H, :W]

    # Vectorized assignment for each channel (R, G, B)
    for ch in range(3):
        ch_mask = full_pattern == ch
        mask[ch] = ch_mask
    return  mask

def make_masks(size, patterns):
    masks = []
    for pattern in patterns:
        mask = compute_mask(size, pattern)
        masks.append( (pattern, mask))
    return masks

def get_mask(pattern, mask):
    for _pattern, mask in mask:
        if np.array_equal(pattern, _pattern): 
            return mask
    return 0

class DemoDataset(Dataset):
    def __init__(self, csv, crop_size=256, buffer=10, validation=False):
        super().__init__()
        self.csv = pd.read_csv(csv)
        self.crop_size = crop_size
        self.validation = validation 
        self.buffer = buffer
        self.masks = make_masks((crop_size+6, crop_size+6), patterns)
    
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, i):
        row = self.csv.iloc[i]
        gt_path = row["gt_out"]
        deg_pattern_path = row["deg_out_ccm"].replace('ccm', 'pattern')
        deg_ccm = np.load(row["deg_out_ccm"])

        gt_image = np.load(gt_path, mmap_mode="r")
        deg_pattern = np.load(deg_pattern_path)

        H, W, C = gt_image.shape
        dims = random_crop_dim((W, H), self.crop_size, self.buffer, validation=self.validation)
        h1, h2, w1, w2 = dims
        gt_image = np.array(gt_image[h1:h2, w1:w2])
        gt_image = gt_image.transpose(2, 0, 1)

        mask = get_mask(deg_pattern, self.masks)
        C, H, W = mask.shape
        dims = random_crop_dim((W, H), self.crop_size, 0, validation=self.validation)
        h1, h2, w1, w2 = dims
        mask = mask[:, h1:h2, w1:w2]
        six_chan = np.concat([gt_image*mask, mask], axis=0)

        output = {
            "gt": torch.from_numpy(gt_image).to(torch.float32).clamp_(0.0, 1.0),
            "six_chan": torch.from_numpy(six_chan).to(torch.float32).clamp_(0.0, 1.0),
            "ccm": torch.from_numpy(deg_ccm).to(torch.float32)
   
        }
        return output