import pandas as pd
import os
from  torch.utils.data import Dataset
import imageio
from colour_demosaicing import (
    ROOT_RESOURCES_EXAMPLES,
    demosaicing_CFA_Bayer_bilinear,
    demosaicing_CFA_Bayer_Malvar2004,
    demosaicing_CFA_Bayer_Menon2007,
    mosaicing_CFA_Bayer)

from src.training.utils import inverse_gamma_tone_curve, cfa_to_sparse
import numpy as np
import torch 
from src.training.align_images import apply_alignment, align_clean_to_noisy
from pathlib import Path
from RawHandler.RawHandler import RawHandler



def global_affine_match(A, D, mask=None):
    """
    Fit D ≈ a + b*A with least squares.
    A, D : 2D arrays, same shape (linear values)
    mask : optional boolean array, True=use pixel
    returns: a, b, D_pred, D_resid (D - (a + b*A))
    """
    A = A.ravel().astype(np.float64)
    D = D.ravel().astype(np.float64)
    if mask is None:
        mask = np.isfinite(A) & np.isfinite(D)
    else:
        mask = mask.ravel() & np.isfinite(A) & np.isfinite(D)

    A0 = A[mask]
    D0 = D[mask]
    # design matrix [1, A]
    X = np.vstack([np.ones_like(A0), A0]).T
    coef, *_ = np.linalg.lstsq(X, D0, rcond=None)
    a, b = coef[0], coef[1]
    D_pred = (a + b * A).reshape(-1)
    D_pred = D_pred.reshape(A.shape) if False else (a + b * A).reshape((-1,))  # keep flatten

    return a, b, (a + b * A)


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

class RawDatasetDNG(Dataset):
    def __init__(self, path, csv, colorspace, crop_size=180, buffer=10, validation=False, run_align=False, dimensions=2000):
        super().__init__()
        self.df = pd.read_csv(csv)
        self.path = path
        self.crop_size = crop_size
        self.buffer = buffer
        self.coordinate_iso = 6400
        self.validation=validation
        self.run_align = run_align
        self.dtype = np.float16
        self.dimensions = dimensions
        self.colorspace = colorspace

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Load images
        try:
            name =  Path(f"{row.bayer_path}").name
            name = str(self.path / name.replace('_bayer.jpg', '.dng'))
            noisy_rh = RawHandler(name)
        except:
            print(name)

        try:
            gt_name =  Path(f"{row.gt_path}").name
            gt_name = str(self.path / gt_name.replace('.jpg', '.dng'))
            gt_rh = RawHandler(gt_name)
        except:
            print(gt_name)

        dims = random_crop_dim(noisy_rh.raw.shape, self.crop_size, self.buffer, validation=self.validation)
        try:
            bayer_data = noisy_rh.apply_colorspace_transform(dims=dims, colorspace=self.colorspace)
            noisy = noisy_rh.as_rgb(dims=dims, colorspace=self.colorspace)
            rggb = noisy_rh.as_rggb(dims=dims, colorspace=self.colorspace)
            sparse = noisy_rh.as_sparse(dims=dims, colorspace=self.colorspace)

            expanded_dims = [dims[0]-self.buffer, dims[1]+self.buffer, dims[2]-self.buffer, dims[3]+self.buffer]
            gt_expanded = gt_rh.as_rgb(dims=expanded_dims, colorspace=self.colorspace)
            aligned = apply_alignment(gt_expanded.transpose(1, 2, 0), row.to_dict())[self.buffer:-self.buffer, self.buffer:-self.buffer]
            gt_non_aligned = gt_expanded.transpose(1, 2, 0)[self.buffer:-self.buffer, self.buffer:-self.buffer]
        except:
            print(name, gt_name)

        # aligned = gt_rh.as_rgb(dims=dims, colorspace=self.colorspace).transpose(1, 2, 0)

        # Convert to tensors
        output = {
            "bayer": torch.tensor(bayer_data).to(float).clip(0,1), 
            "gt_non_aligned": torch.tensor(gt_non_aligned).to(float).permute(2, 0, 1).clip(0,1), 
            "aligned": torch.tensor(aligned).to(float).permute(2, 0, 1).clip(0,1), 
            "sparse": torch.tensor(sparse).to(float).clip(0,1),
            "noisy": torch.tensor(noisy).to(float).clip(0,1), 
            "rggb": torch.tensor(rggb).to(float).clip(0,1),
            "conditioning": torch.tensor([row.iso/self.coordinate_iso]).to(float), 
            # "noise_est": noise_est,
            # "rggb_gt": rggb_gt,
        }
        return output