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

class SmallRawDatasetNumpy(Dataset):
    def __init__(self, path, csv, crop_size=180, buffer=10, validation=False, run_align=False, dimensions=2000):
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
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Load images

        name =  Path(f"{row.bayer_path}").name
        name = name.replace('_bayer.jpg', '.u16.raw')
        bayer_data = np.fromfile(self.path / name, dtype=self.dtype)
        bayer_data = bayer_data.reshape((self.dimensions, self.dimensions))
        
        name =  Path(f"{row.gt_path}").name
        name = name.replace('jpg', 'u16.raw') 
        gt_image = np.fromfile(self.path / name, dtype=self.dtype)
        gt_image = gt_image.reshape((self.dimensions, self.dimensions))

       

        gt_image  = gt_image
        bayer_data = bayer_data
        gt_image = demosaicing_CFA_Bayer_Malvar2004(gt_image)
        demosaiced_noisy = demosaicing_CFA_Bayer_Malvar2004(bayer_data)

        if self.run_align:
            gt_image = (gt_image * 255).astype(np.uint8)
            demosaiced_noisy = (demosaiced_noisy * 255).astype(np.uint8)
            aligned, _, _ = align_clean_to_noisy(gt_image, demosaiced_noisy, refine=False, verbose=False)
            aligned = aligned / 255
        else:
            aligned = apply_alignment(gt_image, row.to_dict())
        h, w, _ = gt_image.shape
        
        #Crop images
        if not self.validation:
            top = np.random.randint(0 + self.buffer, h - self.crop_size - self.buffer)
            left = np.random.randint(0 + self.buffer, w - self.crop_size - self.buffer)
        else:
            top = (h - self.crop_size) // 2
            left = (w - self.crop_size) // 2
            
        if top % 2 != 0: top = top - 1
        if left % 2 != 0: left = left - 1
        bottom = top + self.crop_size
        right = left + self.crop_size
        aligned = aligned[top:bottom, left:right]
        gt_image = gt_image[top:bottom, left:right]
        bayer_data = bayer_data[top:bottom, left:right]
        h, w, _ = gt_image.shape

        demosaiced_noisy = demosaicing_CFA_Bayer_Malvar2004(bayer_data)
        sparse, _ = cfa_to_sparse(bayer_data)
        rggb = bayer_data.reshape(h // 2, 2, w // 2, 2, 1).transpose(1, 3, 4, 0, 2).reshape(4, h // 2, w // 2)

        # # Affine transform to match brightness in gt to noisy
        # a, b, aligned = global_affine_match(aligned, demosaiced_noisy)

        # print(a, b, aligned.shape)
        # gt_image = gt_image * demosaiced_noisy.mean() / gt_image.mean()
        # aligned = aligned * demosaiced_noisy.mean() / aligned.mean()


        # Sim noise method
        mosaiced_gt = mosaicing_CFA_Bayer(aligned)
        rggb_gt = mosaiced_gt.reshape(h // 2, 2, w // 2, 2, 1).transpose(1, 3, 4, 0, 2).reshape(4, h // 2, w // 2)
        noise_est = rggb - rggb_gt

        # Convert to tensors
        output = {
            "bayer": torch.tensor(bayer_data).to(float).clip(0,1), 
            "gt_non_aligned": torch.tensor(gt_image).to(float).permute(2, 0, 1).clip(0,1), 
            "aligned": torch.tensor(aligned).to(float).permute(2, 0, 1).clip(0,1), 
            "sparse": torch.tensor(sparse).to(float).clip(0,1),
            "noisy": torch.tensor(demosaiced_noisy).to(float).permute(2, 0, 1).clip(0,1), 
            "rggb": torch.tensor(rggb).to(float).clip(0,1),
            "conditioning": torch.tensor([row.iso/self.coordinate_iso]).to(float), 
            "noise_est": noise_est,
            "rggb_gt": rggb_gt,
        }
        return output