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
import cv2
from src.training.align_images import apply_alignment


class SmallRawDataset(Dataset):
    def __init__(self, path, csv, crop_size=180, buffer=10, validation=False):
        super().__init__()
        self.df = pd.read_csv(csv)
        self.path = path
        self.crop_size = crop_size
        self.buffer = buffer
        self.coordinate_iso = 6400
        self.validation=validation

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Load images
        with imageio.imopen(self.path / f"{row.noisy_image}_bayer.jpg", "r") as image_resource:
            bayer_data = image_resource.read()

        with imageio.imopen(self.path / f"{row.gt_image}.jpg", "r") as image_resource:
            gt_image = image_resource.read()
        gt_image  = gt_image/255
        bayer_data = bayer_data/255

        aligned = apply_alignment(gt_image, row.to_dict())
        demosaiced_noisy = demosaicing_CFA_Bayer_Malvar2004(bayer_data)

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

        # Translate to linear
        gt_image = inverse_gamma_tone_curve(gt_image)
        aligned = inverse_gamma_tone_curve(aligned)
        bayer_data = inverse_gamma_tone_curve(bayer_data)

        demosaiced_noisy = demosaicing_CFA_Bayer_Malvar2004(bayer_data)

        aligned = aligned * demosaiced_noisy.mean() / aligned.mean()
        gt_image = gt_image * demosaiced_noisy.mean() / gt_image.mean()

        sparse, _ = cfa_to_sparse(bayer_data)
        rggb =bayer_data.reshape(h // 2, 2, w // 2, 2, 1).transpose(1, 3, 4, 0, 2).reshape(4, h // 2, w // 2)

        # Convert to tensors
        output = {
            "bayer": torch.tensor(bayer_data).to(float).clip(0,1), 
            "gt": torch.tensor(gt_image).to(float).permute(2, 0, 1).clip(0,1), 
            "aligned": torch.tensor(aligned).to(float).permute(2, 0, 1).clip(0,1), 
            "sparse": torch.tensor(sparse).to(float).clip(0,1),
            "noisy": torch.tensor(demosaiced_noisy).to(float).permute(2, 0, 1).clip(0,1), 
            "rggb": torch.tensor(rggb).to(float).clip(0,1),
            "conditioning": torch.tensor([row.iso/self.coordinate_iso]).to(float), 
        }
        return output