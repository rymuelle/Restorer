import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
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

import numpy as np
import torch 
import cv2




def align_clean_to_noisy(clean_img, noisy_img, refine=True, verbose=False):
    """
    Aligns the clean image to the noisy image and returns:
      - aligned image
      - best warp matrix (2x3 affine)
      - metrics dict (PSNR/SSIM before and after alignment)
    """

    # --- convert to grayscale float32 for processing ---
    def to_gray_f(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        g = g.astype(np.float32)
        g = (g - g.mean()) / (g.std() + 1e-8)
        return g

    clean_gray = to_gray_f(clean_img)
    noisy_gray = to_gray_f(noisy_img)

    h, w = clean_gray.shape
    aligned = clean_img.copy()

    # --- PHASE CORRELATION (coarse translation) ---
    shift, response = cv2.phaseCorrelate(noisy_gray, clean_gray)  # (dx, dy)
    dx, dy = shift
    M_trans = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)

    # apply translation
    aligned = cv2.warpAffine(clean_img, M_trans, (w, h),
                             flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REFLECT)

    # --- optional ECC refinement (affine) ---
    if refine:
        warp_mode = cv2.MOTION_AFFINE
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 2000, 1e-8)
        try:
            cc, warp_matrix = cv2.findTransformECC(
                noisy_gray,  # template (target)
                cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY).astype(np.float32) if aligned.ndim == 3 else aligned.astype(np.float32),
                warp_matrix,
                warp_mode,
                criteria,
                None,
                5
            )
            if verbose:
                print(f"ECC converged: corr={cc:.5f}")
            # compose the transforms: M_total = M_ECC @ M_trans
            M1 = np.vstack([M_trans, [0, 0, 1]])
            M2 = np.vstack([warp_matrix, [0, 0, 1]])
            M_total = (M2 @ M1)[:2, :]
            aligned = cv2.warpAffine(clean_img, M_total, (w, h),
                                     flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
                                     borderMode=cv2.BORDER_REFLECT)
        except cv2.error as e:
            if verbose:
                print("ECC failed:", e)
            M_total = M_trans
    else:
        M_total = M_trans

    # --- compute metrics ---
    def safe_gray(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    clean_g = safe_gray(clean_img)
    noisy_g = safe_gray(noisy_img)
    aligned_g = safe_gray(aligned)

    before_psnr = psnr(noisy_g, clean_g, data_range=255)
    after_psnr = psnr(noisy_g, aligned_g, data_range=255)
    before_ssim = ssim(noisy_g, clean_g, data_range=255)
    after_ssim = ssim(noisy_g, aligned_g, data_range=255)

    metrics = {
        "PSNR_before": before_psnr,
        "PSNR_after": after_psnr,
        "SSIM_before": before_ssim,
        "SSIM_after": after_ssim,
        "dx": dx,
        "dy": dy,
        "response": response,
    }

    # flatten warp matrix for CSV
    for i in range(2):
        for j in range(3):
            metrics[f"M{i}{j}"] = float(M_total[i, j])

    return aligned, M_total, metrics



def apply_alignment(img, warp_params, interpolation=cv2.INTER_LINEAR):
    """
    Applies a previously estimated affine warp to an image.
    warp_params: dict with keys M00..M12 or a 2x3 numpy array.
    """
    if isinstance(warp_params, dict):
        M = np.array([
            [warp_params["M00"], warp_params["M01"], warp_params["M02"]],
            [warp_params["M10"], warp_params["M11"], warp_params["M12"]],
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




class AlignImages(Dataset):
    def __init__(self, path, csv, crop_size=180, buffer=10, validation=False):
        super().__init__()
        self.df = pd.read_csv(os.path.join(path, csv))
        self.path = path
        self.crop_size = crop_size
        self.buffer = buffer
        self.coordinate_iso = 6400
        self.validation=validation

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Get Row Matrix
        shape=(2,3)
        cols = [f"m{i}{j}" for i in range(shape[0]) for j in range(shape[1])]
        flat = np.array([row.pop(c) for c in cols], dtype=np.float32)
        warp_matrix = flat.reshape(shape)
        warp_matrix

        # Load images
        bayer_path = f"{self.path}/{row.noisy_image}_bayer.jpg"
        with imageio.imopen(bayer_path, "r") as image_resource:
            bayer_data = image_resource.read()

        gt_path = f"{self.path}/{row.gt_image}.jpg"
        with imageio.imopen(gt_path, "r") as image_resource:
            gt_image = image_resource.read()

        demosaiced_noisy = demosaicing_CFA_Bayer_Malvar2004(bayer_data)
        demosaiced_noisy = demosaiced_noisy.astype(np.uint8)
        aligned, matrix, metrics = align_clean_to_noisy(gt_image, demosaiced_noisy, refine=False)
        metrics['iso'] = row.iso
        metrics['std'] = (demosaiced_noisy.astype(int) - aligned.astype(int)).std()
        metrics['bayer_path'] = bayer_path
        metrics['gt_path'] = gt_path
        return gt_image, demosaiced_noisy, aligned, metrics