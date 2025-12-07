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

# from src.training.utils import inverse_gamma_tone_curve, cfa_to_sparse
import numpy as np
import torch
# from src.training.align_images import apply_alignment, align_clean_to_noisy
from pathlib import Path
from RawHandler.RawHandler import RawHandler

# from .align_images import apply_alignment

class RawDatasetDNGDeblur(Dataset):
    def __init__(self, path, csv, colorspace, crop_size=180, buffer=10,
                 validation=False, run_align=False,
                 dimensions=2000,
                 apply_exposure_corr=True,
                 demosaicing_func = demosaicing_CFA_Bayer_Malvar2004,
                 blur_range=[0,300],
                 blur_buffer=30,
                 device='cuda'
                 ):
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
        self.apply_exposure_corr = apply_exposure_corr
        self.demosaicing_func = demosaicing_func
        self.blur_range = blur_range
        self.blur_buffer = blur_buffer
        self.device = device

        files = os.listdir(path)
        files = [f for f in files if 'dng' in f]
        files = [f for f in files if not 'xmp' in f]
        self.rhs = {}
        for file in files:
          self.rhs[file] = RawHandler(f'Cropped_Raw/{file}')


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Load images
        # name = Path(f"{row.bayer_path}").name
        # name = name.replace('_bayer.jpg', '.dng')
        # noisy_rh = self.rhs[name]

        gt_name =  Path(f"{row.gt_path}").name
        gt_name = gt_name.replace('.jpg', '.dng')
        gt_rh = self.rhs[gt_name]
        # gt_rh = list(self.rhs.items())[idx][1]

        dims = random_crop_dim(gt_rh.raw.shape, self.crop_size+self.blur_buffer, self.buffer, validation=self.validation)


        # check_align_matrix(row)
        expanded_dims = [dims[0]-self.buffer, dims[1]+self.buffer, dims[2]-self.buffer, dims[3]+self.buffer]
        gt_expanded = gt_rh.as_rgb(dims=expanded_dims, colorspace=self.colorspace, demosaicing_func=self.demosaicing_func, clip=False)
        # if self.apply_exposure_corr:
        #     gt_expanded[0] *= row['r_scale_factor']
        #     gt_expanded[1] *= row['g_scale_factor']
        #     gt_expanded[2] *= row['b_scale_factor']
        # aligned = apply_alignment(gt_expanded.transpose(1, 2, 0), row.to_dict())[self.buffer:-self.buffer, self.buffer:-self.buffer]
        aligned = gt_expanded.transpose(1, 2, 0)[self.buffer:-self.buffer, self.buffer:-self.buffer]

        debayered = torch.tensor(aligned).permute(2, 0, 1).unsqueeze(0).float()

        # Crop out edges
        debayered = debayered[:, :, self.blur_buffer:-self.blur_buffer, self.blur_buffer:-self.blur_buffer][0]

        # Convert to tensors
        output = {
            "aligned": debayered.to(float).clip(0,1),
            "noisy": debayered.to(float).clip(0,1),
            "conditioning": torch.tensor([row.iso/self.coordinate_iso]).to(float),
        }
        return output




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

def check_align_matrix(row, tolerance=1e-7):
        is_identity = np.isclose(row['M00'], 1.0, atol=tolerance) and \
        np.isclose(row['M01'], 0.0, atol=tolerance) and \
        np.isclose(row['M10'], 0.0, atol=tolerance) and \
        np.isclose(row['M11'], 1.0, atol=tolerance)

        assert is_identity, "Rotations, scalings, or shearing are not tested."


def round_to_nearest_2(number):
  return round(number / 2) * 2


from scipy.stats import multivariate_normal

def random_walk_kernel(n=100, scale=1, std_scale=1, min_val=1e-3, num_bins=101):
    
    covariance = np.array([[1, 0], [0, 1]])
    kernel = np.zeros([num_bins, num_bins])
    ax = np.linspace(-(num_bins-1)/2, (num_bins-1)/2, num_bins)
    xs, ys = np.meshgrid(ax, ax)
    points = np.stack((xs, ys), axis=-1)
    x, y = 0, 0
    x_list = [(x, y)]
    for _ in range(n):
        x += np.random.normal()*scale
        y += np.random.normal()*scale
        mean = np.array([x, y]) 
        x_list.append((x,y))
        pdf = multivariate_normal.pdf(points/std_scale, mean=mean, cov=covariance)
        kernel += pdf

    # Compute center of mass
    x_com = (kernel.sum(axis=1) * ax).sum()/(kernel.sum()+1e-6)
    y_com = (kernel.sum(axis=0) * ax).sum()/(kernel.sum()+1e-6)

    try:
        # Roll by center of mass
        kernel = np.roll(kernel, -int(x_com), axis=0 )
        kernel = np.roll(kernel, -int(y_com), axis=1 )
    except:
        print("failed roll", x_com, y_com, kernel.shape)

    # Compute center of mass
    x_com = (kernel.sum(axis=1) * ax).sum()/(kernel.sum()+1e-6)
    y_com = (kernel.sum(axis=0) * ax).sum()/(kernel.sum()+1e-6)

    # Crop
    x = kernel.sum(axis=0)
    filled_x = np.where(x>min_val)
    x_range = (filled_x[0][0], filled_x[0][-1])
    if x_range[1] > num_bins - x_range[0]:
        x_cut = num_bins - x_range[1]
    else:
        x_cut = x_range[0]
    
    y = kernel.sum(axis=1)
    filled_y = np.where(y>min_val)
    y_range = (filled_y[0][0], filled_y[0][-1])
    if y_range[1] > num_bins - y_range[0]:
        y_cut = num_bins - y_range[1]
    else:
        y_cut = y_range[0]
    cut = min(x_cut, y_cut)
    kernel = kernel[cut:-cut, cut:-cut]

    # Normalize
    kernel = kernel/(kernel.sum())
    # Convert to weights for conv2d
    kernel_shape = kernel.shape
    kernel = torch.tensor(kernel).unsqueeze(0).expand(3,*kernel_shape).unsqueeze(1).float()
    return kernel

def kinematic_kernel(n=100, vel_scale=1e-1, accel_scale=1e-3, num_bins=41, std_scale=.6):
    
    covariance = np.array([[1, 0], [0, 1]])
    kernel = np.zeros([num_bins, num_bins])
    ax = np.linspace(-(num_bins-1)/2, (num_bins-1)/2, num_bins)
    xs, ys = np.meshgrid(ax, ax)
    points = np.stack((xs, ys), axis=-1)
    x, y = 0, 0
    vx, vy =  np.random.normal()*vel_scale,  np.random.normal()*vel_scale
    accx, accy =  np.random.normal()*accel_scale, np.random.normal()*accel_scale
    x_list = [(x, y)]
    for _ in range(n):
        x += vx
        y += vy
        vx += accx
        vy += accy
        mean = np.array([x, y]) 
        x_list.append((x,y))
        pdf = multivariate_normal.pdf(points/std_scale, mean=mean, cov=covariance)
        kernel += pdf


    # Compute center of mass
    x_com = (kernel.sum(axis=1) * ax).sum()/(kernel.sum()+1e-6)
    y_com = (kernel.sum(axis=0) * ax).sum()/(kernel.sum()+1e-6)

    try:
        # Roll by center of mass
        kernel = np.roll(kernel, -round(x_com), axis=0 )
        kernel = np.roll(kernel, -round(y_com), axis=1 )
    except:
        print("failed roll", x_com, y_com, kernel.shape)

    # Compute center of mass
    x_com = (kernel.sum(axis=1) * ax).sum()/(kernel.sum()+1e-6)
    y_com = (kernel.sum(axis=0) * ax).sum()/(kernel.sum()+1e-6)


    # Normalize
    kernel = kernel/(kernel.sum())
    # Convert to weights for conv2d
    kernel_shape = kernel.shape
    kernel = torch.tensor(kernel).unsqueeze(0).expand(3,*kernel_shape).unsqueeze(1).float()
    return kernel


def batched_conv(input, kernel):
  B, C, H, W = input.shape
  with torch.no_grad():
      input_reshaped = input.view(1, B * C, H, W)

      padding = kernel.shape[-1] // 2

      kernel = kernel.view(B * C, kernel.shape[-3], kernel.shape[-2], kernel.shape[-1])
      blurred_reshaped = torch.nn.functional.conv2d(
          input_reshaped,
          kernel,
          stride=1,
          padding=padding,
          groups=B * C
      )

      blurred_batch = blurred_reshaped.view(B, C, H, W)
      return blurred_batch
  

def make_kernel_batch(batch_size, kernel_func=kinematic_kernel, kwargs={}):
    kernels = []
    for i in range(batch_size):
        kernels.append(kernel_func(**kwargs))
    return torch.stack(kernels, axis=0)