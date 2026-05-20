import traceback
from tqdm import tqdm
import pandas as pd
import numpy as np
import imageio
import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path
import re
from collections import defaultdict
import rawpy


from RawHandler.RawHandlerRawpy import RawHandlerRawpy

import numpy as np
import cv2
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift

from src.training.load_config import load_config
from src.training.align_images import align_clean_to_noisy
from src.training.censored_fit import censored_linear_fit_twosided


# Read config
run_config = load_config()
raw_path = Path(run_config['base_data_dir'])
outpath = Path(run_config['numpy_raw_subdir'])
alignment_csv =  outpath / run_config['align_csv']
def is_raw(filename):
    if '.npy' in filename: 
        return False
    if '.xmp' in filename:
        return False
    return True

file_list = [f for f in os.listdir(raw_path) if is_raw(f)]


def pair_images_by_scene(file_list, min_iso=100):
    """
    Given a list of RAW image file paths:
      1. Extract ISO from filenames
      2. Remove files with ISO < min_iso
      3. Group by scene name
      4. Pair each image with the lowest-ISO version of the scene

    Args:
        file_list (list of str): Paths to RAW files
        min_iso (int): Minimum ISO to keep (default=100)

    Returns:
        dict: {scene_name: [(img_path, gt_path), ...]}
    """
    iso_pattern = re.compile(r"_ISO(\d+)_")
    scene_pairs = {}

    # Step 1: Extract iso and scene
    images = []
    for path in file_list:
        filename = os.path.basename(path)
        match = iso_pattern.search(filename)
        if not match:
            continue  # skip if no ISO
        iso = int(match.group(1))
        if iso < min_iso:
            continue  # filter out low ISOs

        # Extract scene name:
        if "_GT_" in filename:
            scene = filename.split("_GT_")[0]
        else:
            # Scene = part before "_ISO"
            scene = filename.split("_ISO")[0]

        images.append((scene, iso, path))

    # Step 2: Group by scene
    grouped = defaultdict(list)
    for scene, iso, path in images:
        grouped[scene].append((iso, path))

    # Step 3: For each scene, pick lowest ISO as GT
    for scene, iso_paths in grouped.items():
        iso_paths.sort(key=lambda x: x[0])  # sort by ISO ascending
        gt_iso, gt_path = iso_paths[0]      # lowest ISO ≥ min_iso
        pairs = [(path, gt_path) for iso, path in iso_paths if path != gt_path]
        scene_pairs[scene] = pairs

    return scene_pairs


pair_file_list = pair_images_by_scene(file_list)

def get_file(impath):
        impath = str(impath)
        rh = RawHandlerRawpy(impath)
        rp = rh.rawpy_object
        image = rp.postprocess(
            user_wb=[1, 1, 1, 1],
            output_color=rawpy.ColorSpace.raw,
            no_auto_bright=True,
            use_camera_wb=False,
            use_auto_wb=False,
            gamma=(1, 1),
            user_flip=0,
            output_bps=16,
            no_auto_scale=True,
            user_black=0,
        ) / rp.white_level
        return image, rh, rh.rgb_colorspace_transform()
        # plt.imshow((image @ ccm.T).clip(0,1)**.5)


def save_and_align_image(image_pairs, dont_align=False):
    # GT
    gt_path = raw_path / image_pairs[1]
    gt_out = outpath / (image_pairs[1] + '.npy')
    gt_out_ccm = outpath / (image_pairs[1] + '_ccm.npy')
    if not os.path.exists(gt_out):
        gt_image, _, gt_ccm = get_file(gt_path)
        np.save(gt_out, gt_image.astype(np.float16))
        np.save(gt_out_ccm, gt_ccm.astype(np.float16))
    else:
        gt_image =  np.load(gt_out, mmap_mode='r')

    # Degraded
    deg_path = raw_path / image_pairs[0]
    deg_out = outpath / (image_pairs[0] + '.npy')
    deg_out_ccm = outpath / (image_pairs[0] + '_ccm.npy')
    deg_out_pattern = outpath / (image_pairs[0] + '_pattern.npy')

    deg_image, noisyrh, deg_ccm = get_file(deg_path)
    iso = noisyrh.full_metadata.get_ISO()
    deg_raw = noisyrh.rawpy_object.raw_image_visible / noisyrh.rawpy_object.white_level
    cfa_pattern =  noisyrh.core_metadata.raw_pattern
    np.save(deg_out, deg_raw.astype(np.float16))
    np.save(deg_out_ccm, deg_ccm.astype(np.float16))
    np.save(deg_out_pattern, cfa_pattern.astype(np.float16))

    if dont_align:  return 0, 0, 0
    # Exposure alignment 
    a, b, sigma = censored_linear_fit_twosided(
        gt_image.astype(np.float32), 
        deg_image.astype(np.float32),
        clip_high=1, clip_low=0
        )
    gt_image = a + b * gt_image

    gt_image, deg_image = (gt_image **.5 * 255).astype(np.uint8), (deg_image ** .5 * 255).astype(np.uint8)
    aligned, M, metrics = align_clean_to_noisy(gt_image, deg_image, blur_noisy=True)
    
    metrics['a'] = a
    metrics['b'] = b
    metrics['sigma'] = sigma
    metrics['iso'] = iso
    metrics['gt_out'] = str(gt_out)
    metrics['gt_out_ccm'] = str(gt_out_ccm)
    metrics['deg_out'] = str(deg_out)
    metrics['deg_out_ccm'] = str(deg_out_ccm)
    metrics['deg_out_cfa'] = str(deg_out_pattern)
    return metrics, aligned/255, deg_image/255



# Run the main preproccessing
outpath.mkdir(parents=True, exist_ok=True)

# Check csv for processed files
processed_deg_files = set()
if alignment_csv.exists():
    try:
        df_existing = pd.read_csv(alignment_csv)
        if 'deg_out' in df_existing.columns:
            processed_deg_files = set(df_existing['deg_out'].dropna().astype(str).tolist())
        print(f"Found existing CSV. Resuming from checkpoint: {len(processed_deg_files)} images already processed.")
    except Exception as e:
        print(f"Could not read existing CSV ({e}). Starting fresh.")

metrics_written_count = 0

for key in tqdm(pair_file_list.keys(), desc="Processing Scenes"):
    image_pairs = pair_file_list[key]
    
    for ip in image_pairs:
        # Check if the image exists
        expected_deg_out = str(outpath / (ip[0] + '.npy'))

        dont_align = False
        if expected_deg_out in processed_deg_files:
            dont_align = True
            if os.path.exists(expected_deg_out):
                continue

        try:
            # Process the image pair
            metric, _, _ = save_and_align_image(ip, dont_align=dont_align)
            # If the CSV already had a file, skip adding it
            if dont_align: 
                continue
            # Write CSV
            df_row = pd.DataFrame([metric])
            
            header_needed = not alignment_csv.exists()
            df_row.to_csv(alignment_csv, mode='a', index=False, header=header_needed)
            processed_deg_files.add(expected_deg_out)
            metrics_written_count += 1
            
        except Exception as e:
            print(f"\n[ERROR] Skipping pair due to failure processing {ip[0]}: {e}")
            continue

print(f"\nProcessing complete! Added {metrics_written_count} new records to {alignment_csv.name}")