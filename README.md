# Restorer

## Overview

This branch contains experimental code for a lightweight, local training workflow to aid in the development of models for **Raw Refinery**. The goal is to enable faster model iteration on low-powered local computers with limited SSD space when access to large GPUs and SSDs are limited.

While functional, it remains a **work in progress**, primarily intended as a proof-of-concept for faster local experimentation.

Because the models are designed to be computationally efficient, the main bottleneck in this workflow is **disk I/O**. Raw image files are high-resolution 12-bit images, too large for the laptop’s internal SSD but too slow to stream from an external HDD. To mitigate this, we generate a smaller, compressed dataset suitable for quick pretraining.

---

## Producing the Smaller Dataset

We pretrain using a compressed **8-bit version** of the noisy raw sensor data and their corresponding **preprocessed (demosaiced)** ground-truth images.

After testing various formats, **JPEG** proved to be the most convenient and performant:

* Faster to read than PNG or NumPy-based formats
* Smaller file sizes even at high quality settings
* Sufficiently preserves high-frequency detail in the Bayer sensor data

*(A possible future improvement would be to store each of the four Bayer channels as separate JPEGs and merge them at runtime, but this implementation opts for simplicity.)*

### Relevant Files

* **`download_rawnind.sh`**
  Downloads the [RawNIND dataset](https://doi.org/10.14428/DVN/DEQCIM) file-by-file, as it’s too large to fetch directly. You can place it in any directory where you wish the dataset to be stored.

* **`config.yaml`**
  Contains global configuration parameters including file paths, training hyperparameters, and general workflow settings.

* **`0_produce_small_dataset.ipynb`**
  Two-part notebook:

  1. Iterates through raw images, aligns a demosaiced ground-truth image to each noisy image, and saves aligned pairs as JPEGs.

     * Alignment first uses a **feature-based method** for coarse alignment.
     * Followed by an **ECC alignment** for sub-pixel precision.
     * The **correlation coefficient** is saved to help identify poorly aligned images.
     * Manual inspection is still recommended—alignment quality is critical for effective model training.
  2. Optionally crops the JPEGs to reduce storage requirements and speed up data loading.

* **`0_align_images.ipynb`**
  Re-runs image alignment. Useful for experimenting with alternative alignment techniques or metrics. Not required if the previous notebook has already aligned the dataset.

---

## Pretraining

With the small dataset prepared, we can begin **local pretraining**.

Training hyperparameters are stored in `config.yaml`. Typical settings:

* **Patch size:** 256×256
* **Batch size:** 2 (for low-memory environments)
* **Optimizer:** Adam
* **Learning rate:** constant (can be scaled for larger GPUs)
* **Epochs:** 75 (sufficient for baseline performance); 150 preferred (~<24 hrs training time)

MLflow is used to track runs, hyperparameters, and metrics.

The dataset is split **80/20** into training and validation subsets.
Regularization methods like L2 weight decay, dropout, or strong augmentations often **harm** reconstruction performance; **random cropping** is typically sufficient.

### Relevant Files

* **`1_pretrain_model.ipynb`**
  Main training notebook. Loads the model, initializes the optimizer, and runs the training loop.
  MLflow integration records hyperparameters and results.
  *(Validation is not yet integrated directly in the training loop.)*
  **To-do:** Add model checkpointing.

* **`1_validate_model.ipynb`**
  Validates trained models.
  Separated from the main loop to allow for manual visual inspection of images with different noise levels, which is often more informative than numerical loss metrics.
  **To-do:** Save validation artifacts (sample images, metrics) to the MLflow run.

* **`src/Restorer/Cond_NAF.py`**
  Defines the neural network model.

* **`src/training/train_loop.py`**
  Implements the core training and validation loops.

* **`src/training/losses/ShadowAwareLoss.py`**
  Custom loss combining:

  * L1 loss
  * Multi-scale SSIM loss
  * Perceptual loss (VGG-like)
    Designed to emphasize realistic and visually appealing reconstructions.

---

## Fine-Tuning

Once pretraining converges, we fine-tune the model on **real raw images**.
This step restores high-frequency detail lost during 8-bit compression.

Fine-tuning uses similar hyperparameters to pretraining, but introduces **Cosine annealing** for smoother learning rate decay.

This portion of the workflow is still will be added shortly.

---

## Deployment

To deploy a trained model, we **trace** it into a TorchScript module for seamless integration with the **Raw Refinery** application.

This is handled in the **`3_make_script.ipynb`** notebook.

---

## Acknowledgments

> Brummer, Benoit; De Vleeschouwer, Christophe. (2025).
> *Raw Natural Image Noise Dataset.*
> [https://doi.org/10.14428/DVN/DEQCIM](https://doi.org/10.14428/DVN/DEQCIM), Open Data @ UCLouvain, V1.

> Chen, Liangyu; Chu, Xiaojie; Zhang, Xiangyu; Chen, Jianhao. (2022).
> *NAFNet: Simple Baselines for Image Restoration.*
> [https://doi.org/10.48550/arXiv.2208.04677](https://doi.org/10.48550/arXiv.2208.04677), arXiv, V1.