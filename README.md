# PGSPN


## Overview
This repository contains the official implementation of **PGSPN (Photometric Guidance Spatial Propagation Network)** for robust depth completion. Our method achieves state-of-the-art performance on challenging underwater depth completion tasks with various sparsity patterns.

**Note:** Currently, only inference code is provided. Training code will be released upon paper acceptance.

## Installation

### Prerequisites
- Docker
- NVIDIA GPU with CUDA support

### Tested GPUs
Our code has been tested and verified on the following GPUs:
- NVIDIA RTX 3090 Ti
- NVIDIA RTX A6000

### Environment Setup

1. **Launch Docker Container**
   ```bash
   bash run_docker.sh up gpu
   ```

2. **Navigate to Workspace** (inside container)
   ```bash
   cd /root/workspace/
   ```

3. **Download Pre-trained Checkpoints**

   Download the following checkpoint files from [Google Drive](https://drive.google.com/drive/folders/1pPJ2MNMdZyg2XhKhqKMC4fYKX0V07Cep?usp=drive_link) and place them in the `checkpoints/` directory:
   - `depth_completion.pth`
   - `transmission_estimator.pth`

   The directory structure should be:
   ```
   /root/workspace/
   ├── checkpoints/
   │   ├── depth_completion.pth
   │   └── transmission_estimator.pth
   └── ...
   ```

## Inference

### Basic Usage

Run inference using the following command structure:

```bash
python3 inference_pgspn.py \
    --input_img_folder <path_to_rgb_images> \
    --input_sparse_folder <path_to_sparse_depth> \
    --output_folder <output_directory> \
    --device <gpu_id> \
    --save_colormap \
    --colormap_inverse
```

### Example Commands

#### 1. Height-biased Sparse Depth (20% density)
```bash
python3 inference_pgspn.py \
    --input_img_folder input_samples/rgb_images \
    --input_sparse_folder input_samples/sparse_height_bias_20 \
    --output_folder results_height_bias_20 \
    --device 0 \
    --save_colormap \
    --colormap_inverse
```

#### 2. SfM-based Sparse Depth (10% density)
```bash
python3 inference_pgspn.py \
    --input_img_folder input_samples/rgb_images \
    --input_sparse_folder input_samples/sparse_sfm_10 \
    --output_folder results_sfm_10 \
    --device 0 \
    --save_colormap \
    --colormap_inverse
```

#### 3. Uniform Sparse Depth (50 points)
```bash
python3 inference_pgspn.py \
    --input_img_folder input_samples/rgb_images \
    --input_sparse_folder input_samples/sparse_uniform_50 \
    --output_folder results_uniform_50 \
    --device 0 \
    --save_colormap \
    --colormap_inverse
```

### Arguments

- `--input_img_folder`: Path to RGB images directory
- `--input_sparse_folder`: Path to sparse depth maps directory
- `--output_folder`: Output directory for completed depth maps
- `--device`: GPU device ID (default: 0)
- `--save_colormap`: Save depth maps as color-coded visualizations
- `--colormap_inverse`: Use inverse colormap for visualization

## Input Format

- **RGB Images**: Standard RGB images in PNG/JPG format
- **Sparse Depth**: Sparse depth maps in compatible format (NPY/PNG)

## Output

The inference script generates:
- Completed dense depth maps
- Color-coded depth visualizations (if `--save_colormap` is enabled)
- Quantitative metrics (if ground truth is available)

---

**Status:** 🔧 Inference code available | Training code to be released