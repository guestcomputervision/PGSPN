#!/usr/bin/env python3
"""
PGSPN Inference Script - Corrected Version
- Fixed: sparse_input is just 1 channel (sparse depth only)
- Fixed: Prior map is not used by the model
- Uses PIL for all image operations (no OpenCV)
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as tr
from PIL import Image
from pathlib import Path
import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import builders
from core.models.network_builder import MODEL_BUILDER
from local_configs.cfg.pgspn import basic_cfg, test_parm


class PGSPNInference:
    """PGSPN Inference - Corrected Implementation"""

    def __init__(self, args):
        self.args = args

        # Fixed input size
        self.input_height = 576
        self.input_width = 960

        # Normalization (ImageNet)
        self.normalizer = tr.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def setup_device(self):
        """Setup computation device"""
        if self.args.device == 'cpu':
            print("🚀 Using CPU")
            return torch.device('cpu')
        else:
            if torch.cuda.is_available():
                device_id = int(self.args.device) if self.args.device.isdigit() else 0
                device = torch.device(f'cuda:{device_id}')
                props = torch.cuda.get_device_properties(device_id)
                print(f"🚀 Using CUDA:{device_id} ({props.name}, {props.total_memory / (1 << 20):.0f}MiB)")
                return device
            else:
                print("⚠ CUDA not available, falling back to CPU")
                return torch.device('cpu')

    def load_models(self, device):
        """Load PGSPN models"""
        print("\n📦 Building Models...")

        # Build models using configurations
        model_stage1 = MODEL_BUILDER.build(basic_cfg['model_stage1_cfg'])
        model_stage2 = MODEL_BUILDER.build(basic_cfg['model_stage2_cfg'])

        # Load checkpoints
        print(f"  Loading Stage 1: {self.args.stage1_checkpoint}")
        checkpoint = torch.load(self.args.stage1_checkpoint, map_location='cpu')
        model_stage1.load_state_dict(checkpoint['model'])

        print(f"  Loading Stage 2: {self.args.stage2_checkpoint}")
        checkpoint = torch.load(self.args.stage2_checkpoint, map_location='cpu')
        model_stage2.load_state_dict(checkpoint['model'])

        # Move to device and set eval mode
        model_stage1.to(device).eval()
        model_stage2.to(device).eval()

        return model_stage1, model_stage2

    def preprocess_images(self, img_path, sparse_path):
        """Preprocess input images"""
        # Load images
        img = Image.open(img_path)
        sparse = Image.open(sparse_path)

        # Resize to target size
        img = img.resize((self.input_width, self.input_height), Image.BICUBIC)
        sparse = sparse.resize((self.input_width, self.input_height), Image.NEAREST)

        # Convert to numpy
        img_np = np.array(img).astype(np.float32) / 255.0
        sparse_np = np.array(sparse).astype(np.float32) / self.args.depth_scaling

        # Clean sparse depth (remove invalid values)
        sparse_np = np.where(np.isinf(sparse_np), 0, sparse_np)
        sparse_np = np.where(np.isnan(sparse_np), 0, sparse_np)

        # Prepare tensors
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()
        img_normalized = self.normalizer(img_tensor).unsqueeze(0)

        # IMPORTANT: sparse_input is just 1 channel (the sparse depth map)
        sparse_input = torch.from_numpy(sparse_np).float().unsqueeze(0).unsqueeze(0)

        # Original image for saving
        origin = torch.from_numpy(img_np.transpose(2, 0, 1)).float().unsqueeze(0)

        return img_normalized, sparse_input, origin

    def apply_sparse_colormap(self, sparse_data, colormap_name='Spectral_r', dilation_size=5):
        """Apply colormap to sparse depth with dilation for better visualization

        Args:
            sparse_data: Sparse depth data
            colormap_name: Name of colormap to use
            dilation_size: Size of dilation kernel to make points more visible
        """
        import cv2

        # Create mask for valid (non-zero) depth values
        valid_mask = sparse_data > 0

        # Normalize only valid values using percentile
        data_clean = np.copy(sparse_data)
        if np.any(valid_mask):
            vmin = np.percentile(sparse_data[valid_mask], 1)
            vmax = np.percentile(sparse_data[valid_mask], 99)
        else:
            vmin, vmax = 0, 1

        # Avoid division by zero
        if vmax <= vmin:
            normalized = np.zeros_like(data_clean)
        else:
            normalized = (data_clean - vmin) / (vmax - vmin)
            normalized = np.clip(normalized, 0, 1)

        # Apply colormap
        if colormap_name == 'Spectral_r':
            cmap = plt.colormaps.get_cmap('Spectral_r')
        else:
            cmap = cm.get_cmap(colormap_name)

        # Apply colormap to normalized data
        colored = cmap(normalized)
        colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)

        # Create white background
        result = np.ones_like(colored_rgb) * 255  # White background

        # Dilate the valid mask to make points more visible
        if dilation_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))

            # Create a dilated version of each color channel
            for c in range(3):
                channel = np.zeros_like(sparse_data, dtype=np.uint8)
                channel[valid_mask] = colored_rgb[valid_mask, c]
                dilated_channel = cv2.dilate(channel, kernel, iterations=1)

                # Only update where dilated but preserve original valid points
                dilated_mask = dilated_channel > 0
                result[dilated_mask, c] = dilated_channel[dilated_mask]
        else:
            # Without dilation, just copy colored values where valid
            result[valid_mask] = colored_rgb[valid_mask]

        return result

    def apply_colormap(self, data, colormap_name, vmin=None, vmax=None, crop_pixels=10, inverse=False):
        """Apply a colormap to grayscale data with optional cropping

        Args:
            data: Input grayscale data
            colormap_name: Name of matplotlib colormap
            vmin: Minimum value for normalization
            vmax: Maximum value for normalization
            crop_pixels: Number of pixels to crop from each edge (default: 10)
            inverse: If True, invert the colormap (close=hot, far=cool)
        """
        # Create a mask for valid (non-zero) values
        valid_mask = data > 0

        # Handle invalid values
        data_clean = np.copy(data)
        data_clean[~valid_mask] = 0.0001  # Small value instead of 0

        # Handle NaN and Inf values
        finite_mask = np.isfinite(data_clean) & valid_mask
        if np.any(finite_mask):
            max_depth_value = np.max(data_clean[finite_mask])
        else:
            max_depth_value = 1.0

        # Replace NaN/Inf with max depth
        nan_inf_mask = ~np.isfinite(data_clean)
        if np.any(nan_inf_mask):
            data_clean[nan_inf_mask] = max_depth_value

        # Normalize data to 0-1 range
        if vmin is None:
            # Use robust percentile-based normalization
            if np.any(valid_mask):
                vmin = np.percentile(data_clean[valid_mask], 1)
            else:
                vmin = 0
        if vmax is None:
            if np.any(valid_mask):
                vmax = np.percentile(data_clean[valid_mask], 99)
            else:
                vmax = max_depth_value

        # Avoid division by zero
        if vmax <= vmin:
            normalized = np.zeros_like(data_clean)
        else:
            normalized = (data_clean - vmin) / (vmax - vmin)
            normalized = np.clip(normalized, 0, 1)

        # Apply inverse if requested (for depth: close=hot, far=cool)
        if inverse:
            normalized = 1.0 - normalized
            normalized[~valid_mask] = 0  # Keep invalid pixels black

        # Set invalid regions to 0 (black)
        normalized[~valid_mask] = 0

        # Apply colormap
        if colormap_name == 'Spectral_r':
            # Use matplotlib's get_cmap for better compatibility
            cmap = plt.colormaps.get_cmap('Spectral_r')
        else:
            cmap = cm.get_cmap(colormap_name)
        colored = cmap(normalized)

        # Convert to uint8 RGB
        colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)

        # Crop the edges if requested
        if crop_pixels > 0:
            h, w = colored_rgb.shape[:2]
            # Ensure we don't crop too much
            if h > 2 * crop_pixels and w > 2 * crop_pixels:
                colored_rgb = colored_rgb[crop_pixels:h-crop_pixels, crop_pixels:w-crop_pixels]

        return colored_rgb

    def save_outputs(self, outputs, img_name, output_dir):
        """Save all output images including colormap versions"""
        # Final depth
        depth = outputs['depth'].cpu().numpy().squeeze()
        depth_scaled = (depth * self.args.depth_scaling).astype(np.uint16)
        Image.fromarray(depth_scaled).save(
            os.path.join(output_dir, 'depth_output', f'{img_name}.png')
        )

        # Save depth with Spectral_r colormap
        if self.args.save_colormap:
            # Always use percentile-based normalization for natural visualization
            depth_colored = self.apply_colormap(depth, 'Spectral_r',
                                               crop_pixels=self.args.colormap_crop,
                                               inverse=self.args.colormap_inverse)
            Image.fromarray(depth_colored).save(
                os.path.join(output_dir, 'depth_colormap', f'{img_name}.png')
            )

        # Enhanced image
        enhanced = outputs['enhanced'].cpu().numpy().squeeze()
        if enhanced.ndim == 3:
            enhanced = enhanced.transpose(1, 2, 0)
        enhanced = (np.clip(enhanced, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(enhanced).save(
            os.path.join(output_dir, 'enhanced_output', f'{img_name}.png')
        )

        # Transmission maps
        trans = outputs['transmission'].cpu().numpy().squeeze()
        for i, channel in enumerate(['red', 'green', 'blue']):
            trans_channel = trans[i]
            trans_scaled = (trans_channel * self.args.depth_scaling).astype(np.uint16)
            Image.fromarray(trans_scaled).save(
                os.path.join(output_dir, f'transmission_{channel}', f'{img_name}.png')
            )

            # Save transmission with turbo colormap
            if self.args.save_colormap:
                # No normalization for transmission - use direct 0-1 range
                trans_colored = self.apply_colormap(trans_channel, 'turbo',
                                                   vmin=0, vmax=1,  # Fixed range, no normalization
                                                   crop_pixels=self.args.colormap_crop,
                                                   inverse=self.args.colormap_inverse)
                Image.fromarray(trans_colored).save(
                    os.path.join(output_dir, f'transmission_{channel}_colormap', f'{img_name}.png')
                )

        # Pseudo depth
        pseudo = outputs['pseudo_depth'].cpu().numpy().squeeze()
        pseudo_scaled = (pseudo * self.args.depth_scaling).astype(np.uint16)
        Image.fromarray(pseudo_scaled).save(
            os.path.join(output_dir, 'pseudo_depth', f'{img_name}.png')
        )

        # Save pseudo depth with Spectral_r colormap
        if self.args.save_colormap:
            # Always use percentile-based normalization
            pseudo_colored = self.apply_colormap(pseudo, 'Spectral_r',
                                                crop_pixels=self.args.colormap_crop,
                                                inverse=self.args.colormap_inverse)
            Image.fromarray(pseudo_colored).save(
                os.path.join(output_dir, 'pseudo_depth_colormap', f'{img_name}.png')
            )

        # Ambient light
        ambient = outputs['ambient_light'].cpu().numpy().squeeze()
        if ambient.ndim == 3:
            ambient = ambient.transpose(1, 2, 0)
        ambient = (np.clip(ambient, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(ambient).save(
            os.path.join(output_dir, 'ambient_light', f'{img_name}.png')
        )

        # Input image
        origin = outputs['origin'].cpu().numpy().squeeze()
        if origin.ndim == 3:
            origin = origin.transpose(1, 2, 0)
        origin = (np.clip(origin, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(origin).save(
            os.path.join(output_dir, 'input', f'{img_name}.png')
        )

        # Sparse depth colormap (if available)
        if self.args.save_colormap and 'sparse_depth' in outputs:
            sparse_depth = outputs['sparse_depth'].cpu().numpy().squeeze()
            # Apply special colormap for sparse depth with dilation
            sparse_colored = self.apply_sparse_colormap(sparse_depth, 'Spectral_r', dilation_size=3)
            Image.fromarray(sparse_colored).save(
                os.path.join(output_dir, 'sparse_depth_colormap', f'{img_name}.png')
            )

    def run(self):
        """Main inference pipeline"""
        print("="*50)
        print("PGSPN Inference - Corrected Version")
        print("="*50)
        print(f"Input images: {self.args.input_img_folder}")
        print(f"Sparse depth: {self.args.input_sparse_folder}")
        print(f"Output: {self.args.output_folder}")
        if self.args.save_colormap:
            print("🎨 Colormap visualization: ENABLED")
            print("  - Depth/Pseudo-depth: Spectral_r colormap")
            print("  - Transmission maps: turbo colormap")
            print("  - Using percentile-based normalization (1%-99%)")
            if self.args.colormap_inverse:
                print("  - Inverse mode: Close=Hot, Far=Cool")
            if self.args.colormap_crop > 0:
                print(f"  - All colormaps: Cropped by {self.args.colormap_crop}px on each edge")
                print(f"  - Original size: 576x960 → Colormap size: {576-2*self.args.colormap_crop}x{960-2*self.args.colormap_crop}")
        print("="*50)

        # Setup device
        device = self.setup_device()

        # Load models
        model_stage1, model_stage2 = self.load_models(device)

        # Get image list
        import glob
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(self.args.input_img_folder, ext)))
        image_files.sort()

        if not image_files:
            raise ValueError(f"No images found in {self.args.input_img_folder}")

        print(f"\n📂 Found {len(image_files)} images")

        # Create output directories
        print("📁 Creating output directories")
        os.makedirs(self.args.output_folder, exist_ok=True)

        subdirs = [
            'depth_output', 'enhanced_output', 'transmission_red',
            'transmission_green', 'transmission_blue', 'pseudo_depth',
            'ambient_light', 'input'
        ]

        # Add colormap directories if enabled
        if self.args.save_colormap:
            subdirs.extend([
                'depth_colormap', 'pseudo_depth_colormap',
                'transmission_red_colormap', 'transmission_green_colormap',
                'transmission_blue_colormap', 'sparse_depth_colormap'
            ])

        for subdir in subdirs:
            os.makedirs(os.path.join(self.args.output_folder, subdir), exist_ok=True)

        print("\n🚀 Starting Inference...")

        successful = 0
        failed = 0

        with torch.no_grad():
            for img_path in tqdm.tqdm(image_files, desc="Processing"):
                img_name = Path(img_path).stem

                # Find corresponding sparse depth
                sparse_candidates = [
                    os.path.join(self.args.input_sparse_folder, f'{img_name}.png'),
                    os.path.join(self.args.input_sparse_folder, f'{img_name}.jpg'),
                ]

                sparse_path = None
                for candidate in sparse_candidates:
                    if os.path.exists(candidate):
                        sparse_path = candidate
                        break

                if sparse_path is None:
                    print(f"\n  ⚠ No sparse depth for {img_name}, skipping...")
                    failed += 1
                    continue

                try:
                    # Preprocess
                    img_normalized, sparse_input, origin = self.preprocess_images(img_path, sparse_path)

                    # Move to device
                    img_normalized = img_normalized.to(device)
                    sparse_input = sparse_input.to(device)
                    origin = origin.to(device)

                    # Stage 1: Transmission estimation
                    input_stage1 = {
                        'rgb': img_normalized,
                        'sparse_input': sparse_input,  # 1 channel
                        'origin': origin,
                        'step': 'stage1'
                    }

                    stage1_output = model_stage1(input_stage1)

                    # Extract Stage 1 outputs
                    transmission = stage1_output['transmission']
                    ambient_light = stage1_output['ambient_light']
                    restoration = stage1_output['restoration']
                    corrected_depth = stage1_output['corrected_depth']
                    best_corrected_depth = stage1_output['best_corrected_depth']

                    # Stage 2: Depth completion with PGPS
                    input_stage2 = {
                        'rgb': img_normalized,
                        'sparse_input': sparse_input,  # Same 1 channel
                        'origin': origin,
                        'transmission': transmission,
                        'corrected_depth': corrected_depth,
                        'best_corrected_depth': best_corrected_depth,
                    }

                    stage2_output = model_stage2(input_stage2)
                    list_feat_depth = stage2_output['list_feat_depth']

                    # Prepare outputs
                    outputs = {
                        'depth': list_feat_depth[-1],  # Use last iteration
                        'enhanced': restoration,
                        'transmission': transmission,
                        'pseudo_depth': best_corrected_depth,
                        'ambient_light': ambient_light,
                        'origin': origin,
                        'sparse_depth': sparse_input.squeeze(0).squeeze(0)  # Add sparse depth for visualization
                    }

                    # Save outputs
                    self.save_outputs(outputs, img_name, self.args.output_folder)
                    successful += 1

                except Exception as e:
                    print(f"\n  ❌ Error processing {img_name}: {str(e)}")
                    failed += 1
                    continue

        print("\n" + "="*50)
        print(f"✅ Processing complete!")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Results saved to: {self.args.output_folder}")
        print("="*50)


def main():
    parser = argparse.ArgumentParser(description='PGSPN Inference - Corrected Version')

    # Required arguments
    parser.add_argument('--input_img_folder', type=str, required=True,
                        help='Path to input images folder')
    parser.add_argument('--input_sparse_folder', type=str, required=True,
                        help='Path to sparse depth folder')

    # Optional arguments
    parser.add_argument('--output_folder', type=str, default='inference_results',
                        help='Output folder path')
    parser.add_argument('--stage1_checkpoint', type=str,
                        default='checkpoints/transmission_estimator.pth',
                        help='Stage 1 checkpoint path')
    parser.add_argument('--stage2_checkpoint', type=str,
                        default='checkpoints/depth_completion.pth',
                        help='Stage 2 checkpoint path')
    parser.add_argument('--depth_scaling', type=float, default=1000.0,
                        help='Depth scaling factor')
    parser.add_argument('--device', type=str, default='0',
                        help='Device (cpu or cuda:0)')
    parser.add_argument('--save_colormap', action='store_true',
                        help='Save colormap visualization for depth and transmission maps')
    parser.add_argument('--colormap_crop', type=int, default=0,
                        help='Number of pixels to crop from edges in colormap (default: 10, set 0 to disable)')
    parser.add_argument('--colormap_inverse', action='store_true',
                        help='Invert colormap for depth (close=hot, far=cool)')

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.input_img_folder):
        raise ValueError(f"Input image folder not found: {args.input_img_folder}")
    if not os.path.exists(args.input_sparse_folder):
        raise ValueError(f"Input sparse folder not found: {args.input_sparse_folder}")
    if not os.path.exists(args.stage1_checkpoint):
        raise ValueError(f"Stage 1 checkpoint not found: {args.stage1_checkpoint}")
    if not os.path.exists(args.stage2_checkpoint):
        raise ValueError(f"Stage 2 checkpoint not found: {args.stage2_checkpoint}")

    # Run inference
    inferencer = PGSPNInference(args)
    inferencer.run()


if __name__ == '__main__':
    main()