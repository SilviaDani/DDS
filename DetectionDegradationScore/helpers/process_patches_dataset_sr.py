#!/usr/bin/env python3
"""
process_patches_dataset_sr.py

Applies SwinIR super-resolution to all extracted image patches from a dataset.
Input images are taken from:   ./unbalanced_dataset_sr/<split>/extracted/
Output images are saved to:    ./unbalanced_dataset_sr/<split>/compressed/

Default model: SwinIR x4 pretrained on DIV2K (requires basicsr).
"""

import os
import cv2
import shutil
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from basicsr.archs.swinir_arch import SwinIR  # Requires basicsr


# Globals for model and device (lazy-loaded once)
model = None
device = None


# ----------------------------- Utility functions ----------------------------- #

def downscale_image(img, scale_factor=4):
    """Downscale an image to simulate degradation."""
    h, w = img.shape[:2]
    new_w, new_h = w // scale_factor, h // scale_factor
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def init_model(scale_factor=4, model_path="./checkpoints/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth"):
    """Initialize and load the SwinIR model."""
    global device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SwinIR(
        upscale=scale_factor,
        in_chans=3,
        img_size=48,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {model_path}.\n"
            "Download the pretrained SwinIR x4 weights and place them under ./checkpoints/"
        )

    pretrained_state = torch.load(model_path, map_location=device)
    if "params" in pretrained_state:
        pretrained_state = pretrained_state["params"]

    model.load_state_dict(pretrained_state, strict=True)
    model = model.to(device)
    model.eval()
    return model


def super_resolve(model, device, img_bgr):
    """Apply SwinIR super-resolution to a single image."""
    img_rgb = img_bgr[:, :, ::-1].astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor).clamp(0, 1)

    output_img = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output_img = (output_img * 255.0).round().astype(np.uint8)
    return output_img[:, :, ::-1]


def process_image_sr(args):
    """Process a single image (downscale → SR → save to compressed)."""
    global model, device
    src_path, extracted_dir, compressed_dir, scale_factor = args

    if model is None:
        model = init_model(scale_factor)

    img = cv2.imread(str(src_path))
    if img is None:
        return False

    # Simulate degradation and super-resolve
    img_downscaled = downscale_image(img, scale_factor)
    img_sr = super_resolve(model, device, img_downscaled)

    # Save output (preserving relative subpath)
    save_path = compressed_dir / src_path.relative_to(extracted_dir)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), img_sr)
    return True


def process_split_sr(base_dir, split_name, scale_factor=4):
    """Process all images for a given dataset split."""
    extracted_dir = Path(base_dir) / split_name / "extracted"
    compressed_dir = Path(base_dir) / split_name / "compressed"

    if not extracted_dir.exists():
        print(f"Warning: {extracted_dir} does not exist. Skipping {split_name}.")
        return

    compressed_dir.mkdir(parents=True, exist_ok=True)

    # Collect images
    image_files = [f for f in extracted_dir.rglob("*.*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    print(f"Found {len(image_files)} images in {extracted_dir}")

    results = []
    for arg in tqdm(
        [(img_path, extracted_dir, compressed_dir, scale_factor) for img_path in image_files],
        desc=f"Processing {split_name} images (SwinIR ×{scale_factor})"
    ):
        results.append(process_image_sr(arg))

    processed = sum(1 for r in results if r)
    skipped = len(results) - processed
    print(f"{split_name} complete → Processed: {processed}, Skipped: {skipped}")


# ----------------------------- Main entry point ----------------------------- #

def main():
    BASE_DIR = "./unbalanced_dataset_sr"
    SPLITS = ["train", "val", "test"]
    SCALE_FACTOR = 4

    print(f"=== SwinIR Super-Resolution Processing ===")
    print(f"Input dataset base: {BASE_DIR}")
    print(f"Scale factor: ×{SCALE_FACTOR}")

    for split in SPLITS:
        process_split_sr(BASE_DIR, split, scale_factor=SCALE_FACTOR)

    print("\nAll splits processed successfully.")
    print(f"Results saved in '{BASE_DIR}/<split>/compressed/'")


if __name__ == "__main__":
    main()
