import cv2
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
from torchvision.transforms.functional import to_tensor, to_pil_image

from basicsr.archs.swinir_arch import SwinIR  # Requires basicsr

# Globals for model and device
model = None
device = None

def downscale_image(img, scale_factor=4):
    h, w = img.shape[:2]
    new_w, new_h = w // scale_factor, h // scale_factor
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

def init_model(scale_factor=4, model_path='checkpoints/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth'):
    global device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SwinIR(
        upscale=scale_factor,
        in_chans=3,
        img_size=48,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],        # <-- corrected: 6 groups
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],    # <-- corrected: 6 groups
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv'
    )

    pretrained_state = torch.load(model_path, map_location=device)
    if 'params' in pretrained_state:
        pretrained_state = pretrained_state['params']

    model.load_state_dict(pretrained_state, strict=True)  # strict=True to catch errors

    model = model.to(device)
    model.eval()
    return model


def super_resolve(model, device, img_bgr):
    # Convert BGR to RGB, normalize to [0, 1]
    img_rgb = img_bgr[:, :, ::-1].astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor).clamp(0, 1)

    # Convert output tensor to uint8 BGR
    output_img = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output_img = (output_img * 255.0).round().astype(np.uint8)
    return output_img[:, :, ::-1]

def process_image_sr(args):
    global model, device
    src_path, old_extracted_dir, extracted_new_path, compressed_new_path, scale_factor = args

    if model is None:
        model = init_model(scale_factor)

    img = cv2.imread(str(src_path))
    if img is None:
        return False

    extracted_save_path = extracted_new_path / src_path.relative_to(old_extracted_dir)
    extracted_save_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src_path), str(extracted_save_path))

    img_downscaled = downscale_image(img, scale_factor)
    img_sr = super_resolve(model, device, img_downscaled)

    compressed_save_path = compressed_new_path / src_path.relative_to(old_extracted_dir)
    compressed_save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(compressed_save_path), img_sr)

    return True

def process_split_sr(old_base_dir, split_name, new_base_dir, scale_factor=4):
    old_extracted_dir = Path(old_base_dir) / split_name / "extracted"
    new_extracted_dir = Path(new_base_dir) / split_name / "extracted"
    new_compressed_dir = Path(new_base_dir) / split_name / "compressed"

    if not old_extracted_dir.exists():
        print(f"Warning: {old_extracted_dir} does not exist, skipping...")
        return

    image_files = [f for f in old_extracted_dir.rglob("*.*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]

    print(f"Found {len(image_files)} images in {old_extracted_dir}")

    process_args = [
        (img_path, old_extracted_dir, new_extracted_dir, new_compressed_dir, scale_factor)
        for img_path in image_files
    ]

    results = []
    for arg in tqdm(process_args, desc=f"Processing {split_name} images with SwinIR"):
        results.append(process_image_sr(arg))

    processed = sum(1 for r in results if r)
    skipped = len(results) - processed
    print(f"{split_name} complete: Processed = {processed}, Skipped = {skipped}")

def main():
    OLD_BASE_DIR = "/andromeda/personal/jdamerini/unbalanced_dataset_coco2017"
    NEW_BASE_DIR = "unbalanced_dataset_sr_dataset"
    SPLITS = ["train", "val", "test"]
    SCALE_FACTOR = 4

    # Clear new dirs if needed
    for split in SPLITS:
        new_split_dir = Path(NEW_BASE_DIR) / split
        if new_split_dir.exists():
            shutil.rmtree(new_split_dir)
        (new_split_dir / "extracted").mkdir(parents=True, exist_ok=True)
        (new_split_dir / "compressed").mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        process_split_sr(OLD_BASE_DIR, split, NEW_BASE_DIR, scale_factor=SCALE_FACTOR)

if __name__ == "__main__":
    main()
