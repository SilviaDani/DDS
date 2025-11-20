#!/usr/bin/env python3
import cv2
import os
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Extract and preprocess image patches from dataset splits.")
    parser.add_argument(
        "--input_root",
        type=str,
        default="./dataset",
        help="Root directory containing dataset splits (train/, val/). Default: local 'dataset/' folder",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./unbalanced_dataset_sr",
        help="Directory where processed images will be saved. Default: 'unbalanced_dataset_sr/' in current folder.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val","test"],
        help="Dataset splits to process (default: train val test).",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=320,
        help="Target output size (square crop dimension, default: 320).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel workers (default: all CPU cores).",
    )
    return parser.parse_args()


def create_split_directories(base_path, splits):
    """
    Create/clean the directory structure for output dataset splits.
    """
    for split in splits:
        path = Path(base_path) / split / "extracted"
        if path.exists():
            print(f"Cleaning {path}")
            shutil.rmtree(path)
        print(f"Creating {path}")
        path.mkdir(parents=True, exist_ok=True)


def process_image(args):
    """
    Process a single image: resize preserving aspect ratio, then center crop to get target_size x target_size.
    Discards images smaller than the target size.
    """
    image_path, output_path, target_size = args

    # Read image (preserve depth and number of channels, works for 8-bit thermal too)
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return False

    height, width = img.shape[:2]

    # Skip if smaller than target size
    if height < target_size or width < target_size:
        return False

    # Preserve aspect ratio during resize
    aspect_ratio = width / height
    if aspect_ratio > 1:  # Landscape
        new_width = int(target_size * aspect_ratio)
        new_height = target_size
    else:  # Portrait or square
        new_width = target_size
        new_height = int(target_size / aspect_ratio)

    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Center crop
    y_start = (new_height - target_size) // 2
    x_start = (new_width - target_size) // 2
    cropped = resized[y_start : y_start + target_size, x_start : x_start + target_size]

    # Save
    cv2.imwrite(str(output_path), cropped)
    return True


def process_split(input_dir, output_dir, split_name, target_size=320, num_workers=None):
    """
    Process all images from a specific split and save them to the output directory.
    """
    if num_workers is None:
        num_workers = os.cpu_count()

    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Warning: Directory {input_dir} does not exist, skipping...")
        return

    # Collect image files
    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.tiff", "*.bmp"):
        image_files.extend(input_path.glob(ext))

    print(f"Found {len(image_files)} images in {input_dir}")

    # Prepare arguments
    process_args = [
        (
            img_path,
            Path(output_dir) / split_name / "extracted" / img_path.name,
            target_size,
        )
        for img_path in image_files
    ]

    # Parallel processing with progress bar
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(
                executor.map(process_image, process_args),
                total=len(process_args),
                desc=f"Processing {split_name} images",
            )
        )

    # Stats
    processed = sum(1 for r in results if r)
    skipped = len(results) - processed
    print(f"{split_name} split complete:")
    print(f"  Processed: {processed}")
    print(f"  Skipped (too small or invalid): {skipped}")


def main():
    args = parse_args()

    # Prepare output structure
    create_split_directories(args.output_root, args.splits)

    # Process splits
    for split_name in args.splits:
        input_dir = Path(args.input_root) / split_name
        process_split(
            input_dir=input_dir,
            output_dir=args.output_root,
            split_name=split_name,
            target_size=args.target_size,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
