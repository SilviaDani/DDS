import argparse
import json
import os
import sys
import numpy as np
import warnings
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# PyTorch / Vision
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# COCO API
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Ultralytics
from ultralytics import YOLO

# ---------------------------------------------------------
# OPTIONAL IMPORTS: DDSRN SCORER & LPIPS
# ---------------------------------------------------------

try:
    # Add project root to path, assuming this script is inside helpers/
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Change this filename if your scorer file has a different name.
    from ddsrnScorer import ddsrnScorer
    from backbones import Backbone

except ImportError as e:
    ddsrnScorer = None
    Backbone = None
    print(f"Warning: DDSRN scorer imports failed: {e}. DDSRN scores will be skipped.")

try:
    import lpips
except ImportError:
    lpips = None
    print("Warning: 'lpips' library not installed. LPIPS scores will be skipped.")

# Filter warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------
# 1. Detection Logic - Optimized with DataLoader
# ---------------------------------------------------------

class COCOImageDataset(Dataset):
    """Dataset to load images efficiently for batch YOLO inference."""

    def __init__(self, image_files, annotation_file):
        self.image_files = [Path(p) for p in image_files]
        self.filename_to_id = {}

        if annotation_file and os.path.exists(annotation_file):
            with open(annotation_file, "r") as f:
                coco_data = json.load(f)

            self.filename_to_id = {
                img["file_name"]: img["id"]
                for img in coco_data["images"]
            }

    def _get_original_filename(self, filepath):
        name = filepath.name

        if name in self.filename_to_id:
            return name

        # Handles names like:
        # 000000123456_gaussian_noise_1.jpg -> 000000123456.jpg
        parts = name.split("_")
        if len(parts) > 1:
            potential = parts[0] + filepath.suffix
            if potential in self.filename_to_id:
                return potential

        return None

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        original_name = self._get_original_filename(image_path)

        if not original_name:
            return None

        try:
            img = Image.open(image_path).convert("RGB")
            return img, {
                "image_id": self.filename_to_id[original_name],
                "path": str(image_path),
            }
        except Exception:
            return None


def coco_collate_fn(batch):
    """Custom collate to handle PIL images and filter invalid samples."""
    batch = [b for b in batch if b is not None]

    if not batch:
        return [], []

    images = [b[0] for b in batch]
    infos = [b[1] for b in batch]

    return images, infos


class COCODetectionGenerator:
    def __init__(self, model_path, device="cuda:0"):
        self.device = device

        print(f"Loading YOLO model: {model_path} to {device}...")
        self.model = YOLO(model_path)

    def yolo_to_coco_format(self, predictions, infos, score_threshold=0.001):
        coco_results = []

        # Map YOLO class index 0-79 to COCO category IDs 1-90
        COCO_MAP = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
            41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
            59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
            80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
        ]

        for pred, info in zip(predictions, infos):
            if pred.boxes is None:
                continue

            boxes = pred.boxes.xyxy.cpu().numpy()
            scores = pred.boxes.conf.cpu().numpy()
            class_ids = pred.boxes.cls.cpu().numpy().astype(int)
            image_id = info["image_id"]

            for box, score, class_id in zip(boxes, scores, class_ids):
                if score < score_threshold:
                    continue

                if class_id >= len(COCO_MAP):
                    continue

                coco_results.append({
                    "image_id": int(image_id),
                    "category_id": int(COCO_MAP[class_id]),
                    "bbox": [
                        float(box[0]),
                        float(box[1]),
                        float(box[2] - box[0]),
                        float(box[3] - box[1]),
                    ],
                    "score": float(score),
                })

        return coco_results

    def generate(
        self,
        image_files,
        annotation_file,
        output_file,
        batch_size=32,
        num_workers=4,
    ):
        dataset = COCOImageDataset(image_files, annotation_file)

        if len(dataset) == 0:
            with open(output_file, "w") as f:
                json.dump([], f)
            return

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=coco_collate_fn,
            pin_memory=True,
        )

        all_results = []

        for images, infos in tqdm(loader, desc="Detection Inference"):
            if not images:
                continue

            results = self.model(
                images,
                verbose=False,
                conf=0.001,
                device=self.device,
            )

            batch_coco = self.yolo_to_coco_format(results, infos)
            all_results.extend(batch_coco)

        with open(output_file, "w") as f:
            json.dump(all_results, f)


# ---------------------------------------------------------
# 2. Perception Logic - LPIPS + DDSRN
# ---------------------------------------------------------

def get_clean_corrupted_pairs(target_files, clean_dir):
    """
    Match corrupted images back to their clean image.

    Supports:
    1. Same filename:
       clean:     000000123456.jpg
       corrupted: 000000123456.jpg

    2. Corruption suffix filename:
       clean:     000000123456.jpg
       corrupted: 000000123456_gaussian_noise_1.jpg
    """
    pairs = []
    clean_candidates = {p.name: p for p in Path(clean_dir).glob("*")}

    for corrupted in target_files:
        corrupted = Path(corrupted)
        clean_name = None

        if corrupted.name in clean_candidates:
            clean_name = corrupted.name
        else:
            parts = corrupted.name.split("_")
            potential = parts[0] + corrupted.suffix

            if potential in clean_candidates:
                clean_name = potential

        if clean_name:
            pairs.append((clean_candidates[clean_name], corrupted))

    return pairs


def compute_perception_sequential(
    target_files,
    clean_dir,
    ddsrn_model_path=None,
    ddsrn_weights_path="yolo11m.pt",
    ddsrn_backbone_name="YOLO_V11_M",
    device="cuda:0",
):
    """
    Computes LPIPS and DDSRN sequentially.

    The output key remains `mean_dds` for compatibility, but it now contains
    the DDSRN scorer output instead of the old detection-based DDS score.
    """
    if not target_files:
        return {}

    pairs = get_clean_corrupted_pairs(target_files, clean_dir)

    if not pairs:
        return {}

    # -------------------------
    # Init LPIPS once
    # -------------------------
    lpips_loss = None
    t_lpips = None

    if lpips is not None:
        lpips_loss = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

        t_lpips = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ])

    # -------------------------
    # Init DDSRN once
    # -------------------------
    ddsrn_model = None
    t_ddsrn = transforms.ToTensor()

    if ddsrnScorer is not None and ddsrn_model_path:
        if Backbone is None:
            print("Warning: Backbone enum unavailable. DDSRN scores will be skipped.")
        else:
            try:
                ddsrn_backbone = getattr(Backbone, ddsrn_backbone_name)

                ddsrn_model = ddsrnScorer(
                    model_path=ddsrn_model_path,
                    backbone=ddsrn_backbone,
                    weights_path=ddsrn_weights_path,
                    device=device,
                ).to(device).eval()

            except AttributeError:
                print(
                    f"Warning: Backbone.{ddsrn_backbone_name} does not exist. "
                    "DDSRN scores will be skipped."
                )
                ddsrn_model = None

            except Exception as e:
                print(f"Warning: DDSRN model failed to load: {e}")
                ddsrn_model = None

    dds_scores = []
    lpips_scores = []

    # -------------------------
    # Sequential evaluation
    # -------------------------
    for clean_path, corr_path in tqdm(pairs, desc="Perception (LPIPS + DDSRN)"):
        try:
            clean_img = Image.open(clean_path).convert("RGB")
            corr_img = Image.open(corr_path).convert("RGB")

            # LPIPS
            if lpips_loss is not None:
                clean_lpips = t_lpips(clean_img).unsqueeze(0).to(device)
                corr_lpips = t_lpips(corr_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    lpips_dist = lpips_loss(clean_lpips, corr_lpips)

                lpips_scores.append(float(lpips_dist.item()))

            # DDSRN
            if ddsrn_model is not None:
                clean_ddsrn = t_ddsrn(clean_img).to(device)
                corr_ddsrn = t_ddsrn(corr_img).to(device)

                with torch.no_grad():
                    dds_score = ddsrn_model(clean_ddsrn, corr_ddsrn)

                dds_scores.append(float(dds_score.detach().cpu().item()))

        except Exception as e:
            print(f"\nWarning: failed perception pair {clean_path} / {corr_path}: {e}")
            continue

    return {
        "mean_dds": float(np.mean(dds_scores)) if dds_scores else 0.0,
        "mean_lpips": float(np.mean(lpips_scores)) if lpips_scores else 0.0,
    }


# ---------------------------------------------------------
# 3. Main Benchmark Runner
# ---------------------------------------------------------

def evaluate_coco_detection(annotation_file, detection_json):
    """
    Run COCO bbox evaluation and return mAP/AP50.
    """
    try:
        coco_gt = COCO(annotation_file)

        with open(detection_json, "r") as f:
            detections = json.load(f)

        if len(detections) == 0:
            print("No detections found.")
            return {
                "mAP": 0.0,
                "AP50": 0.0,
            }

        coco_dt = coco_gt.loadRes(detection_json)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return {
            "mAP": float(coco_eval.stats[0]),
            "AP50": float(coco_eval.stats[1]),
        }

    except Exception as e:
        print(f"COCO eval failed: {e}")
        return {
            "mAP": 0.0,
            "AP50": 0.0,
        }


def find_image_files(directory):
    directory = Path(directory)

    return sorted(
        list(directory.glob("*.jpg"))
        + list(directory.glob("*.jpeg"))
        + list(directory.glob("*.png"))
    )


def run_benchmark(args):
    # YOLO model is used only for detection/mAP evaluation.
    generator = COCODetectionGenerator(args.model, args.device)

    final_metrics = {
        "clean": {},
        "corruptions": {},
    }

    clean_files = find_image_files(args.clean_dir)

    # -------------------------
    # Clean Evaluation
    # -------------------------
    if not args.skip_clean:
        print("\n=== Clean Evaluation ===")

        clean_json = "temp_clean_dets.json"

        generator.generate(
            clean_files,
            args.ann_file,
            clean_json,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )

        final_metrics["clean"] = evaluate_coco_detection(
            args.ann_file,
            clean_json,
        )

        if os.path.exists(clean_json):
            os.remove(clean_json)

    # -------------------------
    # Corruption Evaluation
    # -------------------------
    corruptions = [args.corruption] if args.corruption else [
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "snow",
        "frost",
        "fog",
        "brightness",
        "contrast",
        "elastic_transform",
        "pixelate",
        "jpeg_compression",
    ]

    for corr in corruptions:
        final_metrics["corruptions"][corr] = {}

        print(f"\n>>> Processing {corr}")

        for sev in [1, 2, 3, 4, 5]:
            print(f"Severity {sev}:", end=" ")

            if args.organization == "filename":
                target_files = list(Path(args.corrupted_dir).glob(f"*_{corr}_{sev}.*"))
            else:
                target_files = list((Path(args.corrupted_dir) / corr / str(sev)).glob("*"))

            if not target_files:
                print("No files found.")
                continue

            stats = {}

            # -------------------------
            # 1. mAP Evaluation
            # -------------------------
            temp_json = f"temp_{corr}_{sev}.json"

            generator.generate(
                target_files,
                args.ann_file,
                temp_json,
                batch_size=args.batch_size,
                num_workers=args.workers,
            )

            det_stats = evaluate_coco_detection(args.ann_file, temp_json)
            stats.update(det_stats)

            print(f"mAP: {stats.get('mAP', 0.0):.4f}", end=" | ")

            if os.path.exists(temp_json):
                os.remove(temp_json)

            # -------------------------
            # 2. Perception Metrics
            # -------------------------
            if args.compute_perception:
                p_scores = compute_perception_sequential(
                    target_files=target_files,
                    clean_dir=args.clean_dir,
                    ddsrn_model_path=args.ddsrn_model,
                    ddsrn_weights_path=args.ddsrn_weights,
                    ddsrn_backbone_name=args.ddsrn_backbone,
                    device=args.device,
                )

                stats.update(p_scores)

                print(
                    f"LPIPS: {stats.get('mean_lpips', 0.0):.4f} "
                    f"DDSRN: {stats.get('mean_dds', 0.0):.4f}"
                )
            else:
                print("")

            final_metrics["corruptions"][corr][sev] = stats

    with open(args.output, "w") as f:
        json.dump(final_metrics, f, indent=2)

    print(f"\nSaved to {args.output}")


# ---------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Detection model for mAP
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="YOLO model path used for mAP evaluation, e.g. yolo11m.pt",
    )

    parser.add_argument("--clean_dir", type=str, required=True)
    parser.add_argument("--corrupted_dir", type=str, required=True)
    parser.add_argument("--ann_file", type=str, required=True)
    parser.add_argument("--output", type=str, default="results.json")

    parser.add_argument(
        "--organization",
        type=str,
        default="filename",
        choices=["filename", "folder"],
    )

    # Performance args
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="CPU workers for loading images",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="GPU batch size for YOLO mAP inference",
    )

    parser.add_argument("--skip_clean", action="store_true")
    parser.add_argument("--compute_perception", action="store_true")
    parser.add_argument("--corruption", type=str, default=None)

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    # DDSRN args
    parser.add_argument(
        "--ddsrn_model",
        type=str,
        default="checkpoints/attemptSR_30bins_point8_02_coco17complete_320p_sr_subsamp_444/best_model.pt",
        help="DDSRN checkpoint path, e.g. checkpoints/.../best_model.pt",
    )

    parser.add_argument(
        "--ddsrn_weights",
        type=str,
        default="yolo11m.pt",
        help="Backbone weights path used by the DDSRN feature extractor",
    )

    parser.add_argument(
        "--ddsrn_backbone",
        type=str,
        default="YOLO_V11_M",
        help="Backbone enum name, e.g. YOLO_V11_M",
    )

    args = parser.parse_args()

    if "cuda" in args.device and not torch.cuda.is_available():
        print("Warning: CUDA not available, switching to CPU.")
        args.device = "cpu"

    if args.compute_perception and args.ddsrn_model is None:
        print(
            "Warning: --compute_perception was set but --ddsrn_model was not provided. "
            "DDSRN scores will be skipped; LPIPS may still run."
        )

    run_benchmark(args)