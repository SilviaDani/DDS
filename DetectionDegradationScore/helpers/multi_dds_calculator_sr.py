import torch
import json
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List
import sys
import os
from ultralytics import YOLO
from datetime import datetime
from itertools import islice
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dds_metric import match_predictions  # Make sure this file is in your project


# Dataset class for GT/SR pairs
class GTSRDataset(Dataset):
    def __init__(self, gt_dir, sr_dir, transform=None):
        self.gt_dir = Path(gt_dir)
        self.sr_dir = Path(sr_dir)
        self.transform = transform or transforms.ToTensor()
        self.image_names = sorted([f.name for f in self.gt_dir.iterdir() if f.is_file()])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        gt_path = self.gt_dir / name
        sr_path = self.sr_dir / name

        gt_image = self.transform(Image.open(gt_path).convert("RGB"))
        sr_image = self.transform(Image.open(sr_path).convert("RGB"))

        return {
            "gt": gt_image,
            "sr": sr_image,
            "name": name,
        }


# Dataloader loader
def create_sr_dataloaders(dataset_root, batch_size):
    dataset_root = Path(dataset_root)
    splits = ["train", "val", "test"]
    dataloaders = {}

    for split in splits:
        gt_dir = dataset_root / split / "extracted"
        sr_dir = dataset_root / split / "compressed"

        if not gt_dir.exists() or not sr_dir.exists():
            dataloaders[split] = None
            continue

        dataset = GTSRDataset(gt_dir, sr_dir)
        dataloaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return dataloaders


# Calculator class
class SRErrorScoreCalculator:
    def __init__(self, model_path: str, device: torch.device):
        self.device = device
        self.yolo_model = YOLO(model_path, verbose=False)
        self.yolo_model.to(device)
        self.yolo_model.model.eval()

    def process_batch(self, gt_images: torch.Tensor, sr_images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            gt_preds = self.yolo_model.predict(gt_images, verbose=False)
            sr_preds = self.yolo_model.predict(sr_images, verbose=False)
            matches = match_predictions(gt_preds, sr_preds)
            ddscores = torch.tensor(
                [match["ddscore"] for match in matches], device=self.device
            )
            return ddscores

    def process_split(
        self,
        dataloader: torch.utils.data.DataLoader,
        output_dir: Path,
        split_name: str,
        try_run: bool,
    ) -> Dict[str, float]:
        num_batches = 3 if try_run else len(dataloader)
        batch_iterator = islice(dataloader, 3) if try_run else dataloader

        scores_dict = {}
        for batch in tqdm(batch_iterator, total=num_batches, desc=f"Processing {split_name}"):
            gt_images = batch["gt"].to(self.device)
            sr_images = batch["sr"].to(self.device)
            names = batch["name"]

            ddscores = self.process_batch(gt_images, sr_images)

            for i, name in enumerate(names):
                scores_dict[name] = float(ddscores[i])

        output_dir.mkdir(parents=True, exist_ok=True)
        scores_file = output_dir / "ddscores.json"
        with open(scores_file, "w") as f:
            json.dump(scores_dict, f, indent=4)

        print(f"Saved scores to: {scores_file}")
        return scores_dict


# Utility
def get_timestamp_dir() -> str:
    return datetime.now().strftime("%Y_%m_%d_%H%M%S")


# Main
def main():
    GPU_ID = 0
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    DATA_ROOT = "unbalanced_dataset_sr"
    ATTEMPT = "01_coco17complete_320p_sr_subsamp_444"
    OUTPUT_ROOT = f"ddscores_analysis/mapping/{ATTEMPT}"
    BATCH_SIZE = 210
    MODEL_PATH = "../yolo11m.pt"  # Adjust to your YOLO model path
    TRY_RUN = False

    timestamp = get_timestamp_dir()
    output_root = Path(OUTPUT_ROOT) / timestamp

    print(f"Using device: {device}")
    print(f"Processing dataset from: {DATA_ROOT}")
    print(f"Saving scores to: {output_root}")
    print(f"Using YOLO model: {MODEL_PATH}")

    calculator = SRErrorScoreCalculator(MODEL_PATH, device)

    dataloaders = create_sr_dataloaders(
        dataset_root=DATA_ROOT,
        batch_size=BATCH_SIZE,
    )

    for split_name, loader in dataloaders.items():
        if loader is not None:
            split_dir = output_root / split_name
            print(f"\nProcessing {split_name} split...")
            scores = calculator.process_split(loader, split_dir, split_name, TRY_RUN)
            print(f"Completed {split_name} split: {len(scores)} images processed")
        else:
            print(f"\nSkipping {split_name} split: no valid dataloader found")

    print("\nAll done!")


if __name__ == "__main__":
    main()
