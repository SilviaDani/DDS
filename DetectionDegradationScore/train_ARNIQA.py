import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import wandb
from typing import Dict, Optional, Tuple
import os
from itertools import islice
from scipy.stats import pearsonr
import pandas as pd
import multiprocessing as mp
import random

from ddsrn import create_ddsrn_model
from extractor import load_feature_extractor, FeatureExtractor
from backbones import Backbone
from utils.bin_distribution_visualizer import BinDistributionVisualizer
from dataloader_ARNIQA import create_dynamic_dataloaders

def set_global_seed(seed: int):
    """Set all random seeds for complete reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

class Trainer:
    """
    Trainer for the quality assessment model.
    """

    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        backbone_name: Backbone,
        learning_rate: float = 1e-4,
        checkpoint_dir: Optional[str] = None,
        num_epochs: int = 100,
        yolo_weights_path: str = "yolo11m.pt",
        try_run: bool = False,
        use_online_wandb=True,
        attempt: int = 0,
        batch_size: int = 128,
    ):
        """
        Initialize the trainer with all necessary components.
        """
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.try_run = try_run
        self.use_online_wandb = use_online_wandb
        self.attempt = attempt
        self.batch_size = batch_size
        self.total_epochs = num_epochs
        self.current_epoch = 0
        self.backbone_name = backbone_name

        layer_config = self.backbone_name.config
        layer_indices = layer_config.indices
        feature_channels = layer_config.channels
        weights_path_extractor = (
            yolo_weights_path if self.backbone_name == Backbone.YOLO_V11_M else None
        )

        self.extractor: FeatureExtractor = load_feature_extractor(
            backbone_name=self.backbone_name,
            weights_path=weights_path_extractor,
        ).to(device)

        self.model = create_ddsrn_model(
            feature_channels=feature_channels, layer_indices=layer_indices
        ).to(device)

        self.loss = nn.MSELoss()

        params = list(self.model.parameters())
        self.optimizer = optim.AdamW(
            [
                {
                    "params": params,
                    "lr": learning_rate,
                }
            ],
            weight_decay=1e-3,
        )

        steps_per_epoch = 50 if try_run else len(train_loader)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            max_lr=learning_rate,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.20,
            div_factor=2,
            final_div_factor=1.2,
            three_phase=False,
            anneal_strategy="cos",
        )

        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        """
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        num_batches = 50 if self.try_run else len(self.train_loader)
        train_iterator = (
            islice(self.train_loader, 50) if self.try_run else self.train_loader
        )

        for i, batch in enumerate(
            tqdm(train_iterator, total=num_batches, desc="Training", ncols=120)
        ):
            gt = batch["gt"].to(self.device)
            distorted = batch["distorted"].to(self.device)
            scores = batch["score"].to(self.device)

            gt_features, mod_features = self.extractor.extract_features(
                img_gt=gt, img_mod=distorted
            )

            predictions = self.model(gt_features, mod_features).squeeze()

            batch_preds = predictions.detach().cpu().numpy()
            batch_targets = scores.detach().cpu().numpy()

            all_preds.extend(batch_preds)
            all_targets.extend(batch_targets)

            loss = self.loss(predictions, scores)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            running_loss += loss.item()

        train_correlation, _ = pearsonr(all_preds, all_targets)

        visualizer = BinDistributionVisualizer(n_bins=40, max_score=0.8)
        visualizer.visualize(
            predictions=all_preds,
            epoch=self.current_epoch,
            total_epochs=self.total_epochs,
            output_file=f"{self.checkpoint_dir}/train_log.txt",
        )

        wandb.log(
            {
                "train_correlation": train_correlation,
            }
        )
        
        return {
            "train_loss": running_loss / num_batches,
            "train_correlation": train_correlation,
        }

    @torch.no_grad()
    def validate(self, current_epoch, val_log_file) -> Dict[str, float]:
        """
        Validate model.
        """
        running_loss = 0.0
        all_preds = []
        all_targets = []

        num_batches = 15 if self.try_run else len(self.val_loader)
        val_iterator = islice(self.val_loader, 15) if self.try_run else self.val_loader

        self.model.eval()

        for batch_idx, batch in enumerate(tqdm(
            val_iterator, total=num_batches, desc="Validating", ncols=120
        )):
            gt = batch["gt"].to(self.device)
            distorted = batch["distorted"].to(self.device)
            scores = batch["score"].to(self.device)

            gt_features, mod_features = self.extractor.extract_features(
                img_gt=gt, img_mod=distorted
            )

            predictions = self.model(gt_features, mod_features).squeeze()

            loss = self.loss(predictions, scores)
            running_loss += loss.item()

            batch_preds = predictions.cpu().numpy()
            batch_targets = scores.cpu().numpy()

            all_preds.extend(batch_preds)
            all_targets.extend(batch_targets)

        val_correlation, _ = pearsonr(all_preds, all_targets)

        epoch_mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))

        visualizer = BinDistributionVisualizer(n_bins=40, max_score=0.8)
        visualizer.visualize(
            predictions=all_preds,
            epoch=current_epoch,
            total_epochs=self.total_epochs,
            output_file=val_log_file,
        )

        metrics = {
            "val_loss": running_loss / num_batches,
            "val_mean_pred": np.mean(all_preds),
            "val_std_pred": np.std(all_preds),
            "val_correlation": val_correlation,
            "val_mae": epoch_mae,
        }

        wandb.log(
            {
                "val_correlation": val_correlation,
                "val_mae": epoch_mae,
            }
        )

        return metrics

    @torch.no_grad()
    def compute_test_metrics(self) -> Tuple[float, float]:
        """
        Compute overall Pearson correlation and MAE for test split.
        
        Returns:
            Tuple of (pearson_correlation, mae)
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        num_batches = 10 if self.try_run else len(self.test_loader)
        test_iterator = islice(self.test_loader, 10) if self.try_run else self.test_loader

        for batch_idx, batch in enumerate(tqdm(test_iterator, total=num_batches, desc="Testing", ncols=120)):
            gt = batch["gt"].to(self.device)
            distorted = batch["distorted"].to(self.device)
            scores = batch["score"].to(self.device)

            gt_features, mod_features = self.extractor.extract_features(
                img_gt=gt, img_mod=distorted
            )

            predictions = self.model(gt_features, mod_features).squeeze()

            batch_preds = predictions.cpu().numpy()
            batch_targets = scores.cpu().numpy()

            all_preds.extend(batch_preds)
            all_targets.extend(batch_targets)

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        pearson_corr, _ = pearsonr(all_preds, all_targets)
        mae = np.mean(np.abs(all_preds - all_targets))

        return pearson_corr, mae

    def save_test_metrics_to_csv(self, pearson_corr: float, mae: float) -> None:
        """
        Save overall test metrics to CSV file.
        
        Args:
            pearson_corr: Pearson correlation coefficient
            mae: Mean Absolute Error
        """
        if not self.checkpoint_dir:
            return

        csv_path = self.checkpoint_dir / "test_metrics.csv"
        
        metrics_data = {
            'metric': ['pearson_correlation', 'mae', 'num_samples', 'attempt', 'backbone', 'timestamp'],
            'value': [
                pearson_corr, 
                mae,
                len(self.test_loader.dataset), 
                self.attempt, 
                self.backbone_name.value,
                pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(csv_path, index=False)

        text_path = self.checkpoint_dir / "test_metrics.txt"
        with open(text_path, 'w') as f:
            f.write("TEST SPLIT OVERALL METRICS\n")
            f.write("==========================\n")
            f.write(f"Pearson Correlation: {pearson_corr:.6f}\n")
            f.write(f"MAE: {mae:.6f}\n")
            f.write(f"Number of samples: {len(self.test_loader.dataset)}\n")
            f.write(f"Backbone: {self.backbone_name.value}\n")
            f.write(f"Attempt: {self.attempt}\n")
            f.write(f"Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    def save_checkpoint(
        self, epoch: int, metrics: Dict[str, float], is_best: bool = False
    ) -> None:
        """
        Save model checkpoint.
        """
        if not self.checkpoint_dir:
            return

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
        }

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)

    def train(
        self,
        num_epochs: int,
        early_stopping_patience: int = 15,
    ) -> None:
        """
        Complete training loop with validation and early stopping.
        """
        wandb.init(
            project="ARNIQA_DDSRN",
            mode="offline" if (self.try_run or not self.use_online_wandb) else "online",
            name=f"attempt{self.attempt}",
            config={
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "batch_size": self.train_loader.batch_size,
                "model_type": self.model.__class__.__name__,
                "backbone": self.backbone_name.value,
            },
        )

        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.model.parameters())
        wandb.log(
            {"TrainableParameters": trainable_params, "TotalParameters": total_params}
        )
        wandb.log({"LossObject": self.loss})
        wandb.log({"Batch Size": self.batch_size})

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            current_lr = self.optimizer.param_groups[0]["lr"]

            train_metrics = self.train_epoch()
            val_metrics = self.validate(
                current_epoch=epoch + 1,
                val_log_file=f"{self.checkpoint_dir}/val_log.txt",
            )

            current_lr = self.optimizer.param_groups[0]["lr"]
            wandb.log({"learning_rate": current_lr, **train_metrics, **val_metrics})

            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                patience_counter = 0
                self.save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                break

        pearson_corr, mae = self.compute_test_metrics()
        
        self.save_test_metrics_to_csv(pearson_corr, mae)
        
        wandb.log({
            "test_pearson_correlation": pearson_corr,
            "test_mae": mae,
        })

        wandb.finish()


def create_dataloaders_for_coco2017_splits(dataset_root: str, batch_size: int, backbone_name: Backbone, **kwargs):
    """
    Create dataloaders for COCO2017 style split folders (train2017, val2017, test2017).
    
    Args:
        dataset_root: Root directory containing train2017, val2017, test2017 subfolders
        batch_size: Batch size for dataloaders
        backbone_name: Backbone model name
        **kwargs: Additional arguments for create_dynamic_dataloaders
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    dataset_path = Path(dataset_root)
    
    train_path = dataset_path / "train2017"
    val_path = dataset_path / "val2017" 
    test_path = dataset_path / "test2017"
    
    if train_path.exists() and val_path.exists() and test_path.exists():
        train_loader, _, _ = create_dynamic_dataloaders(
            dataset_root=str(train_path),
            batch_size=batch_size,
            backbone_name=backbone_name,
            **kwargs
        )
        
        _, val_loader, _ = create_dynamic_dataloaders(
            dataset_root=str(val_path),
            batch_size=batch_size,
            backbone_name=backbone_name,
            **kwargs
        )
        
        _, _, test_loader = create_dynamic_dataloaders(
            dataset_root=str(test_path),
            batch_size=batch_size,
            backbone_name=backbone_name,
            **kwargs
        )
        
        return train_loader, val_loader, test_loader
    else:
        standard_train = dataset_path / "train"
        standard_val = dataset_path / "val"
        standard_test = dataset_path / "test"
        
        if standard_train.exists() and standard_val.exists() and standard_test.exists():
            train_loader, _, _ = create_dynamic_dataloaders(
                dataset_root=str(standard_train),
                batch_size=batch_size,
                backbone_name=backbone_name,
                **kwargs
            )
            
            _, val_loader, _ = create_dynamic_dataloaders(
                dataset_root=str(standard_val),
                batch_size=batch_size,
                backbone_name=backbone_name,
                **kwargs
            )
            
            _, _, test_loader = create_dynamic_dataloaders(
                dataset_root=str(standard_test),
                batch_size=batch_size,
                backbone_name=backbone_name,
                **kwargs
            )
            
            return train_loader, val_loader, test_loader
        else:
            return create_dynamic_dataloaders(
                dataset_root=dataset_root,
                batch_size=batch_size,
                backbone_name=backbone_name,
                **kwargs
            )


def main():
    """
    Main training script with dynamic distortion dataloaders.
    """
    mp.set_start_method('spawn', force=True)
    
    GPU_ID = 0
    DEVICE = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    DATASET_ROOT = "/andromeda/personal/jdamerini"
    BATCH_SIZE = 128
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    ATTEMPT = "FASTERRCNN_seed"
    DIR = "dynamic_distortions_sr25"
    CHECKPOINT_DIR = f"checkpoints/attempt{ATTEMPT}_{DIR}"
    TRY_RUN = False
    USE_ONLINE_WANDB = True
    BACKBONE = Backbone.FASTERRCNN_MOBILENET_V3_LARGE_FPN

    # Set global seed for complete reproducibility
    set_global_seed(42)
    
    train_loader, val_loader, test_loader = create_dataloaders_for_coco2017_splits(
        dataset_root=DATASET_ROOT,
        batch_size=BATCH_SIZE,
        backbone_name=BACKBONE,
        num_workers=min(4, os.cpu_count() // 2),
        train_distorted_versions=1,
        val_distorted_versions=1,
        test_distorted_versions=1,
        train_no_distortion_prob=0.01,
        val_no_distortion_prob=0.01,
        test_no_distortion_prob=0.01,
        max_distortions=2,
        distortion_levels=8,
        max_distortions_per_category=1,
        crop_size=320,
        seed=42,
    )

    trainer = Trainer(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        checkpoint_dir=CHECKPOINT_DIR,
        num_epochs=NUM_EPOCHS,
        yolo_weights_path="yolo11m.pt",
        try_run=TRY_RUN,
        use_online_wandb=USE_ONLINE_WANDB,
        attempt=ATTEMPT,
        batch_size=BATCH_SIZE,
        backbone_name=BACKBONE,
    )

    trainer.train(num_epochs=NUM_EPOCHS)


if __name__ == "__main__":
    main()