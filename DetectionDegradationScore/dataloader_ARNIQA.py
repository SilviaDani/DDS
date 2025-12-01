import os
import torch
import numpy as np
import cv2
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple, Dict, Callable, Optional, List
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from dds_metric import match_predictions
from backbones import Backbone

from ARNIQA.utils.utils_data import (
    distort_images,
    distortion_functions,
    distortion_range,
    resize_crop,
)
from basicsr.archs.swinir_arch import SwinIR

# Global SwinIR models
swinir_models = {}
device = "cuda" if torch.cuda.is_available() else "cpu"

def init_swinir_models():
    """Initialize SwinIR models for x4 and x8 scale factors using weight paths"""
    global swinir_models
    
    # x4 and x8 models available
    model_paths = {
        4: 'checkpoints/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth',
        8: 'checkpoints/001_classicalSR_DIV2K_s48w8_SwinIR-M_x8.pth'
    }
    
    for scale_factor, model_path in model_paths.items():
        if not os.path.exists(model_path):
            continue
            
        try:
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
                upsampler='pixelshuffle',
                resi_connection='1conv'
            )
            
            pretrained_state = torch.load(model_path, map_location=device)
            if 'params' in pretrained_state:
                pretrained_state = pretrained_state['params']
            
            model.load_state_dict(pretrained_state, strict=True)
            model = model.to(device)
            model.eval()
            swinir_models[scale_factor] = model
            
        except Exception:
            continue

# Initialize models at import time
init_swinir_models()

def super_resolve_with_swinir(image: np.ndarray, scale_factor: int) -> np.ndarray:
    """
    Apply super-resolution using SwinIR model.
    """
    try:
        # Convert BGR to RGB, normalize to [0, 1]
        img_rgb = image[:, :, ::-1].astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            output = swinir_models[scale_factor](img_tensor).clamp(0, 1)

        # Convert output tensor to uint8 BGR
        output_img = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output_img = (output_img * 255.0).round().astype(np.uint8)
        return output_img[:, :, ::-1]
        
    except Exception:
        return image

class DDSComputation:
    """
    Handles DDS score computation.
    """
    
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.detector = None
        self._initialized = False
    
    def _initialize_detector(self):
        """Initialization of YOLO detector"""
        if not self._initialized:
            self.detector = YOLO("yolo11m.pt").to(self.device)
            self.detector.eval()
            self._initialized = True
    
    def compute_dds(self, ref_img_tensor, degraded_img_tensor, is_normalized: bool = False):
        """
        Compute DDS score for image pair.
        
        Args:
            is_normalized: If True, images are ImageNet normalized and need denormalization
        """
        # Initialize detector on first use
        if not self._initialized:
            self._initialize_detector()
            
        # Ensure tensors are on the correct device and have batch dimension
        if ref_img_tensor.dim() == 3:
            ref_img_tensor = ref_img_tensor.unsqueeze(0)
        if degraded_img_tensor.dim() == 3:
            degraded_img_tensor = degraded_img_tensor.unsqueeze(0)

        # Only denormalize if images are ImageNet normalized
        if is_normalized:
            imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(ref_img_tensor.device)
            imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(ref_img_tensor.device)
            
            ref_img_tensor = torch.clamp(ref_img_tensor * imagenet_std + imagenet_mean, 0, 1)
            degraded_img_tensor = torch.clamp(degraded_img_tensor * imagenet_std + imagenet_mean, 0, 1)

        # Move to device
        img_ref = ref_img_tensor.to(self.device)
        img_deg = degraded_img_tensor.to(self.device)

        # Run inference
        with torch.no_grad():
            ref_preds = self.detector.predict(img_ref, verbose=False)
            deg_preds = self.detector.predict(img_deg, verbose=False)

        # Extract single image predictions
        ref_result = ref_preds[0]
        deg_result = deg_preds[0]

        # Compute DDS
        batch_results = match_predictions([ref_result], [deg_result])
        dds_score = batch_results[0]['ddscore']

        return dds_score.item() if isinstance(dds_score, torch.Tensor) else dds_score

CUSTOM_DISTORTION_RANGES = {
    "brightness": {
        "brighten": [1.0, 2.0, 3.5, 5.0, 8.0],
        "darken": [0.4, 0.7, 0.85, 0.92, 0.97],
        "meanshift": [-0.8, -1.2, 1.5, 2.0, 2.5],
    },
    "color": {
        "colorsat1": [0.1, -0.3, -1.0, -2.0, -3.0],
        "colorsat2": [3.0, 8.0, 20.0, 35.0, 50.0],
        "colordiff": [5, 12, 25, 40, 60],
        "colorshift": [8, 20, 40, 60, 80],
    },
    "noise": {
        "whitenoise": [0.02, 0.05, 0.1, 0.18, 0.3],
        "multnoise": [0.02, 0.05, 0.1, 0.17, 0.25],
        "impulsenoise": [0.02, 0.05, 0.1, 0.16, 0.22],
        "whitenoiseCC": [0.01, 0.025, 0.06, 0.12, 0.2],
    },
    "contrast": {
        "lincontrchange": [-0.2, -0.4, 0.3, 0.6, 0.9],
        "nonlincontrchange": [0.1, 0.05, 0.015, 0.005, 0.001],
        "highsharpen": [3, 8, 15, 25, 35],
    },
    "spatial": {
        "pixelate": [0.1, 0.25, 0.45, 0.65, 0.8],
        "quantization": [12, 6, 3, 2, 1],
        "jitter": [0.08, 0.2, 0.5, 1.0, 1.8],
        "colorblock": [2, 4, 7, 11, 16],
        "noneccpatch": [2, 6, 12, 20, 30],
    },
    "blur": {
        "gaublur": [0.1, 0.3, 0.8, 1.5, 2.5],
        "lensblur": [1, 2, 3, 4, 5],
        "motionblur": [1, 2, 3, 5, 8],
    },
    "compression": {
        "jpeg": [50, 30, 15, 5, 1],
        "jpeg2000": [10, 25, 60, 120, 250],
    },
    "downscale_sr": {
        "downscale_sr_x4": [4],
        "downscale_sr_x8": [8],
    }
}

DISTORTION_CATEGORIES = {
    "blur": ["lensblur"],
    "color": ["colorsat1"],
    "compression": ["jpeg"],
    "noise": ["multnoise"],
    "brightness": ["meanshift"],
    "spatial": ["pixelate"],
    "contrast": ["lincontrchange"],
    "downscale_sr": ["downscale_sr_x4", "downscale_sr_x8"],
}

def downscale_sr_distortion(image: torch.Tensor, scale_factor: int) -> torch.Tensor:
    """
    Apply combined downscale and superresolution distortion.
    """
    if scale_factor <= 1:
        return image
    
    if image.dim() == 3:
        # Convert tensor to numpy for processing
        image_np = image.cpu().numpy()
        if image_np.shape[0] == 3:
            image_np = image_np.transpose(1, 2, 0)
        image_np = (np.clip(image_np, 0, 1) * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Get original dimensions
        h, w = image_bgr.shape[:2]
        
        # Calculate downscaled dimensions
        down_w = w // scale_factor
        down_h = h // scale_factor
        
        # Ensure minimum size
        down_w = max(down_w, 16)
        down_h = max(down_h, 16)
        
        # Downscale using high-quality interpolation
        downscaled = cv2.resize(image_bgr, (down_w, down_h), interpolation=cv2.INTER_AREA)
        
        # Apply super-resolution using SwinIR
        sr_image = super_resolve_with_swinir(downscaled, scale_factor)
        
        # If SR result doesn't match original size, resize to match
        if sr_image.shape[:2] != (h, w):
            sr_image = cv2.resize(sr_image, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Convert back to RGB
        sr_image_rgb = cv2.cvtColor(sr_image, cv2.COLOR_BGR2RGB)
        
        # Convert back to tensor
        sr_image_float = sr_image_rgb.astype(np.float32) / 255.0
        if sr_image_float.shape[2] == 3:
            sr_image_float = sr_image_float.transpose(2, 0, 1)
        return torch.from_numpy(sr_image_float).to(image.device)
    else:
        return image

def downscale_sr_x4_distortion(image: torch.Tensor, value: float) -> torch.Tensor:
    return downscale_sr_distortion(image, 4)

def downscale_sr_x8_distortion(image: torch.Tensor, value: float) -> torch.Tensor:
    return downscale_sr_distortion(image, 8)

distortion_functions.update({
    "downscale_sr_x4": downscale_sr_x4_distortion,
    "downscale_sr_x8": downscale_sr_x8_distortion,
})

distortion_range.update({
    "downscale_sr_x4": [4],
    "downscale_sr_x8": [8],
})

class DynamicDistortionDataset(Dataset):
    """
    Dataset that generates dynamic distortions and computes DDS scores on-the-fly.
    """

    def __init__(
        self,
        root_path: str,
        split: str,
        preprocess: Optional[Callable] = None,
        max_distortions: int = 2,
        distortion_levels: int = 8,
        apply_preprocess: bool = True,
        max_distortions_per_category: int = 1,
        distortion_categories: Dict[str, List[str]] = None,
        dds_device: str = "cuda:0",
        crop_size: int = 320,
        seed: int = 42,
        num_distorted_versions: int = 1,
        fixed_distortions: bool = False,
        no_distortion_prob: float = 0.01,
    ):
        if split not in ["train", "val", "test"]:
            raise ValueError("Split must be one of: train, val, test")

        self.split_path = Path(root_path)
        self.gt_path = self.split_path
        
        self.max_distortions = max_distortions
        self.distortion_levels = distortion_levels
        self.apply_preprocess = apply_preprocess
        self.max_distortions_per_category = max_distortions_per_category
        self.distortion_categories = distortion_categories or DISTORTION_CATEGORIES
        self.crop_size = crop_size
        
        self.split = split
        self.num_distorted_versions = num_distorted_versions
        self.fixed_distortions = fixed_distortions
        self.no_distortion_prob = no_distortion_prob

        self.seed = seed
        self._set_seed()

        self.dds_computer = DDSComputation(device=dds_device)

        if not self.gt_path.exists():
            raise RuntimeError(f"Ground truth directory not found: {self.gt_path}")

        self.image_names = sorted(
            [
                f
                for f in os.listdir(self.gt_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )

        if not self.image_names:
            raise RuntimeError(f"No valid images found in {self.gt_path}")

        if self.fixed_distortions:
            self._pre_generate_fixed_distortions()

        self.base_transform = transforms.ToTensor()
        self.preprocess = preprocess

    def _set_seed(self):
        """Set random seed for reproducibility"""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

    def _pre_generate_fixed_distortions(self):
        """Pre-generate fixed distortion parameters for consistent val/test sets"""
        self.fixed_distortion_params = {}
        
        original_state = random.getstate()
        random.seed(self.seed)
        
        for img_name in self.image_names:
            self.fixed_distortion_params[img_name] = []
            for version_idx in range(self.num_distorted_versions):
                apply_distortion = random.random() > self.no_distortion_prob
                
                if apply_distortion:
                    force_include_sr = random.random() < 0.25
                    distort_functions, distort_values = self.get_distortion_by_categories_with_sr_probability(force_include_sr)
                    self.fixed_distortion_params[img_name].append({
                        'functions': distort_functions,
                        'values': distort_values,
                        'apply_distortion': True
                    })
                else:
                    self.fixed_distortion_params[img_name].append({
                        'functions': [],
                        'values': [],
                        'apply_distortion': False
                    })
        
        random.setstate(original_state)

    def __len__(self) -> int:
        return len(self.image_names) * self.num_distorted_versions

    def get_item_indexes(self, idx: int) -> Tuple[int, int]:
        image_idx = idx // self.num_distorted_versions
        version_idx = idx % self.num_distorted_versions
        return image_idx, version_idx

    def load_and_crop_image(self, img_path: Path) -> Image.Image:
        """Load image and apply crop"""
        pil_image = Image.open(img_path).convert('RGB')
        cropped_image = resize_crop(pil_image, crop_size=self.crop_size, downscale_factor=1)
        return cropped_image

    def get_random_distortion_value(self, category: str, distortion_name: str) -> float:
        if (category in CUSTOM_DISTORTION_RANGES and 
            distortion_name in CUSTOM_DISTORTION_RANGES[category]):
            
            possible_values = CUSTOM_DISTORTION_RANGES[category][distortion_name]
            
            if self.distortion_levels and self.distortion_levels < len(possible_values):
                possible_values = possible_values[:self.distortion_levels]
            
            return random.choice(possible_values)
        else:
            if distortion_name in distortion_range:
                levels = distortion_range[distortion_name]
                if self.distortion_levels and self.distortion_levels < len(levels):
                    levels = levels[:self.distortion_levels]
                return random.choice(levels)
            else:
                return random.uniform(0.1, 0.9)

    def get_distortion_by_categories(self) -> tuple:
        available_categories = list(self.distortion_categories.keys())
        num_categories_to_use = min(self.max_distortions, len(available_categories))
        
        include_sr = random.random() < 0.25
        
        if include_sr and "downscale_sr" in available_categories:
            selected_categories = ["downscale_sr"]
            other_categories = [cat for cat in available_categories if cat != "downscale_sr"]
            if num_categories_to_use > 1:
                additional_categories = random.sample(other_categories, num_categories_to_use - 1)
                selected_categories.extend(additional_categories)
        else:
            selected_categories = random.sample(available_categories, num_categories_to_use)
        
        distort_functions = []
        distort_values = []
        
        for category in selected_categories:
            available_distortions = self.distortion_categories[category]
            selected_distortion = random.choice(available_distortions)
            
            if selected_distortion in distortion_functions:
                distort_func = distortion_functions[selected_distortion]
                distort_value = self.get_random_distortion_value(category, selected_distortion)
                
                distort_functions.append(distort_func)
                distort_values.append(distort_value)
        
        return distort_functions, distort_values

    def get_distortion_by_categories_with_sr_probability(self, force_include_sr: bool) -> tuple:
        available_categories = list(self.distortion_categories.keys())
        num_categories_to_use = min(self.max_distortions, len(available_categories))
        
        if force_include_sr and "downscale_sr" in available_categories:
            selected_categories = ["downscale_sr"]
            other_categories = [cat for cat in available_categories if cat != "downscale_sr"]
            if num_categories_to_use > 1:
                additional_categories = random.sample(other_categories, num_categories_to_use - 1)
                selected_categories.extend(additional_categories)
        else:
            selected_categories = random.sample(available_categories, num_categories_to_use)
        
        distort_functions = []
        distort_values = []
        
        for category in selected_categories:
            available_distortions = self.distortion_categories[category]
            selected_distortion = random.choice(available_distortions)
            
            if selected_distortion in distortion_functions:
                distort_func = distortion_functions[selected_distortion]
                distort_value = self.get_random_distortion_value(category, selected_distortion)
                
                distort_functions.append(distort_func)
                distort_values.append(distort_value)
        
        return distort_functions, distort_values

    def apply_dynamic_distortion(self, image: torch.Tensor, distortion_params: Optional[Dict] = None) -> torch.Tensor:
        if distortion_params is None:
            apply_distortion = random.random() > self.no_distortion_prob
            if apply_distortion:
                distort_functions, distort_values = self.get_distortion_by_categories()
            else:
                distort_functions, distort_values = [], []
        else:
            apply_distortion = distortion_params.get('apply_distortion', True)
            if apply_distortion:
                distort_functions = distortion_params['functions']
                distort_values = distortion_params['values']
            else:
                distort_functions, distort_values = [], []
        
        if distort_functions and distort_values:
            if image.dim() == 3:
                input_image = image.clone()
            else:
                if image.dim() == 4 and image.shape[0] == 1:
                    input_image = image.clone().squeeze(0)
                else:
                    raise ValueError(f"Unexpected image shape: {image.shape}. Expected (3, H, W)")
            if input_image.shape[0] != 3:
                raise ValueError(f"Expected 3-channel image, got {input_image.shape[0]} channels")
            
            try:
                distorted_tensor, _, _ = distort_images(
                    image=input_image,
                    distort_functions=distort_functions,
                    distort_values=distort_values,
                    max_distortions=len(distort_functions),
                    num_levels=self.distortion_levels
                )
                
                if distorted_tensor.dim() != 3:
                    if distorted_tensor.dim() == 4 and distorted_tensor.shape[0] == 1:
                        distorted_tensor = distorted_tensor.squeeze(0)
                    else:
                        raise ValueError(f"Unexpected output shape from distort_images: {distorted_tensor.shape}")
                        
                return distorted_tensor
                
            except Exception as e:
                raise e
        else:
            return image.clone()
    
    def compute_dds_score(self, gt_tensor: torch.Tensor, distorted_tensor: torch.Tensor, is_normalized: bool) -> float:
        """Compute DDS score for the image pair immediately."""
        try:
            dds_score = self.dds_computer.compute_dds(gt_tensor, distorted_tensor, is_normalized)
            return dds_score
        except Exception as e:
            raise RuntimeError(f"DDS computation failed for image pair: {e}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_idx, version_idx = self.get_item_indexes(idx)
        img_name = self.image_names[image_idx]

        try:
            gt_pil = self.load_and_crop_image(self.gt_path / img_name)
            gt_tensor = self.base_transform(gt_pil)

            if gt_tensor.dim() != 3 or gt_tensor.shape[0] != 3:
                raise ValueError(f"Invalid image shape after loading: {gt_tensor.shape}. Expected (3, H, W)")

            if self.fixed_distortions:
                distortion_params = self.fixed_distortion_params[img_name][version_idx]
                distorted_tensor = self.apply_dynamic_distortion(gt_tensor, distortion_params)
            else:
                distorted_tensor = self.apply_dynamic_distortion(gt_tensor)

            if distorted_tensor.dim() != 3 or distorted_tensor.shape[0] != 3:
                raise ValueError(f"Invalid image shape after distortion: {distorted_tensor.shape}. Expected (3, H, W)")

            if self.preprocess and self.apply_preprocess:
                gt_tensor = self.preprocess(gt_tensor)
                distorted_tensor = self.preprocess(distorted_tensor)
                # Images are now normalized - need denormalization for DDS
                is_normalized = True
            else:
                # Images are in [0,1] range - no denormalization needed
                is_normalized = False
            
            quality_score = self.compute_dds_score(gt_tensor, distorted_tensor, is_normalized)

            is_distorted = not torch.equal(gt_tensor, distorted_tensor) if torch.is_tensor(distorted_tensor) else False

            result = {
                "gt": gt_tensor,
                "distorted": distorted_tensor,
                "name": f"{img_name}_v{version_idx}",
                "score": torch.tensor(quality_score, dtype=torch.float32),
                "is_distorted": torch.tensor(is_distorted, dtype=torch.bool),
            }

            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to process image {img_name} (version {version_idx}): {e}")

def worker_init_fn(worker_id):
    """Worker initialization function for DataLoader reproducibility"""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def create_dynamic_dataloaders(
    dataset_root: str,
    batch_size: int,
    backbone_name: Backbone,
    num_workers: int = 4,
    max_distortions: int = 2,
    distortion_levels: int = 8,
    max_distortions_per_category: int = 1,
    dds_device: str = "cuda:0",
    crop_size: int = 320,
    seed: int = 42,
    train_distorted_versions: int = 1,
    val_distorted_versions: int = 1,
    test_distorted_versions: int = 1,
    train_no_distortion_prob: float = 0.01,
    val_no_distortion_prob: float = 0.01,
    test_no_distortion_prob: float = 0.01,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders that use dynamic distortions and compute DDS scores.
    """
    loaders = {}
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if backbone_name in [
        Backbone.VGG_16,
        Backbone.MOBILENET_V3_L,
        Backbone.EFFICIENTNET_V2_M,
        Backbone.FASTERRCNN_MOBILENET_V3_LARGE_FPN,
    ]:
        preprocess_transform = transforms.Compose([
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
        apply_preprocess = True
    elif backbone_name == Backbone.YOLO_V11_M:
        preprocess_transform = None
        apply_preprocess = False
    else:
        raise ValueError(f"Unknown backbone '{backbone_name.value}'")

    split_configs = {
        "train": {
            "num_versions": train_distorted_versions,
            "fixed_distortions": False,
            "shuffle": True,
            "no_distortion_prob": train_no_distortion_prob,
        },
        "val": {
            "num_versions": val_distorted_versions,
            "fixed_distortions": True,
            "shuffle": False,
            "no_distortion_prob": val_no_distortion_prob,
        },
        "test": {
            "num_versions": test_distorted_versions,
            "fixed_distortions": True,
            "shuffle": False,
            "no_distortion_prob": test_no_distortion_prob,
        }
    }

    # Use deterministic seed mapping instead of hash
    split_seeds = {
        'train': seed + 0,
        'val': seed + 1, 
        'test': seed + 2
    }

    for split, config in split_configs.items():
        dataset = DynamicDistortionDataset(
            root_path=dataset_root,
            split=split,
            preprocess=preprocess_transform,
            max_distortions=max_distortions,
            distortion_levels=distortion_levels,
            apply_preprocess=apply_preprocess,
            max_distortions_per_category=max_distortions_per_category,
            distortion_categories=DISTORTION_CATEGORIES,
            dds_device=dds_device,
            crop_size=crop_size,
            seed=split_seeds[split],  # Use deterministic seed mapping
            num_distorted_versions=config["num_versions"],
            fixed_distortions=config["fixed_distortions"],
            no_distortion_prob=config["no_distortion_prob"],
        )

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=config["shuffle"],
            num_workers=num_workers,
            persistent_workers=True,
            pin_memory=True,
            worker_init_fn=worker_init_fn,  # Add worker initialization
        )
    return loaders["train"], loaders["val"], loaders["test"]