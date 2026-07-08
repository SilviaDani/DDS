import torch
import torch.nn.functional as F
import torchvision.ops as ops
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import cv2
from typing import List, Tuple, Dict
import random
from PIL import Image
import torchvision.transforms.functional as TF
import os
import sys
import warnings

# --- Yolo, RT-DETR & Metrics ---
from ultralytics import YOLO, RTDETR

try:
    from dds_metric_faster import match_predictions 
except ImportError:
    print("Warning: dds_metric not found. Scoring will fail.")

# --- YOUR REQUIRED LIBRARY ---
try:
    from imagecorruptions import corrupt, get_corruption_names
except ImportError:
    print("Warning: imagecorruptions not found.") 
    def corrupt(img, severity, corruption_name): return img
    def get_corruption_names(subset): return ["gaussian_noise"]

# ==============================================================================
# 0. ENSEMBLE FUSION & HEATMAP UTILITIES
# ==============================================================================
def fuse_ensemble_predictions(boxes_A: torch.Tensor, boxes_B: torch.Tensor, iou_threshold: float = 0.65) -> torch.Tensor:
    """
    Merges bounding box tensors from two models using NMS.
    Expected input format: [N, 6] tensors where columns are (x1, y1, x2, y2, conf, cls)
    """
    # Safe fallback if one or both models detect nothing
    if boxes_A.numel() == 0 and boxes_B.numel() == 0:
        return torch.empty((0, 6), device=boxes_A.device)
    if boxes_A.numel() == 0: return boxes_B
    if boxes_B.numel() == 0: return boxes_A

    # 1. Concatenate all predictions
    all_boxes = torch.cat([boxes_A, boxes_B], dim=0)
    
    # 2. Extract coordinates, scores, and classes
    coords = all_boxes[:, :4]
    scores = all_boxes[:, 4]
    classes = all_boxes[:, 5]

    # 3. Apply Batched NMS (only suppresses overlapping boxes of the SAME class)
    keep_indices = ops.batched_nms(coords, scores, classes, iou_threshold)

    # 4. Return the filtered, unified bounding boxes
    return all_boxes[keep_indices]

def create_gaussian_objectness_map(image_shape, bboxes):
    """
    Generates a 2D Gaussian heatmap.
    image_shape: (H, W)
    bboxes: list of [x_min, y_min, x_max, y_max] relative to the cropped image
    """
    H, W = image_shape
    mask = torch.zeros((1, H, W), dtype=torch.float32)
    
    if len(bboxes) == 0:
        return mask

    y = torch.arange(0, H, dtype=torch.float32).view(H, 1)
    x = torch.arange(0, W, dtype=torch.float32).view(1, W)

    for box in bboxes:
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
        
        if width <= 0 or height <= 0:
            continue
            
        sigma_x = max(width / 6.0, 1.0)
        sigma_y = max(height / 6.0, 1.0)
        
        gaussian = torch.exp(
            -(((x - center_x) ** 2) / (2 * sigma_x ** 2) + 
              ((y - center_y) ** 2) / (2 * sigma_y ** 2))
        )
        mask[0] = torch.max(mask[0], gaussian)
        
    return mask

# ==============================================================================
# 1. DATASET
# ==============================================================================
class DynamicDistortionDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        yolo_light_path: str = "yolo26ft.pt", 
        detr_light_path: str = "rtdetr-l_ft.pt", 
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        seed: int = 42,
        deterministic: bool = False,
    ):
        self.dataset_root = Path(dataset_root)
        self.yolo_light_path = yolo_light_path
        self.detr_light_path = detr_light_path
        self.mean = mean
        self.std = std
        self.seed = seed
        self.deterministic = deterministic
        
        self.detector_yolo = None 
        self.detector_detr = None
        
        try:
            self.corruption_names = get_corruption_names("all")
        except NameError:
            self.corruption_names = []
            
        if "super_resolution" not in self.corruption_names:
            self.corruption_names.append("super_resolution")
        
        self.severities = [1, 2, 3, 4, 5]
        self.n_types = len(self.corruption_names)
        self.n_levels = len(self.severities)
        self.total_variations = (self.n_types * self.n_levels) + 1

        self.image_files = []
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        for root, _, files in os.walk(str(self.dataset_root)):
            for f in files:
                if os.path.splitext(f)[1].lower() in valid_exts:
                    self.image_files.append(Path(root) / f)
        self.image_files.sort()

    def _init_detector(self):
        """Lazy load lightweight ensemble to CPU for workers."""
        if self.detector_yolo is None:
            self.detector_yolo = YOLO(self.yolo_light_path).to('cpu')
        if self.detector_detr is None:
            self.detector_detr = RTDETR(self.detr_light_path).to('cpu')

    def __len__(self):
        return len(self.image_files)

    def apply_super_resolution_distortion(self, image: np.ndarray, severity: int) -> np.ndarray:
        h, w = image.shape[:2]
        if severity == 1:   scale, interp = 2, cv2.INTER_LINEAR
        elif severity == 2: scale, interp = 3, cv2.INTER_LINEAR
        elif severity == 3: scale, interp = 4, cv2.INTER_CUBIC
        elif severity == 4: scale, interp = 4, cv2.INTER_AREA
        elif severity == 5: scale, interp = 8, cv2.INTER_AREA
        else: return image

        h_small, w_small = max(1, h // scale), max(1, w // scale)
        img_small = cv2.resize(image, (w_small, h_small), interpolation=interp)
        img_sr = cv2.resize(img_small, (w, h), interpolation=cv2.INTER_CUBIC)
        return img_sr

    def _get_randomized_distortion_mask(self, bboxes: List[List[float]], patch_w: int, patch_h: int) -> np.ndarray:
        bg_distorted = random.choice([True, False])
        mask = np.ones((patch_h, patch_w), dtype=np.uint8) if bg_distorted else np.zeros((patch_h, patch_w), dtype=np.uint8)
        
        if not bboxes:
            if not bg_distorted: 
                mask.fill(1)
            return mask
            
        has_distortion = bg_distorted
        
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            patch_x1 = int(max(0, x_min))
            patch_y1 = int(max(0, y_min))
            patch_x2 = int(min(patch_w, x_max))
            patch_y2 = int(min(patch_h, y_max))
            
            if patch_x1 < patch_x2 and patch_y1 < patch_y2:
                obj_distorted = random.choice([True, False])
                if obj_distorted:
                    mask[patch_y1:patch_y2, patch_x1:patch_x2] = 1
                    has_distortion = True
                else:
                    mask[patch_y1:patch_y2, patch_x1:patch_x2] = 0
                    
        if not has_distortion:
            mask.fill(1)
            
        return mask

    def __getitem__(self, idx):
        self._init_detector() 

        img_path = self.image_files[idx]
        filename = img_path.name
        try:
            image = cv2.imread(str(img_path))
            if image is None: raise ValueError("NoneType")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

        h_orig, w_orig = image.shape[:2]
        
        if h_orig < 32 or w_orig < 32:
            target_h = max(h_orig, 32)
            target_w = max(w_orig, 32)
            image = cv2.resize(image, (target_w, target_h))
            h_orig, w_orig = image.shape[:2]

        new_h = (h_orig // 32) * 32
        new_w = (w_orig // 32) * 32
        
        start_y = (h_orig - new_h) // 2
        start_x = (w_orig - new_w) // 2
        
        image = image[start_y : start_y + new_h, start_x : start_x + new_w]

        # --- ENSEMBLE DETECTION ON CPU FOR HEATMAP ---
        res_yolo = self.detector_yolo(image, verbose=False)
        res_detr = self.detector_detr(image, verbose=False)
        
        boxes_yolo = res_yolo[0].boxes.data.cpu() if len(res_yolo) > 0 else torch.empty((0, 6))
        boxes_detr = res_detr[0].boxes.data.cpu() if len(res_detr) > 0 else torch.empty((0, 6))
        
        fused_boxes = fuse_ensemble_predictions(boxes_yolo, boxes_detr, iou_threshold=0.65)
        
        detected_bboxes = []
        if fused_boxes.numel() > 0:
            for box in fused_boxes[:, :4].numpy():
                detected_bboxes.append([box[0], box[1], box[2], box[3]])

        # 1. Generate Heatmap
        heatmap_tensor = create_gaussian_objectness_map((new_h, new_w), detected_bboxes)

        # 2. Setup Deterministic/Random Sampling
        if self.deterministic:
            variation_id = (idx * 997) % self.total_variations
        else:
            variation_id = random.randint(0, self.total_variations - 1)

        if variation_id == 0:
            severity = 0
            distortion_name = "clean"
            distorted_image = image.copy()
        else:
            adj_id = variation_id - 1
            type_idx = adj_id // self.n_levels
            sev_idx = adj_id % self.n_levels
            
            distortion_name = self.corruption_names[type_idx]
            severity = self.severities[sev_idx]

            if distortion_name == "super_resolution":
                distorted_base = self.apply_super_resolution_distortion(image, severity)
            else:
                try:
                    distorted_base = corrupt(image, severity=severity, corruption_name=distortion_name)
                except Exception:
                    distorted_base = image.copy()

            # 3. Apply Local Distortions Using Detected BBoxes
            mask_1c = self._get_randomized_distortion_mask(detected_bboxes, new_w, new_h)
            mask_3c = np.repeat(mask_1c[:, :, np.newaxis], 3, axis=2)
            distorted_image = np.where(mask_3c == 1, distorted_base, image)

        # Normalize and convert to tensors
        gt_pil = Image.fromarray(image)
        dist_pil = Image.fromarray(distorted_image)

        gt_tensor_01 = TF.to_tensor(gt_pil)
        dist_tensor_01 = TF.to_tensor(dist_pil)

        gt_tensor = TF.normalize(gt_tensor_01, self.mean, self.std)
        dist_tensor = TF.normalize(dist_tensor_01, self.mean, self.std)

        return {
            "gt": gt_tensor,
            "distorted": dist_tensor,
            "gt_01": gt_tensor_01,       
            "dist_01": dist_tensor_01,   
            "heatmap": heatmap_tensor,  
            "score": torch.tensor(-1.0), 
            "distortion_name": distortion_name,
            "severity": severity,
            "path": str(filename)
        }

# ==============================================================================
# 2. BATCH SCORER (ENSEMBLE)
# ==============================================================================
class BatchDDSWrapper:
    def __init__(self, dataloader, yolo_path="yolov8x.pt", detr_path="rtdetr-l.pt", device="cuda:0"):
        self.dataloader = dataloader
        self.device = device
        
        print(f"Initializing Ensemble Batch DDS Scorers on {self.device}...")
        self.yolo = YOLO(yolo_path).to(self.device)
        self.detr = RTDETR(detr_path).to(self.device)
        
        # Warmup
        dummy = np.zeros((32, 32, 3), dtype=np.uint8)
        self.yolo.predict(dummy, device=self.device, verbose=False, half=True)
        self.detr.predict(dummy, device=self.device, verbose=False, half=True)

    def __iter__(self):
        for batch in self.dataloader:
            yield self.process_batch(batch)

    def __len__(self):
        return len(self.dataloader)
    
    def __getattr__(self, name):
        return getattr(self.dataloader, name)

    def process_batch(self, batch):
        gt_01 = batch.pop("gt_01")
        dist_01 = batch.pop("dist_01")
        severities = batch["severity"]
        
        gt_gpu = gt_01.to(self.device, non_blocking=True)
        dist_gpu = dist_01.to(self.device, non_blocking=True)
        
        with torch.no_grad():
            # Clean strict predictions (Conf=0.25)
            yolo_gt = self.yolo.predict(gt_gpu, verbose=False, half=True, conf=0.25)
            detr_gt = self.detr.predict(gt_gpu, verbose=False, half=True, conf=0.25)
            
            # Distorted wide-net predictions (Conf=0.05)
            yolo_dist = self.yolo.predict(dist_gpu, verbose=False, half=True, conf=0.05)
            detr_dist = self.detr.predict(dist_gpu, verbose=False, half=True, conf=0.05)
            
        fused_gt_tensors = []
        fused_dist_tensors = []
        
        for i in range(len(gt_gpu)):
            # Fuse Ground Truth
            gt_y = yolo_gt[i].boxes.data
            gt_d = detr_gt[i].boxes.data
            fused_gt_tensors.append(fuse_ensemble_predictions(gt_y, gt_d))
            
            # Fuse Distorted
            dist_y = yolo_dist[i].boxes.data
            dist_d = detr_dist[i].boxes.data
            fused_dist_tensors.append(fuse_ensemble_predictions(dist_y, dist_d))
        
        # Score the fused boxes
        results = match_predictions(fused_gt_tensors, fused_dist_tensors)
        
        final_scores = []
        for i, res in enumerate(results):
            if severities[i] == 0:
                final_scores.append(0.0)
            else:
                if isinstance(res, dict):
                    score = res.get('ddscore', 0.0)
                else:
                    score = res.item() if hasattr(res, 'item') else float(res)
                final_scores.append(score)
                
        batch["score"] = torch.tensor(final_scores, dtype=torch.float32)
        
        return batch

# ==============================================================================
# 3. LOADER CREATION
# ==============================================================================
def collate_yolo_compatible(batch: List[Dict]) -> Dict:
    gt_list = [item['gt'] for item in batch]
    dist_list = [item['distorted'] for item in batch]
    gt_01_list = [item['gt_01'] for item in batch]
    dist_01_list = [item['dist_01'] for item in batch]
    heatmap_list = [item['heatmap'] for item in batch] 
    
    paths = [item['path'] for item in batch]
    dist_names = [item['distortion_name'] for item in batch]
    severities = [item['severity'] for item in batch]

    max_h = max([t.size(1) for t in gt_list])
    max_w = max([t.size(2) for t in gt_list])

    def pad_and_stack(tensor_list, h_target, w_target):
        padded_tensors = []
        for t in tensor_list:
            c, h, w = t.size() 
            pad_h = h_target - h
            pad_w = w_target - w
            padded = F.pad(t, (0, pad_w, 0, pad_h), value=0)
            padded_tensors.append(padded)
        return torch.stack(padded_tensors)

    return {
        "gt": pad_and_stack(gt_list, max_h, max_w),
        "distorted": pad_and_stack(dist_list, max_h, max_w),
        "gt_01": pad_and_stack(gt_01_list, max_h, max_w),
        "dist_01": pad_and_stack(dist_01_list, max_h, max_w),
        "heatmap": pad_and_stack(heatmap_list, max_h, max_w),
        "path": paths,
        "distortion_name": dist_names,
        "severity": severities
    }

def create_dynamic_dataloader(
    dataset_root: str,
    batch_size: int,
    backbone_name: str,
    num_workers: int = 8, 
    seed: int = 42,
    deterministic: bool = False,
    yolo_heavy_path: str = "yolo26ft.pt", 
    detr_heavy_path: str = "rtdetr-l_ft.pt",
    yolo_light_path: str = "yolo26ft.pt", 
    detr_light_path: str = "rtdetr-l_ft.pt",
    device: str = "cuda:0",
    **kwargs
):
    mean, std = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)

    dataset = DynamicDistortionDataset(
        dataset_root=dataset_root,
        yolo_light_path=yolo_light_path, 
        detr_light_path=detr_light_path,
        mean=mean,
        std=std,
        seed=seed,
        deterministic=deterministic,
    )

    raw_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(not deterministic),
        num_workers=num_workers,
        pin_memory=True, 
        drop_last=(not deterministic),
        collate_fn=collate_yolo_compatible,
        persistent_workers=(num_workers > 0),
        prefetch_factor=8
    )
    
    loader = BatchDDSWrapper(
        raw_loader,
        yolo_path=yolo_heavy_path,
        detr_path=detr_heavy_path,
        device=device
    )
    
    return loader