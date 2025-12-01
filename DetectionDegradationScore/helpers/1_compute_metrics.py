import argparse
import json
import glob
import os
import sys
import numpy as np
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# COCO API
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Torch / Vision
from torch.utils.data import Dataset

# ---------------------------------------------------------
# OPTIONAL IMPORTS (DDS & LPIPS)
# ---------------------------------------------------------
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dds_metric import match_predictions
except ImportError:
    match_predictions = None
    print("Warning: 'dds_metric.py' not found. DDS scores will be skipped.")

try:
    import lpips
except ImportError:
    lpips = None
    print("Warning: 'lpips' library not installed. LPIPS scores will be skipped.")

# Set multiprocessing start method for CUDA
if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

# ---------------------------------------------------------
# 1. Detection Logic
# ---------------------------------------------------------

class COCODetectionGenerator:
    def __init__(self, model_path, device='cuda:0'):
        self.device = device
        self.model_path = model_path
        
    class COCODataset(Dataset):
        def __init__(self, image_files, annotation_file):
            self.image_files = [Path(p) for p in image_files]
            
            # Load annotation mapping
            if annotation_file and os.path.exists(annotation_file):
                with open(annotation_file, 'r') as f:
                    coco_data = json.load(f)
                self.filename_to_id = {img['file_name']: img['id'] for img in coco_data['images']}
            else:
                raise ValueError("Annotation file is required!")
        
        def _get_original_filename(self, filepath):
            name = filepath.name
            if name in self.filename_to_id: return name
            
            # Try splitting: 0000123_noise_5.jpg -> 0000123.jpg
            parts = name.split('_')
            if len(parts) > 1:
                potential = parts[0] + filepath.suffix
                if potential in self.filename_to_id: return potential
            return None

        def __len__(self):
            return len(self.image_files)
        
        def __getitem__(self, idx):
            image_path = self.image_files[idx]
            original_name = self._get_original_filename(image_path)
            
            if not original_name:
                return None # Skip invalid files
                
            return {
                'image_path': str(image_path),
                'image_id': self.filename_to_id[original_name],
                'original_name': original_name
            }

    @staticmethod
    def process_image(item):
        """Worker function for detection inference"""
        if item is None: return []
        
        from ultralytics import YOLO
        
        # Load model inside worker
        model = YOLO(item['model_path'], verbose=False)
        
        try:
            image = Image.open(item['image_path']).convert("RGB")
            # YOLO usually auto-selects GPU, but you can force it if needed using device=...
            results = model(image, verbose=False, conf=0.001)
            return COCODetectionGenerator.yolo_to_coco_format(results, item['image_id'])
        except Exception as e:
            return []

    @staticmethod
    def yolo_to_coco_format(predictions, image_id, score_threshold=0.001):
        coco_results = []
        
        # FIX: Map YOLO 0-79 index to COCO 1-90 Category ID
        COCO_MAP = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
            41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
            59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
            80, 81, 82, 84, 85, 86, 87, 88, 89, 90
        ]
        
        if not predictions: return []
            
        for pred in predictions:
            if pred.boxes is None: continue
            
            boxes = pred.boxes.xyxy.cpu().numpy()
            scores = pred.boxes.conf.cpu().numpy()
            class_ids = pred.boxes.cls.cpu().numpy().astype(int)
            
            for box, score, class_id in zip(boxes, scores, class_ids):
                if score < score_threshold: continue
                if class_id >= len(COCO_MAP): continue
                
                coco_results.append({
                    'image_id': int(image_id),
                    'category_id': int(COCO_MAP[class_id]),
                    'bbox': [float(box[0]), float(box[1]), float(box[2]-box[0]), float(box[3]-box[1])],
                    'score': float(score)
                })
        return coco_results

    def generate(self, image_files, annotation_file, output_file, num_workers=4):
        dataset = self.COCODataset(image_files, annotation_file)
        
        items = []
        for i in range(len(dataset)):
            item = dataset[i]
            if item:
                item['model_path'] = self.model_path
                items.append(item)
        
        all_results = []
        if num_workers > 0:
            with mp.Pool(num_workers) as pool:
                for res in tqdm(pool.imap_unordered(self.process_image, items), total=len(items), desc="Inference"):
                    all_results.extend(res)
        else:
            for item in tqdm(items, desc="Inference"):
                all_results.extend(self.process_image(item))
                
        with open(output_file, 'w') as f:
            json.dump(all_results, f)

# ---------------------------------------------------------
# 2. Perception Logic (DDS & LPIPS)
# ---------------------------------------------------------

def perception_worker(args):
    clean_path, corrupted_path, model_path, lpips_net = args
    results = {'dds': None, 'lpips': None}
    
    import torch
    from torchvision import transforms
    from PIL import Image
    from ultralytics import YOLO
    import lpips
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        clean_img = Image.open(clean_path).convert("RGB")
        corr_img = Image.open(corrupted_path).convert("RGB")
        
        # 1. Compute LPIPS
        if lpips is not None:
            t_lpips = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            c_t = t_lpips(clean_img).unsqueeze(0).to(device)
            n_t = t_lpips(corr_img).unsqueeze(0).to(device)
            
            loss_fn = lpips.LPIPS(net=lpips_net, verbose=False).to(device)
            with torch.no_grad():
                dist = loss_fn(c_t, n_t)
                results['lpips'] = float(dist.item())

        # 2. Compute DDS
        if model_path and match_predictions is not None:
            model = YOLO(model_path, verbose=False) 
            clean_res = model(clean_img, verbose=False)[0]
            corr_res = model(corr_img, verbose=False)[0]
            
            matches = match_predictions(clean_res, corr_res)
            if matches:
                results['dds'] = float(matches[0].get("ddscore", 0.0))
            else:
                results['dds'] = 0.0

    except Exception as e:
        pass
        
    return results

def compute_perception(target_files, clean_dir, model_path, workers=4):
    if not target_files: return {}

    pairs = []
    clean_candidates = {p.name: p for p in Path(clean_dir).glob("*")}
    
    for corrupted in target_files:
        corrupted = Path(corrupted)
        clean_name = None
        
        if corrupted.name in clean_candidates:
            clean_name = corrupted.name
        
        if not clean_name:
            parts = corrupted.name.split('_')
            potential = parts[0] + corrupted.suffix
            if potential in clean_candidates:
                clean_name = potential
        
        if clean_name:
            pairs.append((
                str(clean_candidates[clean_name]), 
                str(corrupted),
                model_path,
                'alex'
            ))
            
    if not pairs:
        print("Warning: Could not pair any corrupted images for perception.")
        return {}
        
    dds_scores = []
    lpips_scores = []
    
    if workers > 0:
        with mp.Pool(workers) as pool:
            for res in tqdm(pool.imap(perception_worker, pairs), total=len(pairs), desc="Perception"):
                if res['dds'] is not None: dds_scores.append(res['dds'])
                if res['lpips'] is not None: lpips_scores.append(res['lpips'])
    else:
        for p in tqdm(pairs, desc="Perception"):
            res = perception_worker(p)
            if res['dds'] is not None: dds_scores.append(res['dds'])
            if res['lpips'] is not None: lpips_scores.append(res['lpips'])

    return {
        'mean_dds': float(np.mean(dds_scores)) if dds_scores else 0.0,
        'mean_lpips': float(np.mean(lpips_scores)) if lpips_scores else 0.0
    }

# ---------------------------------------------------------
# 3. Main Benchmark Runner
# ---------------------------------------------------------

def run_benchmark(args):
    generator = COCODetectionGenerator(args.model, args.device)
    final_metrics = {'clean': {}, 'corruptions': {}}
    
    # --- Clean Evaluation ---
    if not args.skip_clean:
        print("\n=== Clean Evaluation ===")
        clean_files = sorted(list(Path(args.clean_dir).glob("*.jpg")) + list(Path(args.clean_dir).glob("*.png")))
        clean_json = "temp_clean_dets.json"
        
        generator.generate(clean_files, args.ann_file, clean_json, args.workers)
        
        try:
            coco_gt = COCO(args.ann_file)
            coco_dt = coco_gt.loadRes(clean_json)
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            final_metrics['clean'] = {
                'mAP': coco_eval.stats[0],
                'AP50': coco_eval.stats[1]
            }
        except Exception as e:
            print(f"Clean eval failed: {e}")
            final_metrics['clean'] = {'mAP': 0.0}
            
        if os.path.exists(clean_json): os.remove(clean_json)

    # --- Corruption Evaluation ---
    corruptions = [args.corruption] if args.corruption else [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ]
    
    for corr in corruptions:
        final_metrics['corruptions'][corr] = {}
        print(f"\n>>> Processing {corr}")
        
        for sev in [1, 2, 3, 4, 5]:
            print(f"Severity {sev}:", end=" ")
            
            # Find Files
            if args.organization == 'filename':
                target_files = list(Path(args.corrupted_dir).glob(f"*_{corr}_{sev}.*"))
            else:
                target_files = list((Path(args.corrupted_dir) / corr / str(sev)).glob("*"))
                
            if not target_files:
                print("No files found.")
                continue
                
            stats = {}
            
            # mAP Evaluation
            temp_json = f"temp_{corr}_{sev}.json"
            generator.generate(target_files, args.ann_file, temp_json, args.workers)
            
            try:
                coco_gt = COCO(args.ann_file)
                coco_dt = coco_gt.loadRes(temp_json)
                evaluator = COCOeval(coco_gt, coco_dt, 'bbox')
                evaluator.evaluate()
                evaluator.accumulate()
                evaluator.summarize()
                
                stats['mAP'] = evaluator.stats[0]
                stats['AP50'] = evaluator.stats[1]
                print(f"mAP: {evaluator.stats[0]:.4f}", end=" | ")
            except Exception:
                print("Eval Failed (no dets?)", end=" | ")
                stats['mAP'] = 0.0
            
            if os.path.exists(temp_json): os.remove(temp_json)
            
            # Perception Metrics
            if args.compute_perception:
                p_scores = compute_perception(
                    target_files, 
                    args.clean_dir, 
                    args.model,
                    args.workers
                )
                stats.update(p_scores)
                print(f"LPIPS: {stats.get('mean_lpips',0):.4f} DDS: {stats.get('mean_dds',0):.4f}")
            else:
                print("")
                
            final_metrics['corruptions'][corr][sev] = stats

    with open(args.output, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    print(f"\nSaved to {args.output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='YOLO model path')
    parser.add_argument('--clean_dir', type=str, required=True)
    parser.add_argument('--corrupted_dir', type=str, required=True)
    parser.add_argument('--ann_file', type=str, required=True)
    parser.add_argument('--output', type=str, default='results.json')
    parser.add_argument('--organization', type=str, default='filename', choices=['filename', 'folder'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--skip_clean', action='store_true')
    parser.add_argument('--compute_perception', action='store_true', help='Compute LPIPS/DDS')
    parser.add_argument('--corruption', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:0', help='Device (e.g. cuda:0, cpu)')
    
    args = parser.parse_args()
    run_benchmark(args)