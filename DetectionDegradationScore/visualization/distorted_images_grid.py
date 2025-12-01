import torch
import sys
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from ultralytics import YOLO
import numpy as np
import warnings
import lpips
from skimage.metrics import structural_similarity as ssim_skimage
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
import cv2
import glob
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader import ImagePairDataset
from dds_metric import match_predictions


def tensor_to_numpy(tensor_img):
    """Convert tensor image to numpy array for skimage metrics"""
    # Convert from (C, H, W) to (H, W, C) and denormalize if needed
    numpy_img = tensor_img.cpu().numpy().transpose(1, 2, 0)
    
    # If image is normalized to [0,1], scale to [0,255]
    if numpy_img.max() <= 1.0:
        numpy_img = (numpy_img * 255).astype(np.uint8)
    else:
        numpy_img = numpy_img.astype(np.uint8)
    
    return numpy_img

def calculate_ssim_skimage(img1, img2):
    """Calculate SSIM using skimage"""
    img1_np = tensor_to_numpy(img1)
    img2_np = tensor_to_numpy(img2)
    
    # Convert to grayscale for SSIM if needed, or use multichannel
    if img1_np.shape[-1] == 3:
        return ssim_skimage(img1_np, img2_np, channel_axis=2, data_range=255)
    else:
        return ssim_skimage(img1_np, img2_np, data_range=255)

def calculate_psnr_skimage(img1, img2):
    """Calculate PSNR using skimage with error handling"""
    img1_np = tensor_to_numpy(img1)
    img2_np = tensor_to_numpy(img2)
    
    # Suppress the divide by zero warning and handle it
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            psnr_value = psnr_skimage(img1_np, img2_np, data_range=255)
            # If PSNR is infinite (identical images), return a high value
            if np.isinf(psnr_value):
                return 100.0
            return psnr_value
        except:
            return 0.0

def draw_detections_on_image_cv2(numpy_image, detections, color=(255, 0, 0), thickness=2): 
    """
    Draw detection boxes, class labels, and confidence scores on numpy image using OpenCV
    """
    img_copy = numpy_image.copy()
    
    # Get detection data
    if hasattr(detections, 'boxes') and detections.boxes is not None:
        boxes = detections.boxes.xyxy.cpu().numpy()  # Bounding boxes
        confs = detections.boxes.conf.cpu().numpy()  # Confidence scores
        
        # Get class IDs and names
        if hasattr(detections, 'names') and detections.names is not None:
            class_names = detections.names
            class_ids = detections.boxes.cls.cpu().numpy().astype(int) if detections.boxes.cls is not None else None
        else:
            # Fallback: use generic class names if not available
            class_names = {0: 'object'}
            class_ids = [0] * len(boxes)
        
        for i, (box, conf) in enumerate(zip(boxes, confs)):
            x1, y1, x2, y2 = map(int, box)
            
            # Get class name
            if class_ids is not None and i < len(class_ids):
                class_id = class_ids[i]
                class_name = class_names.get(class_id, f'class_{class_id}')
            else:
                class_name = 'object'
            
            # Draw bounding box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
            
            # Create label with class name and confidence
            label = f"{class_name}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)[0]
            
            # Draw background for text
            cv2.rectangle(img_copy, (x1, y1 - label_size[1] - 5), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw text
            cv2.putText(img_copy, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    
    return img_copy

def create_5_image_comparison_cv2(gt_img, compressed_imgs, gt_detections, comp_detections_list, 
                                 comp_metrics_list, save_path, padding=20):
    """
    Create a combined image with 5 images in a row: 1 GT + 4 compressed versions
    """
    # Convert tensors to numpy arrays for OpenCV
    gt_np = tensor_to_numpy(gt_img)
    compressed_nps = [tensor_to_numpy(img) for img in compressed_imgs]
    
    # Convert BGR to RGB if needed (OpenCV uses BGR)
    if gt_np.shape[-1] == 3:
        gt_np = cv2.cvtColor(gt_np, cv2.COLOR_RGB2BGR)
        compressed_nps = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in compressed_nps]
    
    # Draw detection boxes on all images - ALL GREEN now
    gt_with_detections = draw_detections_on_image_cv2(gt_np, gt_detections, color=(255, 0, 0))  # Green in BGR
    
    compressed_with_detections = []
    for comp_img, comp_det in zip(compressed_nps, comp_detections_list):
        comp_with_det = draw_detections_on_image_cv2(comp_img, comp_det, color=(255, 0, 0))  # Green in BGR
        compressed_with_detections.append(comp_with_det)
    
    # Get dimensions
    gt_height, gt_width = gt_with_detections.shape[:2]
    comp_heights = [img.shape[0] for img in compressed_with_detections]
    comp_widths = [img.shape[1] for img in compressed_with_detections]
    
    # Use consistent height (max height among all images)
    max_height = max(gt_height, max(comp_heights))
    
    # Resize all images to have the same height while maintaining aspect ratio
    def resize_to_height(img, target_height):
        h, w = img.shape[:2]
        scale = target_height / h
        new_w = int(w * scale)
        return cv2.resize(img, (new_w, target_height))
    
    gt_resized = resize_to_height(gt_with_detections, max_height)
    compressed_resized = [resize_to_height(img, max_height) for img in compressed_with_detections]
    
    # Get new widths after resizing
    gt_width_resized = gt_resized.shape[1]
    comp_widths_resized = [img.shape[1] for img in compressed_resized]
    
    # Calculate total dimensions for combined image
    total_width = gt_width_resized + sum(comp_widths_resized) + padding * 6
    total_height = max_height + padding * 2 + 180  # Space for metrics with labels
    
    # Create white background image
    combined_img = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255
    
    # Calculate positions for images
    current_x = padding
    image_y = padding + 30  # Reduced space since no titles for compressed images
    
    # Paste ground truth image
    combined_img[image_y:image_y+max_height, current_x:current_x+gt_width_resized] = gt_resized
    current_x += gt_width_resized + padding
    
    # Store the x positions where each compressed image starts
    comp_image_starts = []
    
    # Paste compressed images
    for i, comp_img in enumerate(compressed_resized):
        comp_image_starts.append(current_x)
        combined_img[image_y:image_y+max_height, current_x:current_x+comp_widths_resized[i]] = comp_img
        current_x += comp_widths_resized[i] + padding
    
    # Add only ground truth title
    title_y = padding + 10
    
    # Ground truth title (centered above GT image)
    gt_title = "Ground Truth"
    gt_title_size = cv2.getTextSize(gt_title, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
    gt_title_x = padding + (gt_width_resized - gt_title_size[0]) // 2
    cv2.putText(combined_img, gt_title, (gt_title_x, title_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
    
    # Add metrics below images - with metric names and properly aligned under each compressed image
    metrics_start_y = image_y + max_height + padding + 20
    metric_line_height = 25
    
    # Metric names (will be displayed before values for each compressed image)
    metric_names = ["DDS: ", "LPIPS: ", "SSIM: ", "PSNR: "]
    
    # Add metrics for each compressed image - with metric names and sharp text
    for comp_idx, metrics in enumerate(comp_metrics_list):
        comp_metric_values = [
            f"{metrics['dds']:.4f}",
            f"{metrics['lpips']:.4f}", 
            f"{metrics['ssim']:.4f}",
            f"{metrics['psnr']:.2f} dB"
        ]
        
        # Calculate center position for this compressed image's metrics
        comp_start_x = comp_image_starts[comp_idx]
        comp_center_x = comp_start_x + comp_widths_resized[comp_idx] // 2
        
        for i, (metric_name, metric_value) in enumerate(zip(metric_names, comp_metric_values)):
            y_pos = metrics_start_y + i * metric_line_height
            
            # Combine metric name and value
            full_text = metric_name + metric_value
            text_size = cv2.getTextSize(full_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
            text_x = comp_center_x - text_size[0] // 2
            
            # Draw text with sharp rendering
            cv2.putText(combined_img, full_text, (text_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
    
    # Save the combined image with high quality
    cv2.imwrite(save_path, combined_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

def group_images_by_name_root(images_root):
    """
    Group images by their name root (without the _X suffix)
    """
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_root, 'compressed', ext)))
        image_files.extend(glob.glob(os.path.join(images_root, 'compressed', '**', ext), recursive=True))
    
    # Group by name root
    grouped_images = defaultdict(list)
    
    for image_path in image_files:
        filename = os.path.basename(image_path)
        # Extract name root (remove _X before extension)
        name_parts = filename.rsplit('_', 1)
        if len(name_parts) == 2 and name_parts[1].split('.')[0].isdigit():
            name_root = name_parts[0]
            compressed_idx = int(name_parts[1].split('.')[0])
            grouped_images[name_root].append((compressed_idx, filename, image_path))
        else:
            # If no _X pattern, treat as single image
            name_root = filename.rsplit('.', 1)[0]
            grouped_images[name_root].append((1, filename, image_path))
    
    # Sort each group by compressed index and ensure we have exactly 4 compressed versions
    valid_groups = {}
    for name_root, images in grouped_images.items():
        images.sort(key=lambda x: x[0])  # Sort by compressed index
        compressed_indices = [idx for idx, _, _ in images]
        
        # Check if we have compressed versions 1-4
        if all(i in compressed_indices for i in [1, 2, 3, 4]):
            # Take exactly compressed versions 1, 2, 3, 4
            compressed_files = []
            for target_idx in [1, 2, 3, 4]:
                for idx, filename, path in images:
                    if idx == target_idx:
                        compressed_files.append((filename, path))
                        break
            
            if len(compressed_files) == 4:
                valid_groups[name_root] = compressed_files
    
    return valid_groups

def process_image_groups_with_metrics(
    model_path: str,
    imgs_root: str,
    output_dir: str,
    batch_size: int = 32,
    device: str = "cuda:0",
):
    """
    Process image groups and create 5-image comparisons (1 GT + 4 compressed)
    """
    # Set device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Load YOLO model
    print(f"Loading YOLO model from {model_path}")
    detector = YOLO(model_path)
    detector.to(device)
    
    # Load LPIPS model
    lpips_model = lpips.LPIPS(net='alex').to(device)
    
    # Group images by name root
    print("Grouping images by name root...")
    image_groups = group_images_by_name_root(imgs_root)
    print(f"Found {len(image_groups)} valid image groups")
    
    # Debug: print first group to verify we have different files
    if image_groups:
        first_group = next(iter(image_groups.items()))
        print(f"First group '{first_group[0]}' files:")
        for filename, path in first_group[1]:
            print(f"  - {filename} -> {path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    print("Processing image groups...")
    with torch.no_grad():
        for name_root, compressed_files in tqdm(image_groups.items(), desc="Processing groups"):
            # Load ground truth image - ground truth has _1 suffix
            gt_path = os.path.join(imgs_root, 'extracted', f"{name_root}_1.jpg")
            if not os.path.exists(gt_path):
                gt_path = os.path.join(imgs_root, 'extracted', f"{name_root}_1.png")
                if not os.path.exists(gt_path):
                    print(f"Warning: Ground truth image not found for {name_root}_1")
                    continue
            
            # Load ground truth image
            gt_image = cv2.imread(gt_path)
            if gt_image is None:
                print(f"Warning: Could not load ground truth image {gt_path}")
                continue
            
            print(f"Processing {name_root}: GT from {os.path.basename(gt_path)}")
            
            gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
            gt_tensor = torch.from_numpy(gt_image.transpose(2, 0, 1)).float() / 255.0
            gt_tensor = gt_tensor.to(device)
            
            # Load compressed images
            compressed_tensors = []
            compressed_paths = []
            
            for comp_filename, comp_path in compressed_files:
                comp_image = cv2.imread(comp_path)
                if comp_image is None:
                    print(f"Warning: Could not load compressed image {comp_path}")
                    continue
                
                print(f"  - Compressed: {comp_filename} from {comp_path}")
                
                comp_image = cv2.cvtColor(comp_image, cv2.COLOR_BGR2RGB)
                comp_tensor = torch.from_numpy(comp_image.transpose(2, 0, 1)).float() / 255.0
                comp_tensor = comp_tensor.to(device)
                compressed_tensors.append(comp_tensor)
                compressed_paths.append(comp_path)
            
            if len(compressed_tensors) != 4:
                print(f"Warning: Expected 4 compressed images for {name_root}, got {len(compressed_tensors)}")
                continue
            
            # Verify we have different files by checking file sizes
            file_sizes = [os.path.getsize(path) for path in compressed_paths]
            unique_sizes = len(set(file_sizes))
            if unique_sizes < 4:
                print(f"Warning: Only {unique_sizes} unique file sizes for {name_root} - some images might be similar")
            
            # Run detection on all images
            gt_prediction = detector.predict(gt_tensor.unsqueeze(0), verbose=False)[0]
            comp_predictions = detector.predict(torch.stack(compressed_tensors), verbose=False)
            
            # Calculate metrics for each compressed image vs ground truth
            comp_metrics_list = []
            for i, (comp_tensor, comp_path) in enumerate(zip(compressed_tensors, compressed_paths)):
                comp_pred = comp_predictions[i]
                
                # Calculate DDS score
                matches = match_predictions([gt_prediction], [comp_pred])
                ddscore = matches[0]["ddscore"]
                if isinstance(ddscore, torch.Tensor):
                    ddscore = float(ddscore.item())
                
                # Calculate LPIPS
                lpips_score = lpips_model(gt_tensor.unsqueeze(0), comp_tensor.unsqueeze(0)).item()
                
                # Calculate SSIM
                ssim_score = calculate_ssim_skimage(gt_tensor, comp_tensor)
                
                # Calculate PSNR
                psnr_score = calculate_psnr_skimage(gt_tensor, comp_tensor)
                
                comp_metrics = {
                    'dds': ddscore,
                    'lpips': lpips_score,
                    'ssim': ssim_score,
                    'psnr': psnr_score
                }
                comp_metrics_list.append(comp_metrics)
                
                print(f"    Metrics for {os.path.basename(comp_path)}: "
                      f"DDS={ddscore:.4f}, LPIPS={lpips_score:.4f}, "
                      f"SSIM={ssim_score:.4f}, PSNR={psnr_score:.2f}")
            
            # Create output filename
            output_filename = f"{name_root}_comparison.jpg"
            output_path = os.path.join(output_dir, output_filename)
            
            # Create and save the 5-image comparison
            create_5_image_comparison_cv2(gt_tensor, compressed_tensors, gt_prediction, 
                                        comp_predictions, comp_metrics_list, output_path)
            
            # Store results
            results.append({
                "name_root": name_root,
                "output_image": output_filename,
                "compressed_filenames": [f[0] for f in compressed_files],
                "comp_metrics": comp_metrics_list
            })

def main():
    # Configuration
    MODEL_PATH = "yolo11m.pt"
    IMGS_ROOT = "progressive_distorted_dataset/test"
    OUTPUT_DIR = "image_group_comparisons"
    BATCH_SIZE = 32
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    process_image_groups_with_metrics(
        model_path=MODEL_PATH,
        imgs_root=IMGS_ROOT,
        output_dir=OUTPUT_DIR,
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )

if __name__ == "__main__":
    main()