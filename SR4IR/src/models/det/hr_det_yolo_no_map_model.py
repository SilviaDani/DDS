import os
import os.path as osp
import torch
import math
import cv2
import numpy as np
import torch.nn.functional as F
from ultralytics import YOLO, RTDETR
from archs import build_network
from utils.common import save_on_master
from utils.det import MetricLogger, get_coco_api_from_dataset
from .base_model import BaseModel

def make_model(opt):
    return HRDetectionModel(opt)

class HRDetectionModel(BaseModel):
    """Object Detection model with dual inference and custom visualization."""

    def __init__(self, opt):
        super(HRDetectionModel, self).__init__(opt)
        
        # 1. Load Models
        yolo_path = opt.get('network_det', {}).get('weights_yolo', 'yolo26ft.pt') 
        self.yolo_model = YOLO(yolo_path)
        self.yolo_model.to(self.device)

        rtdetr_path = opt.get('network_det', {}).get('weights_rtdetr', 'rtdetr-l_ft.pt') 
        self.rtdetr_model = RTDETR(rtdetr_path)
        self.rtdetr_model.to(self.device)

        #KITTI_VISDRONE FOR KITTI & VISDRONE <---
        self.coco_to_voc_map = {
            0: 0,  1: 1,  2: 2,  
            3: 3,  4: 4,  5: 5   
        }

        # --- NEW: Class Names and COCO-style Colors (BGR Format for OpenCV) ---
        self.class_names = {
            0: 'pedestrian',
            1: 'person', 
            2: 'bicycle', 
            3: 'car', 
            4: 'van', 
            5: 'truck'
        }
        
        # 3. Colors (BGR for OpenCV)
        self.box_color_bgr = (0, 255, 0)     # Green
        self.label_color_bgr = (0, 255, 0)   # Green Background
        self.text_color_bgr = (0, 0, 0)      # Black Text
        
        # 4. Setup Directories
        results_base = self.opt.get('path', {}).get('results_root') or \
                       self.opt.get('path', {}).get('experiments_root', './experiments')
        
        self.vis_dir = osp.join(results_base, 'visualization_output_visdroneGT')
        self.yolo_vis_dir = osp.join(self.vis_dir, 'visualize_yolo')
        self.rtdetr_vis_dir = osp.join(self.vis_dir, 'visualize_rtdetr')
        
        os.makedirs(self.yolo_vis_dir, exist_ok=True)
        os.makedirs(self.rtdetr_vis_dir, exist_ok=True)

    def draw_predictions(self, img, outputs_dict):
        """Custom Drawing: Green box, Green tab, Black text + Conf."""
        out_img = img.copy()
        h, w = out_img.shape[:2]
        
        boxes = outputs_dict['boxes']
        labels = outputs_dict['labels']
        scores = outputs_dict['scores']
        
        # Dynamic sizing based on resolution
        font_scale = max(0.5, h / 700.0) 
        box_thickness = max(2, int(h / 300.0))
        text_thickness = max(1, int(font_scale * 2))
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        ignored_label_ids = {0, 1}

        for i in range(len(boxes)):
            if scores[i] < 0.3:
                continue

            lbl_id = int(labels[i].item())

            if lbl_id in ignored_label_ids:
                continue

            x1, y1, x2, y2 = map(int, boxes[i].tolist())
            name = self.class_names.get(lbl_id, f"ID_{lbl_id}")
            label_text = f"{name} {float(scores[i]):.2f}"
            
            # Text size for background tab
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, text_thickness)
            
            # 1. Draw Bounding Box
            cv2.rectangle(out_img, (x1, y1), (x2, y2), self.box_color_bgr, box_thickness)
            
            # 2. Draw Label Tab (Green background)
            bg_top = max(0, y1 - text_h - 15)
            cv2.rectangle(out_img, (x1, bg_top), (x1 + text_w + 10, y1), self.label_color_bgr, -1)
            
            # 3. Draw Text (Black)
            cv2.putText(out_img, label_text, (x1 + 5, y1 - 7), font, font_scale, self.text_color_bgr, text_thickness)
            
        return out_img

    @torch.inference_mode()
    def evaluate(self, data_loader_test, epoch=0):
        # Handle evaluation frequency
        if hasattr(self, 'eval_freq') and (epoch % self.eval_freq != 0):
            return
        
        print(f"\n[*] Visualizing detections to: {osp.abspath(self.vis_dir)}")
        metric_logger = MetricLogger(delimiter="  ")
        header = "Inference:"

        def format_results(results, w_orig, h_orig):
            outputs = []
            for result in results:
                boxes = result.boxes.xyxy.cpu()
                scores = result.boxes.conf.cpu()
                cls_ids = result.boxes.cls.cpu().tolist()
                
                f_boxes, f_scores, f_labels = [], [], []
                for i, c_id in enumerate(cls_ids):
                    if c_id in self.coco_to_voc_map:
                        f_labels.append(self.coco_to_voc_map[c_id])
                        box = boxes[i].clone() 
                        box[0::2] = box[0::2].clamp(0, w_orig)
                        box[1::2] = box[1::2].clamp(0, h_orig)
                        f_boxes.append(box)
                        f_scores.append(scores[i])

                if len(f_boxes) > 0:
                    outputs.append({
                        'boxes': torch.stack(f_boxes),
                        'scores': torch.stack(f_scores),
                        'labels': torch.tensor(f_labels, dtype=torch.int64)
                    })
                else:
                    outputs.append({
                        'boxes': torch.empty((0, 4)),
                        'scores': torch.empty(0),
                        'labels': torch.empty(0, dtype=torch.int64)
                    })
            return outputs
        
        # Main Evaluation Loop
        for (img_hr_list, target_list), filename in metric_logger.log_every(data_loader_test, 100, self.text_logger, header, return_filename=True):
            img_hr_list = [img.to(self.device) for img in img_hr_list]
            img_hr_batch = self.list_to_batch(img_hr_list)
            
            # Padding for model compatibility (stride 32)
            _, _, h_orig, w_orig = img_hr_batch.shape
            h_pad, w_pad = math.ceil(h_orig / 32) * 32, math.ceil(w_orig / 32) * 32
            img_hr_padded = F.pad(img_hr_batch, (0, w_pad - w_orig, 0, h_pad - h_orig), value=0)

            # Dual Model Inference
            yolo_res = self.yolo_model(img_hr_padded, verbose=False)
            rtdetr_res = self.rtdetr_model(img_hr_padded, verbose=False)
            
            outputs_yolo = format_results(yolo_res, w_orig, h_orig)
            outputs_rtdetr = format_results(rtdetr_res, w_orig, h_orig)

            # Draw and Save Batch
            for idx, target in enumerate(target_list):
                img_id = target.get("image_id", idx)
                if isinstance(img_id, torch.Tensor): img_id = img_id.item()
                save_name = f"{img_id}.jpg"
                
                # Tensor to OpenCV BGR Image
                img_np = (img_hr_list[idx].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                # Save YOLO Predictions
                cv2.imwrite(osp.join(self.yolo_vis_dir, save_name), self.draw_predictions(img_bgr, outputs_yolo[idx]))
                
                # Save RT-DETR Predictions
                cv2.imwrite(osp.join(self.rtdetr_vis_dir, save_name), self.draw_predictions(img_bgr, outputs_rtdetr[idx]))

        print("\n[!] Inference complete. All images saved. mAP skipped.")
        return