import os
import os.path as osp
import torch
import math
import cv2
import numpy as np
import torch.nn.functional as F

from archs import build_network
from utils.common import save_on_master
from utils.det import MetricLogger, SmoothedValue
from .base_model import BaseModel

def make_model(opt):
    return HRDetectionModel(opt)

class HRDetectionModel(BaseModel):
    """Object Detection model (Faster R-CNN) with Green/Black visualization."""

    def __init__(self, opt):
        super(HRDetectionModel, self).__init__(opt)
        
        # define network detction
        self.net_det = build_network(opt['network_det'], self.text_logger, task=self.task, tag='net_det')
        self.load_network(self.net_det, name='network_det', tag='net_det')
        self.net_det = self.model_to_device(self.net_det, is_trainable=True)
        self.print_network(self.net_det, tag='net_det')
        
        # 2. Visual Settings (The "Green Version")
        self.box_color_bgr = (0, 255, 0)     # Green
        self.label_color_bgr = (0, 255, 0)   # Green Tab
        self.text_color_bgr = (0, 0, 0)      # Black Text
        
        self.class_names = {
            0: 'pedestrian',
            1: 'person', 
            2: 'bicycle', 
            3: 'car', 
            4: 'van', 
            5: 'truck'
        }

        # 3. Setup Directories (Using opt path since self.exp_dir is unavailable)
        results_base = self.opt.get('path', {}).get('results_root') or \
                       self.opt.get('path', {}).get('experiments_root', './experiments')
        
        self.vis_save_path = osp.join(results_base, 'visualization_output_visdroneGT/visualize_faster')
        os.makedirs(self.vis_save_path, exist_ok=True)

    def draw_predictions(self, img, output):
        """Standard Green Box / Black Text Drawing logic."""
        out_img = img.copy()
        h, w = out_img.shape[:2]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, h / 700.0)
        box_thickness = max(2, int(h / 350.0))
        text_thickness = max(1, int(font_scale * 2))

        boxes = output.get('boxes', [])
        labels = output.get('labels', [])
        scores = output.get('scores', [])

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

            (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, text_thickness)

            # Draw Box
            cv2.rectangle(out_img, (x1, y1), (x2, y2), self.box_color_bgr, box_thickness)
            # Draw Tab
            bg_top = max(0, y1 - text_h - 15)
            cv2.rectangle(out_img, (x1, bg_top), (x1 + text_w + 10, y1), self.label_color_bgr, -1)
            # Draw Text
            cv2.putText(out_img, label_text, (x1 + 5, y1 - 7), font, font_scale, self.text_color_bgr, text_thickness)
        
        return out_img

    @torch.inference_mode()
    def evaluate(self, data_loader_test, epoch=0):
        if hasattr(self, 'eval_freq') and (epoch % self.eval_freq != 0):
            return
        
        self.set_mode('eval')
        metric_logger = MetricLogger(delimiter="  ")
        header = "Test Visualization:"
        
        print(f"\n[*] SAVING VISUALIZATIONS TO: {osp.abspath(self.vis_save_path)}\n")

        for (img_hr_list, target_list), filename in metric_logger.log_every(data_loader_test, 100, self.text_logger, header, return_filename=True):
            img_hr_list = [img.to(self.device) for img in img_hr_list]
            
            # Faster R-CNN Inference
            outputs_hr, _ = self.net_det(img_hr_list)
            
            for i in range(len(img_hr_list)):
                # --- UNIQUE FILENAME FIX ---
                if "image_id" in target_list[i]:
                    img_id = target_list[i]["image_id"]
                    if isinstance(img_id, torch.Tensor): img_id = img_id.item()
                    curr_fname = f"{img_id}.jpg"
                elif isinstance(filename, list) and len(filename) > i:
                    curr_fname = filename[i]
                else:
                    curr_fname = f"img_{i}.jpg"

                # Move specific image output to CPU
                cpu_output = {k: v.to(torch.device("cpu")) for k, v in outputs_hr[i].items()}
                
                # Convert Tensor to BGR Image
                img_np = (img_hr_list[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                # Draw and Save
                drawn_img = self.draw_predictions(img_bgr, cpu_output)
                
                save_path = osp.join(self.vis_save_path, osp.basename(str(curr_fname)))
                if not save_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                    save_path += ".jpg"
                cv2.imwrite(save_path, drawn_img)

        self.text_logger.write(f"Evaluation complete. Images saved in: {self.vis_save_path}")
        return

    def set_mode(self, mode):
        if mode == 'train': self.net_det.train()
        elif mode == 'eval': self.net_det.eval()

    def save(self, epoch):            
        # Optional: update save path logic here if self.exp_dir is missing globally
        pass