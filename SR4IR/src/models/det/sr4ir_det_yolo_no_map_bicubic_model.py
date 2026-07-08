import argparse
import os
import os.path as osp
import torch
import torchvision
import math
import cv2  # <--- Import OpenCV
import numpy as np # <--- Import Numpy for tensor conversion
import torch.nn.functional as F
# IMPORT RTDETR alongside YOLO
from ultralytics import YOLO, RTDETR 
from archs import build_network
from torch.nn.functional import interpolate
from utils.common import quantize, calculate_psnr_batch
from utils.det import MetricLogger, get_coco_api_from_dataset, CocoEvaluator

from .base_model import BaseModel


def make_model(opt):
    return SR4IRDetectionModel(opt)

class SR4IRDetectionModel(BaseModel):
    def __init__(self, opt):
        print("sr4ir_det_model.py Initializing for DUAL INFERENCE (YOLOv8x & RT-DETR)...")
        super().__init__(opt)
        
        # 2. YOLOv8x
        yolo_path = opt.get('network_det', {}).get('weights_yolo', 'yolo26ft.pt') 
        self.yolo_model = YOLO(yolo_path)
        self.yolo_model.to(self.device)

        # # 3. RT-DETR
        # rtdetr_path = opt.get('network_det', {}).get('weights_rtdetr', 'rtdetr-l_ft.pt') 
        # self.rtdetr_model = RTDETR(rtdetr_path)
        # self.rtdetr_model.to(self.device)

        # 4. Class Remapping Logic

        # #(KITTY_VISDRONE to ODVIRAT)
        # self.coco_to_voc_map = {
        #     0: 4,  1: 4,  2: 1,  
        #     3: 2,  4: 3,  5: 3   
        # }

        # # --- NEW: Class Names and COCO-style Colors (BGR Format for OpenCV) ---
        # self.class_names = {
        #     1: 'bike',
        #     2: 'car', 
        #     3: 'vehicle',
        #     4: 'person', 
        # }

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

        # YOLO/RT-DETR contiguous index (0-79) to VOC class mapping (1-20)
        # self.coco_to_voc_map = {
        #     0: 15,  # person
        #     1: 2,   # bicycle
        #     2: 7,   # car
        #     3: 14,  # motorcycle -> motorbike
        #     4: 1,   # airplane -> aeroplane
        #     5: 6,   # bus
        #     6: 19,  # train
        #     8: 4,   # boat
        #     14: 3,  # bird
        #     15: 8,  # cat
        #     16: 12, # dog
        #     17: 13, # horse
        #     18: 17, # sheep
        #     19: 10, # cow
        #     39: 5,  # bottle
        #     56: 9,  # chair
        #     57: 18, # couch -> sofa
        #     58: 16, # potted plant -> pottedplant
        #     60: 11, # dining table -> diningtable
        #     62: 20  # tv -> tvmonitor
        # }

        # self.class_names = {
        #     1: 'aeroplane',
        #     2: 'bicycle',
        #     3: 'bird',
        #     4: 'boat',
        #     5: 'bottle',
        #     6: 'bus',
        #     7: 'car',
        #     8: 'cat',
        #     9: 'chair',
        #     10: 'cow',
        #     11: 'diningtable',
        #     12: 'dog',
        #     13: 'horse',
        #     14: 'motorbike',
        #     15: 'person',
        #     16: 'pottedplant',
        #     17: 'sheep',
        #     18: 'sofa',
        #     19: 'train',
        #     20: 'tvmonitor'
        # }

        
        self.box_color_bgr = (0, 255, 0)     # Green for the bounding box
        self.label_color_bgr = (0, 255, 0)   # Green for the label background
        self.text_color_bgr = (0, 0, 0)      # Black for the text inside
        # ----------------------------------------------------------------------

        # Directories for visualization
        self.vis_dir = osp.join(self.opt.get('path', {}).get('experiments_root', './experiments'))
        if self.opt['test'].get('visualize', False):
            self.yolo_vis_dir = osp.join(self.vis_dir, 'visualization_output_bicubic/visualize_yolo')
        
            os.makedirs(self.yolo_vis_dir, exist_ok=True)

    @torch.inference_mode()
    def evaluate(self, data_loader_test, epoch=0):
        metric_logger = MetricLogger(delimiter="  ")
        # coco = get_coco_api_from_dataset(data_loader_test.dataset)
        
        # coco_evaluator_yolo = CocoEvaluator(coco, ["bbox"])
        # coco_evaluator_rtdetr = CocoEvaluator(coco, ["bbox"])
        
        def format_results(results, w_orig, h_orig):
            outputs = []
            for result in results:
                boxes = result.boxes.xyxy.cpu()
                scores = result.boxes.conf.cpu()
                cls_ids = result.boxes.cls.cpu().tolist()
                
                final_boxes, final_scores, final_labels = [], [], []

                for i, c_id in enumerate(cls_ids):
                    if c_id in self.coco_to_voc_map:
                        final_labels.append(self.coco_to_voc_map[c_id])
                        box = boxes[i].clone() 
                        box[0::2] = box[0::2].clamp(0, w_orig)
                        box[1::2] = box[1::2].clamp(0, h_orig)
                        final_boxes.append(box)
                        final_scores.append(scores[i])

                if len(final_boxes) > 0:
                    outputs.append({
                        'boxes': torch.stack(final_boxes),
                        'scores': torch.stack(final_scores),
                        'labels': torch.tensor(final_labels, dtype=torch.int64)
                    })
                else:
                    outputs.append({
                        'boxes': torch.empty((0, 4)),
                        'scores': torch.empty(0),
                        'labels': torch.empty(0, dtype=torch.int64)
                    })
            return outputs

        for (img_hr_list, target_list), filename in metric_logger.log_every(data_loader_test, 100, self.text_logger, "Test:", return_filename=True):

            # 1. SR Generation: simple bicubic baseline
            img_hr_list = [img.to(self.device) for img in img_hr_list]
            img_hr_batch = self.list_to_batch(img_hr_list)

            # Degrade HR -> LR
            img_lr_batch = quantize(
                interpolate(
                    img_hr_batch,
                    scale_factor=1 / self.scale,
                    mode="bicubic",
                    align_corners=False,
                )
            )

            # Bicubic LR -> SR
            img_sr_batch = interpolate(
                img_lr_batch,
                scale_factor=self.scale,
                mode="bicubic",
                align_corners=False,
            )

            img_sr_batch = torch.clamp(img_sr_batch, 0, 1)

            # 2. Padding
            _, _, h_orig, w_orig = img_sr_batch.shape
            h_pad, w_pad = math.ceil(h_orig / 32) * 32, math.ceil(w_orig / 32) * 32
            img_sr_padded = F.pad(img_sr_batch, (0, w_pad - w_orig, 0, h_pad - h_orig), value=0)

            # 3. Dual Inference
            yolo_results = self.yolo_model(img_sr_padded, verbose=False)
            #rtdetr_results = self.rtdetr_model(img_sr_padded, verbose=False)
            
            # 4. Format outputs using helper (We do this FIRST now so we have the remapped labels)
            outputs_yolo = format_results(yolo_results, w_orig, h_orig)
            #outputs_rtdetr = format_results(rtdetr_results, w_orig, h_orig)

            # Helper function to draw on images
            # --- NEW: Custom OpenCV Plotting matching your PIL style ---
            # --- UPDATED: Remapped Labels + Original Confidence Scores ---
            def draw_predictions(img, outputs_dict):
                out_img = img.copy()
                h, w = out_img.shape[:2]
                
                boxes = outputs_dict['boxes']
                labels = outputs_dict['labels']   # These are already remapped (1-4)
                scores = outputs_dict['scores']   # Original COCO confidence scores
                
                # Dynamic scaling based on image height (mimics PIL h/30)
                font_scale = max(0.5, h / 660.0) 
                box_thickness = 5 
                text_thickness = max(1, int(font_scale * 2))
                
                ignored_label_ids = {0, 1}  # 0 = pedestrian, 1 = person

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = map(int, boxes[i].tolist())
                    
                    # 1. Get the remapped ID
                    remapped_id = int(labels[i].item())

                    # Skip pedestrian/person
                    if remapped_id in ignored_label_ids:
                        continue
                    
                    # 2. Get the name from your custom dictionary
                    class_name = self.class_names.get(remapped_id, f"ID_{remapped_id}")
                    
                    # 3. Get the original confidence score
                    conf_score = float(scores[i].item())
                    
                    # 4. Combine them: "car 0.92"
                    label_text = f"{class_name} {conf_score:.2f}"
                                    
                    # --- DRAWING LOGIC ---
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, text_thickness)
                    
                    # Draw Green Box
                    cv2.rectangle(out_img, (x1, y1), (x2, y2), self.box_color_bgr, box_thickness)
                    
                    # Position label background (preventing it from going off-screen)
                    bg_top = max(0, y1 - text_h - 15)
                    bg_bottom = y1
                    
                    # Draw Solid Green Background for Text
                    cv2.rectangle(out_img, (x1, bg_top), (x1 + text_w + 10, bg_bottom), self.label_color_bgr, -1)
                    
                    # Draw Black Text
                    cv2.putText(out_img, label_text, (x1 + 5, y1 - 7), font, font_scale, self.text_color_bgr, text_thickness)
                
                return out_img

            if self.opt['test'].get('visualize', False):
                for idx, target in enumerate(target_list):
                    img_id = int(target["image_id"])
                    save_name = f"{img_id}.jpg"
                    
                    # Convert the unpadded SR tensor (RGB) back to a numpy image (BGR) for OpenCV
                    img_np = (img_sr_batch[idx].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    img_bgr_base = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                    # YOLO Drawing & Saving
                    yolo_img = draw_predictions(img_bgr_base, outputs_yolo[idx])
                    cv2.imwrite(osp.join(self.yolo_vis_dir, save_name), yolo_img)
                    
                    # RT-DETR Drawing & Saving
                    #rtdetr_img = draw_predictions(img_bgr_base, outputs_rtdetr[idx])
                    #cv2.imwrite(osp.join(self.rtdetr_vis_dir, save_name), rtdetr_img)
                # -------------------------------------------------------------------

            # 5. Update Evaluators
            res_yolo = {target["image_id"]: output for target, output in zip(target_list, outputs_yolo)}
            #res_rtdetr = {target["image_id"]: output for target, output in zip(target_list, outputs_rtdetr)}
            
        #     coco_evaluator_yolo.update(res_yolo)
        #     coco_evaluator_rtdetr.update(res_rtdetr)
    
        # # 6. Finalize YOLO
        # print("\nSynchronizing YOLOv8x evaluator...")
        # coco_evaluator_yolo.synchronize_between_processes()
        # coco_evaluator_yolo.accumulate()
        # coco_evaluator_yolo.summarize()
        # stats_yolo = coco_evaluator_yolo.coco_eval['bbox'].stats
        
        # # 7. Finalize RT-DETR
        # print("\nSynchronizing RT-DETR evaluator...")
        # coco_evaluator_rtdetr.synchronize_between_processes()
        # coco_evaluator_rtdetr.accumulate()
        # coco_evaluator_rtdetr.summarize()
        # stats_rtdetr = coco_evaluator_rtdetr.coco_eval['bbox'].stats
        
        # # 8. Print Clean Comparison
        # print(f"\n====== SR EVALUATION RESULTS (VOC Classes) ======")
        # print(f"{'Metric':<30} | {'YOLOv8x':<10} | {'RT-DETR':<10}")
        # print("-" * 55)
        # print(f"{'mAP (IoU=0.5:0.95)':<30} | {stats_yolo[0]:.4f}     | {stats_rtdetr[0]:.4f}")
        # print(f"{'mAP (IoU=0.50)':<30} | {stats_yolo[1]:.4f}     | {stats_rtdetr[1]:.4f}")
        # print(f"{'mAP (IoU=0.75)':<30} | {stats_yolo[2]:.4f}     | {stats_rtdetr[2]:.4f}")
        # print("-" * 55)
        # print(f"{'mAP_small (Area < 32^2)':<30} | {stats_yolo[3]:.4f}     | {stats_rtdetr[3]:.4f}")
        # print(f"{'mAP_medium (32^2 <= Area < 96^2)':<30} | {stats_yolo[4]:.4f}     | {stats_rtdetr[4]:.4f}")
        # print(f"{'mAP_large (Area >= 96^2)':<30} | {stats_yolo[5]:.4f}     | {stats_rtdetr[5]:.4f}")
        # print("=================================================\n")