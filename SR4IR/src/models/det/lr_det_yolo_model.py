import os
import os.path as osp
import torch
import math
import torch.nn.functional as F
from archs import build_network
from losses import build_loss
from torch.nn.functional import interpolate
from utils.common import save_on_master, quantize, calculate_psnr_batch, calculate_lpips_batch, visualize_image
from utils.det import MetricLogger, SmoothedValue, get_coco_api_from_dataset, _get_iou_types, CocoEvaluator
from ultralytics import YOLO, RTDETR
from .base_model import BaseModel


def make_model(opt):
    return LRDetectionModel(opt)


class LRDetectionModel(BaseModel):
    """Object Detection model."""

    def __init__(self, opt):
        super(LRDetectionModel, self).__init__(opt)
        
        # define network up
        self.net_up = self.model_to_device(torch.nn.UpsamplingBilinear2d(scale_factor=self.scale), is_trainable=False)
        
        # 2. YOLOv8x
        yolo_path = opt.get('network_det', {}).get('weights_yolo', 'yolo26ft.pt') 
        self.yolo_model = YOLO(yolo_path)
        self.yolo_model.to(self.device)

        # 3. RT-DETR
        rtdetr_path = opt.get('network_det', {}).get('weights_rtdetr', 'rtdetr-l_ft.pt') 
        self.rtdetr_model = RTDETR(rtdetr_path)
        self.rtdetr_model.to(self.device)

        # 4. Class Remapping Logic 
        
        #(KITTY_VISDRONE to ODVIRAT)
        self.coco_to_voc_map = {
            0: 4,  1: 4,  2: 1,  
            3: 2,  4: 3,  5: 3   
        }

        # --- NEW: Class Names and COCO-style Colors (BGR Format for OpenCV) ---
        self.class_names = {
            1: 'bike',
            2: 'car', 
            3: 'vehicle',
            4: 'person', 
        }
        
        #KITTI_VISDRONE FOR KITTI & VISDRONE <---
        # self.coco_to_voc_map = {
        #     0: 0,  1: 1,  2: 2,  
        #     3: 3,  4: 4,  5: 5   
        # }

        # # --- NEW: Class Names and COCO-style Colors (BGR Format for OpenCV) ---
        # self.class_names = {
        #     0: 'pedestrian',
        #     1: 'person', 
        #     2: 'bicycle', 
        #     3: 'car', 
        #     4: 'van', 
        #     5: 'truck'
        # }

        
        self.box_color_bgr = (0, 0, 255)     # Red
        self.label_color_bgr = (5, 100, 255) # Orange
        # ----------------------------------------------------------------------

        # Directories for visualization
        self.vis_dir = osp.join(self.opt.get('path', {}).get('experiments_root', './experiments'), 'det/034_SR4IR_swinir_x4_woPCGrad_30epochs_3warmup_ODVirat')
        if self.opt['test'].get('visualize', False):
            self.yolo_vis_dir = osp.join(self.vis_dir, 'visualize_yolo')
            self.rtdetr_vis_dir = osp.join(self.vis_dir, 'visualize_rtdetr')
        
            os.makedirs(self.yolo_vis_dir, exist_ok=True)
            os.makedirs(self.rtdetr_vis_dir, exist_ok=True)
        
    @torch.inference_mode()
    def evaluate(self, data_loader_test, epoch=0):
        if hasattr(self, 'eval_freq') and (epoch % self.eval_freq != 0):
            return

        metric_logger = MetricLogger(delimiter="  ")
        header = "Test:"
        coco = get_coco_api_from_dataset(data_loader_test.dataset)
        
        coco_evaluator_yolo = CocoEvaluator(coco, ["bbox"])
        coco_evaluator_rtdetr = CocoEvaluator(coco, ["bbox"])
        
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
            
        for (img_hr_list, target_list), filename in metric_logger.log_every(data_loader_test, 1000, self.text_logger, header, return_filename=True):
            img_hr_list = list(img_hr.to(self.device) for img_hr in img_hr_list)
            target_list = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in target_list]
            
            # make on-the-fly LR image
            img_hr_batch = self.list_to_batch(img_hr_list)
            img_lr_batch = self.net_up(quantize(interpolate(img_hr_batch, scale_factor=(1/self.scale), mode='bicubic')))
            img_lr_list = self.batch_to_list(img_lr_batch, img_list=img_hr_list)
            
            # 2. Padding
            _, _, h_orig, w_orig = img_lr_batch.shape
            h_pad, w_pad = math.ceil(h_orig / 32) * 32, math.ceil(w_orig / 32) * 32
            img_lr_padded = F.pad(img_lr_batch, (0, w_pad - w_orig, 0, h_pad - h_orig), value=0)

            # 3. Dual Inference
            yolo_results = self.yolo_model(img_lr_padded, verbose=False)
            rtdetr_results = self.rtdetr_model(img_lr_padded, verbose=False)
            
            # 4. Format outputs using helper (We do this FIRST now so we have the remapped labels)
            outputs_yolo = format_results(yolo_results, w_orig, h_orig)
            outputs_rtdetr = format_results(rtdetr_results, w_orig, h_orig)

            # --- NEW: Custom OpenCV Plotting matching your image style ---
            # Helper function to draw on images
            def draw_predictions(img, outputs_dict):
                out_img = img.copy()
                boxes = outputs_dict['boxes']
                labels = outputs_dict['labels']
                
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = map(int, boxes[i].tolist())
                    lbl = int(labels[i].item())
                    
                    # Fetch customized name
                    name = self.class_names.get(lbl, f"ID_{lbl}")
                    
                    # 1. Draw Bounding Box (Red)
                    cv2.rectangle(out_img, (x1, y1), (x2, y2), self.box_color_bgr, 2)
                    
                    # 2. Draw Text (Orange, placed just inside top edge, no confidence)
                    cv2.putText(out_img, name, (x1 + 7, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.label_color_bgr, 2)
                
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
                    rtdetr_img = draw_predictions(img_bgr_base, outputs_rtdetr[idx])
                    cv2.imwrite(osp.join(self.rtdetr_vis_dir, save_name), rtdetr_img)
                # -------------------------------------------------------------------
            if self.opt['test'].get('calculate_lpips', False):
                lpips, valid_batch_size = calculate_lpips_batch(quantize(img_lr_batch), img_hr_batch, self.net_lpips)
                metric_logger.meters["lpips"].update(lpips.item(), n=valid_batch_size)

            # 5. Update Evaluators
            res_yolo = {target["image_id"]: output for target, output in zip(target_list, outputs_yolo)}
            res_rtdetr = {target["image_id"]: output for target, output in zip(target_list, outputs_rtdetr)}
            
            coco_evaluator_yolo.update(res_yolo)
            coco_evaluator_rtdetr.update(res_rtdetr)
        
        metric_summary = f"{header}"
        if self.opt['test'].get('calculate_lpips', False):
            metric_summary = self.add_metric(metric_summary, 'LPIPS', metric_logger.lpips.global_avg, epoch)
        self.text_logger.write(metric_summary)
    
        # 6. Finalize YOLO
        print("\nSynchronizing YOLOv8x evaluator...")
        coco_evaluator_yolo.synchronize_between_processes()
        coco_evaluator_yolo.accumulate()
        coco_evaluator_yolo.summarize()
        stats_yolo = coco_evaluator_yolo.coco_eval['bbox'].stats
        
        # 7. Finalize RT-DETR
        print("\nSynchronizing RT-DETR evaluator...")
        coco_evaluator_rtdetr.synchronize_between_processes()
        coco_evaluator_rtdetr.accumulate()
        coco_evaluator_rtdetr.summarize()
        stats_rtdetr = coco_evaluator_rtdetr.coco_eval['bbox'].stats
        
        # 8. Print Clean Comparison
        print(f"\n====== SR EVALUATION RESULTS (VOC Classes) ======")
        print(f"{'Metric':<30} | {'YOLOv8x':<10} | {'RT-DETR':<10}")
        print("-" * 55)
        print(f"{'mAP (IoU=0.5:0.95)':<30} | {stats_yolo[0]:.4f}     | {stats_rtdetr[0]:.4f}")
        print(f"{'mAP (IoU=0.50)':<30} | {stats_yolo[1]:.4f}     | {stats_rtdetr[1]:.4f}")
        print(f"{'mAP (IoU=0.75)':<30} | {stats_yolo[2]:.4f}     | {stats_rtdetr[2]:.4f}")
        print("-" * 55)
        print(f"{'mAP_small (Area < 32^2)':<30} | {stats_yolo[3]:.4f}     | {stats_rtdetr[3]:.4f}")
        print(f"{'mAP_medium (32^2 <= Area < 96^2)':<30} | {stats_yolo[4]:.4f}     | {stats_rtdetr[4]:.4f}")
        print(f"{'mAP_large (Area >= 96^2)':<30} | {stats_yolo[5]:.4f}     | {stats_rtdetr[5]:.4f}")
        print("=================================================\n")