import os
import os.path as osp
import torch
import math
import cv2
import numpy as np
import torch.nn.functional as F
from torch.nn.functional import interpolate

from archs import build_network
from losses import build_loss
from utils.common import save_on_master, quantize, calculate_psnr_batch
from utils.det import MetricLogger, SmoothedValue

from .base_model import BaseModel

def make_model(opt):
    return SR4IRDetectionModel(opt)

class SR4IRDetectionModel(BaseModel):
    """Integrated SR + Detection model with custom Green/Black visualization."""

    def __init__(self, opt):
        print("sr4ir_det_model.py Initializing SR4IRDetectionModel...")
        super().__init__(opt)
        
        # 1. Networks
        self.net_up = self.model_to_device(torch.nn.UpsamplingBilinear2d(scale_factor=self.scale), is_trainable=False)
        
        opt['network_sr']['scale'] = self.scale
        self.net_sr = build_network(opt['network_sr'], self.text_logger, tag='net_sr')
        self.load_network(self.net_sr, name='network_sr', tag='net_sr')
        self.net_sr = self.model_to_device(self.net_sr, is_trainable=True)
        
        self.net_det = build_network(opt['network_det'], self.text_logger, task=self.task, tag='net_det')
        self.load_network(self.net_det, name='network_det', tag='net_det')
        self.net_det = self.model_to_device(self.net_det, is_trainable=True)
        
        # 2. Visual Settings (The "Green Version")
        self.box_color_bgr = (0, 255, 0)     # Green
        self.label_color_bgr = (0, 255, 0)   # Green Tab
        self.text_color_bgr = (0, 0, 0)      # Black Text
        
        # self.class_names = {
        #     1: 'bike',
        #     2: 'car', 
        #     3: 'vehicle',
        #     4: 'person', 
        # }

        self.class_names = {
            0: 'pedestrian',
            1: 'person', 
            2: 'bicycle', 
            3: 'car', 
            4: 'van', 
            5: 'truck'
        }

        # 3. Directories
        self.vis_dir = osp.join(self.opt.get('path', {}).get('experiments_root', './experiments'))
        
        if self.opt['test'].get('visualize', False):
            # Create a specific sub-folder for the detector visualizations
            self.vis_save_path = osp.join(self.vis_dir, 'visualization_output_visdrone/visualize_faster')
            os.makedirs(self.vis_save_path, exist_ok=True)

    def set_mode(self, mode):
        if mode == 'train':
            self.net_sr.train()
            self.net_det.train()
        elif mode == 'eval':
            self.net_sr.eval()
            self.net_det.eval()

    def visualize(self, img_tensor, output, filename):
        """Draws green boxes and black text on a green tab background."""
        # Convert Tensor [3, H, W] to BGR image
        img_np = (img_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]

        # Dynamic scaling based on image height
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, h / 700.0)
        box_thickness = max(2, int(h / 350.0))
        text_thickness = max(1, int(font_scale * 2))

        boxes = output.get('boxes', [])
        labels = output.get('labels', [])
        scores = output.get('scores', [])

        for i in range(len(boxes)):
            if scores[i] < 0.3: continue  # Threshold for clean visualization

            x1, y1, x2, y2 = map(int, boxes[i].tolist())
            lbl_id = int(labels[i].item())
            name = self.class_names.get(lbl_id, f"ID_{lbl_id}")
            label_text = f"{name} {float(scores[i]):.2f}"

            (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, text_thickness)

            # Draw Box
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), self.box_color_bgr, box_thickness)
            
            # Position tab so it stays on screen
            bg_top = max(0, y1 - text_h - 15)
            # Draw Solid Tab
            cv2.rectangle(img_bgr, (x1, bg_top), (x1 + text_w + 10, y1), self.label_color_bgr, -1)
            # Draw Black Text
            cv2.putText(img_bgr, label_text, (x1 + 5, y1 - 7), font, font_scale, self.text_color_bgr, text_thickness)

        save_path = osp.join(self.vis_save_path, os.path.basename(filename))
        if not save_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            save_path += ".jpg"
        cv2.imwrite(save_path, img_bgr)

    @torch.inference_mode()
    def evaluate(self, data_loader_test, epoch=0):
        if hasattr(self, 'eval_freq') and (epoch % self.eval_freq != 0):
            return
        
        self.set_mode(mode='eval')
        metric_logger = MetricLogger(delimiter="  ")
        header = "Test Visualization:"
        
        abs_save_path = osp.abspath(self.vis_save_path)
        print(f"\n" + "="*60)
        print(f"[*] SAVING GREEN-LABEL VISUALIZATIONS TO:\n    {abs_save_path}")
        print("="*60 + "\n")

        for (img_hr_list, target_list), filename in metric_logger.log_every(data_loader_test, 100, self.text_logger, header, return_filename=True):
            img_hr_list = [img.to(self.device) for img in img_hr_list]
            
            img_hr_batch = self.list_to_batch(img_hr_list)
            img_lr_batch = quantize(interpolate(img_hr_batch, scale_factor=(1/self.scale), mode='bicubic'))
            img_sr_batch = self.net_sr(img_lr_batch)
            img_sr_list = self.batch_to_list(img_sr_batch, img_list=img_hr_list)
            
            outputs_sr, _ = self.net_det(img_sr_list)

            if self.opt['test'].get('visualize', False):
                for i in range(len(img_sr_list)):
                    # --- FIX: GET UNIQUE FILENAME FROM TARGETS ---
                    # 1. Check if 'image_id' exists in the target metadata (standard for COCO/VisDrone)
                    if "image_id" in target_list[i]:
                        img_id = target_list[i]["image_id"]
                        # Handle tensor image_ids
                        if isinstance(img_id, torch.Tensor):
                            img_id = img_id.item()
                        curr_fname = f"{img_id}.jpg"
                    # 2. If no image_id, use the filename list if it matches batch size
                    elif isinstance(filename, list) and len(filename) == len(img_sr_list):
                        curr_fname = filename[i]
                    # 3. Last resort fallback: a generic name that won't overwrite everything
                    else:
                        curr_fname = f"batch_img_{i}.jpg"
                    # ---------------------------------------------

                    cpu_output = {k: v.to(torch.device("cpu")) for k, v in outputs_sr[i].items()}
                    self.visualize(img_sr_list[i], cpu_output, curr_fname)
            
        self.text_logger.write(f"Inference complete. Results saved at: {abs_save_path}")
        return

    def setup_optimizers(self):
        train_opt = self.opt['train']
        
        # Optimizer SR
        optim_type_sr = train_opt['optim_sr'].pop('type')
        self.optimizer_sr = self.get_optimizer(optim_type_sr, self.net_sr.parameters(), **train_opt['optim_sr'])
        self.optimizers.append(self.optimizer_sr)
        
        # Optimizer Detection
        optim_type_det = train_opt['optim_det'].pop('type')
        net_det_parameters = [p for p in self.net_det.parameters() if p.requires_grad]
        self.optimizer_det = self.get_optimizer(optim_type_det, net_det_parameters, **train_opt['optim_det'])
        self.optimizers.append(self.optimizer_det)

    def save(self, epoch):
        checkpoint = {
            "epoch": epoch,
            "net_sr": self.get_bare_model(self.net_sr).state_dict(),
            "net_det": self.get_bare_model(self.net_det).state_dict(),
            "optimizer_sr": self.optimizer_sr.state_dict(),
            "optimizer_det": self.optimizer_det.state_dict(),
        }
        
        save_path = osp.join(self.exp_dir, 'checkpoints', f"checkpoint_{epoch:03d}.pth")
        save_on_master(checkpoint, save_path)
        
        # Also save separate model weights
        save_on_master(self.get_bare_model(self.net_sr).state_dict(), osp.join(self.exp_dir, 'models', f"net_sr_{epoch:03d}.pth"))
        save_on_master(self.get_bare_model(self.net_det).state_dict(), osp.join(self.exp_dir, 'models', f"net_det_{epoch:03d}.pth"))

    def resume_training(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.get_bare_model(self.net_sr).load_state_dict(checkpoint['net_sr'])
        self.get_bare_model(self.net_det).load_state_dict(checkpoint['net_det'])
        self.optimizer_sr.load_state_dict(checkpoint['optimizer_sr'])
        self.optimizer_det.load_state_dict(checkpoint['optimizer_det'])
        return checkpoint['epoch'] + 1