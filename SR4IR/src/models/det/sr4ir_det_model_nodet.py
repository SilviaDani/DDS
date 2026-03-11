import os
import os.path as osp
import torch
import math

from archs import build_network
from losses import build_loss
from torch.nn.functional import interpolate
from utils.common import save_on_master, quantize, calculate_psnr_batch, visualize_image, calculate_lpips_batch
from utils.det import MetricLogger, SmoothedValue, get_coco_api_from_dataset, _get_iou_types, CocoEvaluator

from .base_model import BaseModel

def make_model(opt):
    return SR4IRDetectionModel(opt)

class SR4IRDetectionModel(BaseModel):
    """Base Super-Resolution model for Object Detection."""

    def __init__(self, opt):
        print("sr4ir_det_model.py Initializing SR4IRDetectionModel...")
        super().__init__(opt)
        
        # define network up
        self.net_up = self.model_to_device(torch.nn.UpsamplingBilinear2d(scale_factor=self.scale), is_trainable=False)
        
        # define network sr
        opt['network_sr']['scale'] = self.scale
        self.net_sr = build_network(opt['network_sr'], self.text_logger, tag='net_sr')
        self.load_network(self.net_sr, name='network_sr', tag='net_sr')
        self.net_sr = self.model_to_device(self.net_sr, is_trainable=True)
        self.print_network(self.net_sr, tag='net_sr')
        
        # define network detction
        self.net_det = build_network(opt['network_det'], self.text_logger, task=self.task, tag='net_det')
        self.load_network(self.net_det, name='network_det', tag='net_det')
        
        # [MODIFIED]: Set is_trainable to False for the detector
        # self.net_det = self.model_to_device(self.net_det, is_trainable=True)
        self.net_det = self.model_to_device(self.net_det, is_trainable=False)
        self.print_network(self.net_det, tag='net_det')
        
    def set_mode(self, mode):
        if mode == 'train':
            self.net_sr.train()
            # [MODIFIED]: Force detector to stay in eval mode
            # self.net_det.train()
            self.net_det.eval()
        elif mode == 'eval':
            self.net_sr.eval()
            self.net_det.eval()
        else:
            raise NotImplementedError(f"mode {mode} is not supported")
        
    def init_training_settings(self, data_loader_train):
        self.set_mode(mode='train')
        train_opt = self.opt['train']

        # phase 1
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt'], self.text_logger).to(self.device)
            
        if train_opt.get('tdp_opt'):
            # task driven perceptual loss
            self.cri_tdp = build_loss(train_opt['tdp_opt'], self.text_logger).to(self.device)
      
        # phase 2
        # [MODIFIED]: Commented out all detector losses
        # if train_opt.get('det_sr_opt'):
        #     self.cri_det_sr = build_loss(train_opt['det_sr_opt'], self.text_logger).to(self.device)
        # 
        # if train_opt.get('det_hr_opt'):
        #     self.cri_det_hr = build_loss(train_opt['det_hr_opt'], self.text_logger).to(self.device)
        #     
        # if train_opt.get('det_cqmix_opt'):
        #     self.cri_det_cqmix = build_loss(train_opt['det_cqmix_opt'], self.text_logger).to(self.device)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers(len(data_loader_train), name='sr', optimizer=self.optimizer_sr)
        
        # [MODIFIED]: Commented out det scheduler setup
        # self.setup_schedulers(len(data_loader_train), name='det', optimizer=self.optimizer_det)
        
        # set up saving directories
        os.makedirs(osp.join(self.exp_dir, 'models'), exist_ok=True)
        os.makedirs(osp.join(self.exp_dir, 'checkpoints'), exist_ok=True)
        
        # eval freq
        self.eval_freq = train_opt.get('eval_freq', 1)
        
        # warmup epoch
        self.warmup_epoch = train_opt.get('warmup_epoch', -1)
        self.text_logger.write("NOTICE: total epoch: {}, warmup epoch: {}".format(train_opt['epoch'], self.warmup_epoch))
        
    def setup_optimizers(self):
        train_opt = self.opt['train']
        
        # optimizer sr
        optim_type = train_opt['optim_sr'].pop('type')
        self.optimizer_sr = self.get_optimizer(optim_type, self.net_sr.parameters(), **train_opt['optim_sr'])
        self.optimizers.append(self.optimizer_sr)
        
        # [MODIFIED]: Commented out det optimizer
        # optimizer det
        # optim_type = train_opt['optim_det'].pop('type')
        # net_det_parameters = [p for p in self.net_det.parameters() if p.requires_grad]
        # self.optimizer_det = self.get_optimizer(optim_type, net_det_parameters, **train_opt['optim_det'])
        # self.optimizers.append(self.optimizer_det)

    def get_gradients(self, loss, parameters, retain_graph=False):
        """Compute gradients for a given loss."""
        return torch.autograd.grad(
            loss, parameters, retain_graph=retain_graph, 
            allow_unused=True  # For parameters not used in the loss
        )

    def project_conflicting_gradients(self, main_grads, aux_grads, parameters):
        """PCGrad: Project conflicting gradients to avoid cancellation."""
        for param, g_main, g_aux in zip(parameters, main_grads, aux_grads):
            if g_main is not None and g_aux is not None:
                # Store original shapes
                main_shape = g_main.shape
                aux_shape = g_aux.shape
                
                # Flatten gradients to compute similarity
                g_main_flat = g_main.contiguous().view(-1)
                g_aux_flat = g_aux.contiguous().view(-1)
                
                # Compute cosine similarity
                cos_sim = torch.cosine_similarity(
                    g_main_flat, g_aux_flat, dim=0
                )
                
                # If gradients conflict (cosine < 0), project aux gradient
                if cos_sim < 0:
                    g_aux_projected = g_aux_flat - (g_aux_flat @ g_main_flat) * g_main_flat / (g_main_flat.norm()**2 + 1e-8)
                    combined = g_main_flat + g_aux_projected
                else:
                    combined = g_main_flat + g_aux_flat
                
                # Reshape back to original dimensions before assignment
                param.grad = combined.contiguous().view(main_shape)
            elif g_main is not None:
                param.grad = g_main  # Only main gradient exists
                
    def train_one_epoch(self, data_loader_train, train_sampler, epoch):
        self.set_mode(mode='train')
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr_sr", SmoothedValue(window_size=1, fmt="{value}"))
        
        # [MODIFIED]: Commented out lr_det meter
        # metric_logger.add_meter("lr_det", SmoothedValue(window_size=1, fmt="{value}"))
        
        if self.dist:
            train_sampler.set_epoch(epoch)
            
        if epoch < self.warmup_epoch + 1:
            self.text_logger.write("NOTICE: Doing warm-up")
            
        # NOTE: without warmup, training explodes!!
        lr_scheduler_s = None
        lr_scheduler_d = None
        if epoch == 1:
            warmup_factor = 1.0 / len(data_loader_train)
            warmup_iters = len(data_loader_train)
            lr_scheduler_s = torch.optim.lr_scheduler.LinearLR(
                self.optimizer_sr, start_factor=warmup_factor, total_iters=warmup_iters)
            # [MODIFIED]: Commented out det scheduler
            # lr_scheduler_d = torch.optim.lr_scheduler.LinearLR(
            #     self.optimizer_det, start_factor=warmup_factor, total_iters=warmup_iters)

        header = f"Epoch: [{epoch}, Name {self.opt['name']}]"
        for iter, (img_hr_list, target_list) in enumerate(metric_logger.log_every(data_loader_train, self.opt['print_freq'], self.text_logger, header)):
            img_hr_list = list(img_hr.to(self.device) for img_hr in img_hr_list)
            target_list = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in target_list]
            current_iter = iter + len(data_loader_train)*(epoch-1)

            # make on-the-fly LR image
            img_hr_batch = self.list_to_batch(img_hr_list)
            img_lr_batch = quantize(interpolate(img_hr_batch, scale_factor=(1/self.scale), mode='bicubic'))
            
            # phase 1;
            # update net_sr, freeze net_cls
            img_sr_batch = self.net_sr(img_lr_batch)
            img_sr_list = self.batch_to_list(img_sr_batch, img_list=img_hr_list)
            
            # [MODIFIED]: Harmless, but kept for redundancy. Det is already frozen.
            for p in self.net_det.parameters(): p.requires_grad = False
            self.optimizer_sr.zero_grad()
            l_total_sr = 0
            if hasattr(self, 'cri_pix'):
                l_pix = self.cri_pix(img_sr_batch, img_hr_batch)
                metric_logger.meters["l_pix"].update(l_pix.item()) 
                self.tb_logger.add_scalar('losses/l_pix', l_pix.item(), current_iter)
                l_total_sr += l_pix                
            if epoch > self.warmup_epoch and hasattr(self, 'cri_tdp'):
                self.net_det.eval()  # Ensure detector is in eval mode (BatchNorm/Dropout fixed)
                
                # ---------------------------------------------------------
                # STEP 1: Get the independent proposals (No gradients needed here)
                # ---------------------------------------------------------
                with torch.no_grad():
                    _, _, init_feat_sr = self.net_det(img_sr_list, return_feats=True)
                    _, _, init_feat_hr = self.net_det(img_hr_list, return_feats=True)
                    
                # ---------------------------------------------------------
                # STEP 2: Combine the proposals for each image in the batch
                # ---------------------------------------------------------
                combined_proposals = []
                # Both lists contain tensors of shape [N, 4]. We concatenate them so 
                # both SR and HR get evaluated on the exact same union of boxes.
                for p_sr, p_hr in zip(init_feat_sr['proposals'], init_feat_hr['proposals']):
                    combined_proposals.append(torch.cat([p_sr, p_hr], dim=0))

                # # ---------------------------------------------------------
                # # STEP 2v2: Extracting clean ground truth proposals
                # # ---------------------------------------------------------
                # # The dataloader's 'targets' is a list of dicts. 
                # # We just extract the 'boxes' tensor from each dict in the batch.
                # gt_proposals = [t["boxes"] for t in target_list]
                # ---------------------------------------------------------
                # STEP 3: Second pass forcing the exact same combined proposals
                # ---------------------------------------------------------
                # Gradients will flow through this pass to update your SR generator
                _, _, total_feat_sr = self.net_det(img_sr_list, manual_proposals=combined_proposals, return_feats=True)
                _, _, total_feat_hr = self.net_det(img_hr_list, manual_proposals=combined_proposals, return_feats=True)
                
                # ---------------------------------------------------------
                # STEP 4: Build the feature dictionaries for the loss
                # ---------------------------------------------------------             

                #  Check if the tensor is completely empty (0 boxes across the WHOLE batch)
                if total_feat_sr['roi_features'].shape[0] == 0:
                    # Multiply by 0.0 to safely zero the loss while keeping it attached to the graph
                    l_tdp = total_feat_sr['roi_features'].sum() * 0.0
                else:
                    # If boxes exist, calculate the loss normally
                    feat_sr = {'roi_features': total_feat_sr['roi_features']}
                    feat_hr = {'roi_features': total_feat_hr['roi_features']}
                    l_tdp = self.cri_tdp(feat_hr, feat_sr)
                
                metric_logger.meters["l_tdp"].update(l_tdp.item())
                
                # wo pcgrad
                self.tb_logger.add_scalar('losses/l_tdp', l_tdp.item(), current_iter)
                l_total_sr += l_tdp
                l_total_sr.backward()
            else:
                # Standard backward pass
                l_total_sr.backward()
            self.optimizer_sr.step()
            
            # --- [ADDED: VOC to COCO forward mapping for Phase 2] ---
            # If you ever uncomment Phase 2 below, swap `target_list` with `target_list_coco`
            # voc_to_coco = {
            #     0: 0, 1: 5, 2: 2, 3: 16, 4: 9, 5: 44, 6: 6, 7: 3, 8: 17,
            #     9: 62, 10: 21, 11: 67, 12: 18, 13: 19, 14: 4, 15: 1,
            #     16: 64, 17: 20, 18: 63, 19: 7, 20: 72
            # }
            # target_list_coco = []
            # for t in target_list:
            #     t_mapped = {k: v for k, v in t.items()}
            #     if 'labels' in t_mapped:
            #         t_mapped['labels'] = torch.tensor(
            #             [voc_to_coco[int(l)] for l in t['labels']], 
            #             dtype=t['labels'].dtype, 
            #             device=self.device
            #         )
            #     target_list_coco.append(t_mapped)
            # --------------------------------------------------------

            # [MODIFIED]: ENTIRE PHASE 2 COMMENTED OUT
            # phase 2;
            # update network det, freeze net_cls
            # img_sr_batch = self.net_sr(img_lr_batch).detach()
            # img_sr_list = self.batch_to_list(img_sr_batch, img_list=img_hr_list)
            # for p in self.net_det.parameters(): p.requires_grad = True
            # self.optimizer_det.zero_grad()
            # l_total_det = 0
            # if hasattr(self, 'cri_det_sr'):
            #     _, loss_dict_sr = self.net_det(img_sr_list, target_list_coco) # <-- Use mapped targets here!
            #     l_det_sr = self.cri_det_sr(loss_dict_sr)
            #     ...
            
            # psnr, lr
            psnr, valid_batch_size = calculate_psnr_batch(quantize(img_sr_batch), img_hr_batch)
            metric_logger.meters["psnr"].update(psnr.item(), n=valid_batch_size)
            metric_logger.update(lr_sr=round(self.optimizer_sr.param_groups[0]["lr"], 8))
            # [MODIFIED]: Commented out det lr logging
            # metric_logger.update(lr_det=round(self.optimizer_det.param_groups[0]["lr"], 8))
            
            # update learning rate
            if epoch == 1:
                lr_scheduler_s.step()
                # [MODIFIED]: Commented out det scheduler step
                # lr_scheduler_d.step()
            else:
                self.update_learning_rate()
        return
            
    @torch.inference_mode()
    def evaluate(self, data_loader_test, epoch=0):
        if hasattr(self, 'eval_freq') and (epoch % self.eval_freq != 0):
            return
        
        self.set_mode(mode='eval')
        metric_logger = MetricLogger(delimiter="  ")
        header = "Test:"
        
        coco = get_coco_api_from_dataset(data_loader_test.dataset)
        iou_types = _get_iou_types(self.net_det)
        coco_evaluator = CocoEvaluator(coco, iou_types)
        
        num_processed_samples = 0
        for (img_hr_list, target_list), filename in metric_logger.log_every(data_loader_test, 1000, self.text_logger, header, return_filename=True):
            img_hr_list = list(img_hr.to(self.device) for img_hr in img_hr_list)
            target_list = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in target_list]

            # make on-the-fly LR image
            img_hr_batch = self.list_to_batch(img_hr_list)
            img_lr_batch = quantize(interpolate(img_hr_batch, scale_factor=(1/self.scale), mode='bicubic'))
            
            # perform SR
            img_sr_batch = self.net_sr(img_lr_batch)
            img_sr_list = self.batch_to_list(img_sr_batch, img_list=img_hr_list)
            
            # object detection
            if torch.cuda.is_available(): torch.cuda.synchronize()
            outputs_sr, _ = self.net_det(img_sr_list)

            # --- [ADDED: COCO to VOC backward mapping for evaluation] ---
            # coco_to_voc = {
            #     5: 1, 2: 2, 16: 3, 9: 4, 44: 5, 6: 6, 3: 7, 17: 8, 62: 9,
            #     21: 10, 67: 11, 18: 12, 19: 13, 4: 14, 1: 15, 64: 16,
            #     20: 17, 63: 18, 7: 19, 72: 20
            # }

            coco_to_voc = {1:1, 2:2, 3:3, 4:4, 6:6, 8:8} #for visdrone, already converted in annotations


            outputs_voc = []
            for output in outputs_sr:
                mapped_labels = []
                keep_mask = []
                
                # Check every prediction the COCO model made
                for label in output["labels"]:
                    l_val = int(label)
                    if l_val in coco_to_voc:
                        mapped_labels.append(coco_to_voc[l_val])
                        keep_mask.append(True)
                    else:
                        # Throw away predictions for things not in VOC
                        keep_mask.append(False)
                
                # Filter out the invalid predictions and map remaining labels
                keep = torch.tensor(keep_mask, dtype=torch.bool, device=output["labels"].device)
                
                # Create mapped dictionary natively on the device first, then push to CPU
                out_mapped = {
                    "boxes": output["boxes"][keep].to(torch.device("cpu")),
                    "scores": output["scores"][keep].to(torch.device("cpu")),
                    "labels": torch.tensor(mapped_labels, dtype=output["labels"].dtype, device=torch.device("cpu"))
                }
                outputs_voc.append(out_mapped)

            # Replace the raw COCO outputs with our filtered VOC outputs
            outputs_sr = outputs_voc
            # ------------------------------------------------------------

            # visualizing tool
            if self.opt['test'].get('visualize', False): # and (num_processed_samples < 20):
                self.visualize(img_sr_list[0], outputs_sr[0], filename)

            # evaluation on validation batch
            batch_size = len(img_sr_list)
            psnr, valid_batch_size = calculate_psnr_batch(quantize(img_sr_batch), img_hr_batch)
            metric_logger.meters["psnr"].update(psnr.item(), n=valid_batch_size)
            if self.opt['test'].get('calculate_lpips', False):
                lpips, valid_batch_size = calculate_lpips_batch(quantize(img_sr_batch), img_hr_batch, self.net_lpips)
                metric_logger.meters["lpips"].update(lpips.item(), n=valid_batch_size)
            
            res = {target["image_id"]: output for target, output in zip(target_list, outputs_sr)}
            coco_evaluator.update(res)
            num_processed_samples += batch_size
    
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        coco_evaluator.synchronize_between_processes()
        
        # logging training state
        metric_summary = f"{header}"
        metric_summary = self.add_metric(metric_summary, 'PSNR', metric_logger.psnr.global_avg, epoch)
        if self.opt['test'].get('calculate_lpips', False):
            metric_summary = self.add_metric(metric_summary, 'LPIPS', metric_logger.lpips.global_avg, epoch)
        self.text_logger.write(metric_summary)

        # accumulate predictions from all images
        coco_evaluator.accumulate()
        coco_evaluator.summarize(self.text_logger, tag='SR')
        return

    def save(self, epoch):
        # Create full checkpoint dictionary
        checkpoint = {
            "epoch": epoch,
            "opt": self.opt,
            "net_sr": self.get_bare_model(self.net_sr).state_dict(),
            # [MODIFIED]: Stopped saving net_det weights in checkpoint since they don't change
            # "net_det": self.get_bare_model(self.net_det).state_dict(),
            "schedulers": [],
            "optimizers": [] # Added optimizers list
        }
        
        # Save Scheduler states
        for s in self.schedulers:
            checkpoint['schedulers'].append(s.state_dict())

        # Save Optimizer states (CRITICAL FOR RESUMING)
        for o in self.optimizers:
            checkpoint['optimizers'].append(o.state_dict())
                
        # Save logic
        if epoch % self.opt['train']['save_freq'] == 0:
            save_on_master(self.get_bare_model(self.net_sr).state_dict(), osp.join(self.exp_dir, 'models', "net_sr_{:03d}.pth".format(epoch)))
            # [MODIFIED]: Commented out saving det models to save disk space
            # save_on_master(self.get_bare_model(self.net_det).state_dict(), osp.join(self.exp_dir, 'models', "net_det_{:03d}.pth".format(epoch)))
            save_on_master(checkpoint, osp.join(self.exp_dir, 'checkpoints', "checkpoint_{:03d}.pth".format(epoch)))
            
        save_on_master(self.get_bare_model(self.net_sr).state_dict(), osp.join(self.exp_dir, 'models', "net_sr_latest.pth"))
        # [MODIFIED]: Commented out saving det models to save disk space
        # save_on_master(self.get_bare_model(self.net_det).state_dict(), osp.join(self.exp_dir, 'models', "net_det_latest.pth"))
        save_on_master(checkpoint, osp.join(self.exp_dir, 'checkpoints', "checkpoint_latest.pth"))
        return

    def resume_training(self, checkpoint_path):
        """
        Loads the entire training state (epoch, weights, optimizers, schedulers)
        to resume exactly where you left off.
        """
        if not osp.exists(checkpoint_path):
            print(f"No checkpoint found at {checkpoint_path}, starting from scratch.")
            return 0  # Start from epoch 0

        print(f"Resuming training from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # 1. Load Model Weights
        self.get_bare_model(self.net_sr).load_state_dict(checkpoint['net_sr'])
        
        # [MODIFIED]: Commented out loading det weights since they aren't saved/changed
        # self.get_bare_model(self.net_det).load_state_dict(checkpoint['net_det'])

        # 2. Load Optimizers (The Momentum)
        if 'optimizers' in checkpoint:
            for i, state in enumerate(checkpoint['optimizers']):
                if i < len(self.optimizers):
                    self.optimizers[i].load_state_dict(state)
        
        # 3. Load Schedulers (The Learning Rate)
        if 'schedulers' in checkpoint:
            for i, state in enumerate(checkpoint['schedulers']):
                if i < len(self.schedulers):
                    self.schedulers[i].load_state_dict(state)

        # 4. Return the next epoch
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming successful. Starting from Epoch {start_epoch}")
        
        return start_epoch