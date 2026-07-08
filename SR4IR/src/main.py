import datetime
import os.path as osp
import time
import torch
import torch.utils.data
import utils.common

from models import make_model
from data import load_data


def main():
    # load opt and args from yaml
    opt, args = utils.common.parse_options()
    opt = utils.common.init_distributed_mode(opt)
    
    # deterministic option for reproduction
    if opt.get('deterministic', False):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        
    # make model
    model = make_model(opt)
    utils.common.copy_opt_file(args.opt, osp.join('experiments', opt['task'], opt['name']))
    
    # prepare data loader
    data_loader_train, data_loader_test, train_sampler, test_sampler = load_data(opt)
    
    # training
    if opt.get('train', False):
        model.init_training_settings(data_loader_train)
        
        # --- RESUME LOGIC FIXED ---
        if opt.get('resume', False):
            # We assume opt['resume'] contains the string path to the .pth file
            # model.resume_training returns the NEXT epoch (e.g., 11)
            start_epoch = model.resume_training(opt['resume'])
            print (f"Resuming training from epoch {start_epoch}")
            end_epoch = opt['train']['epoch']
        else:
            start_epoch = 1
            end_epoch = opt['train']['epoch']
        # --------------------------
        
        model.text_logger.write(f"Start training from epoch {start_epoch}")
        start_time = time.time()
        
        for epoch in range(start_epoch, end_epoch+1):
            model.train_one_epoch(data_loader_train, train_sampler, epoch)
            model.evaluate(data_loader_test, epoch)
            model.save(epoch)
            
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        model.text_logger.write(f"Training time {total_time_str}")

    else:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
        if opt.get('calculate_cost'):
            model.calculate_cost()
        else:
            model.evaluate(data_loader_test, test_sampler)


if __name__ == "__main__":
    main()

#CUDA_VISIBLE_DEVICES=0 python src/main.py -opt options/det/034_DDSFeat_SR4IR_swinir_x4.yml