from .cls import CELoss
from .common import L1Loss, MSELoss, FeatureLoss, PerceptualLoss, DDSRNFeatureLoss
from .det import DETLoss
from .gan import GANLoss
from .seg import AUXCELoss
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights

import sys
# Add the path to the DetectionDegradationScore directory
sys.path.append('../DetectionDegradationScore')
from backbones import Backbone


def build_loss(opt_loss, logger):
    """Build loss from options.
    """

    loss_type = opt_loss.pop('type')


    # If the loss requires a backbone, evaluate it from string
    if 'backbone' in opt_loss and isinstance(opt_loss['backbone'], str):
        opt_loss['backbone'] = eval(opt_loss['backbone']) 
        opt_loss['weights_path'] = "FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT"
        
    loss = eval(loss_type)(**opt_loss)
    logger.write(f'Loss {loss_type} is created')
    
    return loss
