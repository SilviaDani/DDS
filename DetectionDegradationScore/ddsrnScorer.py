import torch
from ddsrn import create_ddsrn_model
from extractor import load_feature_extractor, FeatureExtractor
from backbones import Backbone
from torchvision.transforms import ToTensor
from PIL import Image
from torchvision import transforms
import collections


class ddsrnScorer(torch.nn.Module):
    def __init__(self, model_path: str, backbone: Backbone, weights_path: str, device="cuda"):
        super().__init__()
        self.device = device

        # Save backbone and model info
        self.backbone = backbone
        self.weights_path = weights_path

        # Load DDSRN model
        self.ddsrn = create_ddsrn_model(
            feature_channels=backbone.config.channels,
            layer_indices=backbone.config.indices,
        ).to(device).eval()

        checkpoint = torch.load(model_path, map_location=device)
        self.ddsrn.load_state_dict(checkpoint["model_state_dict"])
        self.ddsrn.eval()

        # Load feature extractor
        self.extractor: FeatureExtractor = load_feature_extractor(
            backbone_name=backbone,
            weights_path=weights_path
        ).to(device).eval()

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img1: Tensor [B, C, H, W] or [C, H, W]
            img2: Tensor [B, C, H, W] or [C, H, W]

        Returns:
            similarity score (lower means more similar): Tensor [B] or scalar
        """
        # Add batch dimension if necessary
        img1 = self._ensure_batch(img1).to(self.device)
        img2 = self._ensure_batch(img2).to(self.device)

        # Extract features
        feat1, feat2 = self.extractor.extract_features(img1, img2)

        # Predict similarity
        score = self.ddsrn(feat1, feat2).squeeze()

        return score
    
    @staticmethod
    def _ensure_batch(img: torch.Tensor) -> torch.Tensor:
        return img.unsqueeze(0) if img.dim() == 3 else img
    

def DDSRN(ref_img_path, deg_img_path):
    """
    Compute the DDS score using DDSRN (Detection Degradation Score Regression Network) between 
    a reference and a degraded image.

    Args:
        ref_img_path (str): Path to the reference (original/clean) image.
        deg_img_path (str): Path to the degraded (e.g., compressed, noisy) image.

    Returns:
        float: DDSRN score indicating the estimated degradation in object detection quality.
            A score closer to 0 indicates high similarity; closer to 1 indicates high degradation.
    """
    
    # Load model
    metric =  ddsrnScorer(
        model_path="checkpoints/attempt38_40bins_point8_07_coco17complete_320p_qual_20_25_30_35_40_45_50_subsamp_444/best_model.pt",
        backbone=Backbone.YOLO_V11_M,
        weights_path="yolo11m.pt",
        device="cuda"
    )
    # Load images
    ref_img = Image.open(ref_img_path).convert("RGB")
    deg_img = Image.open(deg_img_path).convert("RGB")

    # Compute DDSRN distance score
    return metric(ToTensor()(ref_img), ToTensor()(deg_img))


class ddsrnFeatScorer(torch.nn.Module):
    def __init__(self, model_path: str, backbone: Backbone, device="cuda"):
        super().__init__()
        self.device = device

        self.backbone = backbone

        # Initialize DDSRN model
        self.ddsrn = create_ddsrn_model(
            feature_channels=backbone.config.channels,
            layer_indices=backbone.config.indices,
        ).to(device).eval()

        checkpoint = torch.load(model_path, map_location=device)
        self.ddsrn.load_state_dict(checkpoint["model_state_dict"])
        self.ddsrn.eval()

    def forward(self, feats1: dict, feats2: dict) -> torch.Tensor:
        """
        Args:
            feats1: Dict[str, Tensor] - Features from reference image
            feats2: Dict[str, Tensor] - Features from degraded image

        Returns:
            similarity score (lower = more similar): Tensor [B] or scalar
        """
        # Make sure keys match
        keys1 = set(feats1.keys())
        keys2 = set(feats2.keys())
        assert keys1 == keys2, f"Feature keys mismatch: {keys1} vs {keys2}"

        # Convert to dict[int, Tensor] expected by DDSRN
        feat1_single = {k: v[0].to(self.device) if isinstance(v, list) else v.to(self.device) for k, v in feats1.items()}
        feat2_single = {k: v[0].to(self.device) if isinstance(v, list) else v.to(self.device) for k, v in feats2.items()}

        # Pass to DDSRN model
        score = self.ddsrn(feat1_single, feat2_single).squeeze()

        final_score = score.mean()

        return final_score
    
class ddsrnFeatScorer_FasterRCNN(torch.nn.Module):
    def __init__(self, model_path: str, backbone: Backbone, device="cuda"):
        super().__init__()
        self.device = device

        self.backbone = backbone

        # Initialize DDSRN model
        self.ddsrn = create_ddsrn_model(
            feature_channels=backbone.config.channels,
            layer_indices=backbone.config.indices,
        ).to(device).eval()

        checkpoint = torch.load(model_path, map_location=device)
        self.ddsrn.load_state_dict(checkpoint["model_state_dict"])
        self.ddsrn.eval()

    def move_to_device(self, obj, device):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, list):
            return [self.move_to_device(o, device) for o in obj]
        elif isinstance(obj, dict) or isinstance(obj, collections.OrderedDict):
            return {k: self.move_to_device(v, device) for k, v in obj.items()}
        else:
            return obj


    def forward(self, feats1: dict, feats2: dict) -> torch.Tensor:
        keys1 = set(feats1.keys())
        keys2 = set(feats2.keys())
        assert keys1 == keys2, f"Feature keys mismatch: {keys1} vs {keys2}"

        feat1_single = self.move_to_device(feats1, self.device)
        feat2_single = self.move_to_device(feats2, self.device)

        score = self.ddsrn(feat1_single, feat2_single).squeeze()

        final_score = score.mean()

        return final_score

def DDSRN_FASTER_RCNN_MOBILENET_V3_LARGE_FPN(ref_img_path, deg_img_path, model_path, device="cuda"):
    """
    Compute the DDSRN score using the FasterRCNN_MOBILENET_V3_LARGE_FPN backbone.
    Args:
        ref_img_path (str): Path to the reference image.
        deg_img_path (str): Path to the degraded image.
        model_path (str): Path to the trained DDSRN model checkpoint.
        device (str): Device to run inference on.
    Returns:
        float: DDSRN score.
    """
    backbone = Backbone.FASTERRCNN_MOBILENET_V3_LARGE_FPN

    # Preprocessing: Resize, CenterCrop, ToTensor, Normalize (ImageNet)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load images
    ref_img = preprocess(Image.open(ref_img_path).convert("RGB")).unsqueeze(0)
    deg_img = preprocess(Image.open(deg_img_path).convert("RGB")).unsqueeze(0)

    # Load DDSRN model
    metric = ddsrnScorer(
        model_path=model_path,
        backbone=backbone,
        weights_path=None,
        device=device
    )

    # Compute DDSRN score
    with torch.no_grad():
        score = metric(ref_img, deg_img)
    return score.item()


