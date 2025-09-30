import torch
import utils.det.transforms as T


class DetectionPresetTrain:
    def __init__(
        self,
        *,
        crop_size=0,
        mean=(123.0, 117.0, 104.0),
        augmentations=None,
    ):

        transforms = []

        # Fixed size crop if crop_size > 0
        if crop_size > 0:
            transforms.append(T.FixedSizeCrop(size=(crop_size, crop_size), fill=mean))

        # Add augmentations based on config list
        if augmentations is not None:

            # Horizontal flip
            if 'hflip' in augmentations:
                transforms.append(T.RandomHorizontalFlip(p=0.5))

            # IoU Crop
            if 'iou_crop' in augmentations:
                transforms.append(T.RandomIoUCrop())

            # Zoom out
            if 'zoom_out' in augmentations:
                transforms.append(T.RandomZoomOut())

            # Photometric distortions
            if 'photometric' in augmentations:
                transforms.append(T.RandomPhotometricDistort())

            # Scale jitter
            if 'scale_jitter' in augmentations:
                target_size = (crop_size, crop_size) if crop_size > 0 else (512, 512)
                transforms.append(T.ScaleJitter(target_size=target_size))

            # Random shortest side resize
            if 'shortest_size' in augmentations:
                min_size = 256
                max_size = crop_size if crop_size > 0 else 512
                transforms.append(T.RandomShortestSize(min_size=min_size, max_size=max_size))

        # Always convert to tensor and normalize dtype and scale
        transforms.extend([
            T.PILToTensor(),
            T.ToDtype(torch.float, scale=True),
        ])

        self.transforms = T.Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)


class DetectionPresetEval:
    def __init__(self):
        transforms = [
            T.PILToTensor(),
            T.ToDtype(torch.float, scale=True),
        ]
        self.transforms = T.Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)
