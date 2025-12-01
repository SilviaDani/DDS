import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from tqdm import tqdm
import shutil
from pathlib import Path
import multiprocessing as mp
import random
from basicsr.archs.swinir_arch import SwinIR

# Globals for model and device
model_x4 = None
model_x8 = None
device = None

class ProgressiveDistortionGenerator:
    """Apply progressive mixed distortions with visible progression"""
    
    def __init__(self):
        self._precompute_kernels()
        
    def _precompute_kernels(self):
        """Precompute blur kernels"""
        self.blur_kernels = {
            1: self._create_gaussian_kernel(3, 0.3),
            2: self._create_gaussian_kernel(3, 0.6),
            3: self._create_gaussian_kernel(5, 0.8),
            4: self._create_gaussian_kernel(5, 1.2),
            5: self._create_gaussian_kernel(7, 1.5)
        }
    
    def _create_gaussian_kernel(self, size, sigma):
        kernel = np.fromfunction(
            lambda x, y: (1/(2*np.pi*sigma**2)) * 
                         np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)),
            (size, size)
        )
        return kernel / np.sum(kernel)
    
    def add_gaussian_noise(self, image, severity=0):
        if severity == 0:
            return image
            
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
            if image_np.shape[0] == 3:
                image_np = image_np.transpose(1, 2, 0)
            image_np = image_np.astype(np.uint8)
            return_tensor = True
        else:
            image_np = image
            return_tensor = False
        
        image_float = image_np.astype(np.float32) / 255.0
        noise_var = severity * 0.02
        noise = np.random.normal(0, noise_var, image_float.shape)
        noisy_image = image_float + noise
        noisy_image = np.clip(noisy_image, 0, 1)
        noisy_image = (noisy_image * 255).astype(np.uint8)
        
        if return_tensor:
            if noisy_image.shape[2] == 3:
                noisy_image = noisy_image.transpose(2, 0, 1)
            return torch.from_numpy(noisy_image).float()
        return noisy_image
    
    def add_motion_blur(self, image, severity=0):
        if severity == 0:
            return image
            
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
            if image_np.shape[0] == 3:
                image_np = image_np.transpose(1, 2, 0)
            image_np = image_np.astype(np.uint8)
            return_tensor = True
        else:
            image_np = image
            return_tensor = False
        
        if severity < 0.2:
            kernel = self.blur_kernels[1]
        elif severity < 0.4:
            kernel = self.blur_kernels[2]
        elif severity < 0.6:
            kernel = self.blur_kernels[3]
        elif severity < 0.8:
            kernel = self.blur_kernels[4]
        else:
            kernel = self.blur_kernels[5]
        
        blurred_image = cv2.filter2D(image_np, -1, kernel)
        
        if return_tensor:
            if blurred_image.shape[2] == 3:
                blurred_image = blurred_image.transpose(2, 0, 1)
            return torch.from_numpy(blurred_image).float()
        return blurred_image
    
    def adjust_contrast(self, image, severity=0):
        if severity == 0:
            return image
            
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
            if image_np.shape[0] == 3:
                image_np = image_np.transpose(1, 2, 0)
            image_np = image_np.astype(np.uint8)
            return_tensor = True
        else:
            image_np = image
            return_tensor = False
        
        if severity <= 0.5:
            factor = 1.0 - (severity * 0.6)
        else:
            factor = 1.0 + ((severity - 0.5) * 1.0)
        
        image_pil = Image.fromarray(image_np)
        enhancer = ImageEnhance.Contrast(image_pil)
        contrast_image = enhancer.enhance(factor)
        contrast_array = np.array(contrast_image)
        
        if return_tensor:
            contrast_array = contrast_array.transpose(2, 0, 1)
            return torch.from_numpy(contrast_array).float()
        return contrast_array
    
    def adjust_brightness(self, image, severity=0):
        if severity == 0:
            return image
            
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
            if image_np.shape[0] == 3:
                image_np = image_np.transpose(1, 2, 0)
            image_np = image_np.astype(np.uint8)
            return_tensor = True
        else:
            image_np = image
            return_tensor = False
        
        if severity <= 0.5:
            factor = 1.0 - (severity * 0.6)
        else:
            factor = 1.0 + ((severity - 0.5) * 1.0)
        
        image_pil = Image.fromarray(image_np)
        enhancer = ImageEnhance.Brightness(image_pil)
        bright_image = enhancer.enhance(factor)
        bright_array = np.array(bright_image)
        
        if return_tensor:
            bright_array = bright_array.transpose(2, 0, 1)
            return torch.from_numpy(bright_array).float()
        return bright_array
    
    def add_jpeg_compression(self, image, severity=0):
        if severity == 0:
            return image
            
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
            if image_np.shape[0] == 3:
                image_np = image_np.transpose(1, 2, 0)
            image_np = image_np.astype(np.uint8)
            return_tensor = True
        else:
            image_np = image
            return_tensor = False
        
        quality = max(30, 95 - int(severity * 65))
        
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encimg = cv2.imencode('.jpg', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR), encode_param)
        compressed_image = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
        compressed_image = cv2.cvtColor(compressed_image, cv2.COLOR_BGR2RGB)
        
        if return_tensor:
            compressed_image = compressed_image.transpose(2, 0, 1)
            return torch.from_numpy(compressed_image).float()
        return compressed_image
    
    def add_sr_distortion(self, image, apply_sr=False, model_x4=None, model_x8=None, device=None, scale_factor=4):
        if not apply_sr:
            return image
            
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
            if image_np.shape[0] == 3:
                image_np = image_np.transpose(1, 2, 0)
            image_np = image_np.astype(np.uint8)
            return_tensor = True
        else:
            image_np = image
            return_tensor = False
        
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        h, w = image_bgr.shape[:2]
        
        # ALWAYS downscale by the specified scale factor first
        new_w, new_h = w // scale_factor, h // scale_factor
        downscaled = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Then upscale back to original size using appropriate method
        if scale_factor == 2:
            # For x2, use bicubic interpolation (no x2 model available)
            sr_image = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_CUBIC)
        elif scale_factor == 4 and model_x4 is not None:
            # For x4, use x4 SR model
            sr_image = self.super_resolve(model_x4, device, downscaled, scale_factor=4)
        elif scale_factor == 8 and model_x8 is not None:
            # For x8, use x8 SR model
            sr_image = self.super_resolve(model_x8, device, downscaled, scale_factor=8)
        else:
            # Fallback to bicubic if model not available
            sr_image = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_CUBIC)
        
        sr_image_rgb = cv2.cvtColor(sr_image, cv2.COLOR_BGR2RGB)
        
        if return_tensor:
            sr_image_rgb = sr_image_rgb.transpose(2, 0, 1)
            return torch.from_numpy(sr_image_rgb).float()
        return sr_image_rgb
    
    def super_resolve(self, model, device, img_bgr, scale_factor):
        """
        Super-resolve image using SwinIR model
        The model is trained to upscale from the given scale_factor back to original size
        """
        img_rgb = img_bgr[:, :, ::-1].astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor).clamp(0, 1)
        output_img = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output_img = (output_img * 255.0).round().astype(np.uint8)
        return output_img[:, :, ::-1]
    
    def apply_distortion_pipeline(self, image, distortion_config, model_x4=None, model_x8=None, device=None):
        if isinstance(image, torch.Tensor):
            working_image = image.clone()
        else:
            working_image = image.copy()
        
        # Apply all distortions except SR first
        distortion_order = ['contrast', 'brightness', 'blur', 'noise', 'jpeg']
        
        for distortion_type in distortion_order:
            severity = distortion_config.get(distortion_type, 0)
            if severity > 0:
                if distortion_type == 'noise':
                    working_image = self.add_gaussian_noise(working_image, severity)
                elif distortion_type == 'blur':
                    working_image = self.add_motion_blur(working_image, severity)
                elif distortion_type == 'contrast':
                    working_image = self.adjust_contrast(working_image, severity)
                elif distortion_type == 'brightness':
                    working_image = self.adjust_brightness(working_image, severity)
                elif distortion_type == 'jpeg':
                    working_image = self.add_jpeg_compression(working_image, severity)
        
        # Apply SR distortion LAST (after all other distortions)
        apply_sr = distortion_config.get('sr', False)
        scale_factor = distortion_config.get('sr_scale', 4)
        if apply_sr:
            working_image = self.add_sr_distortion(working_image, apply_sr, model_x4, model_x8, device, scale_factor)
                
        return working_image
    
    def generate_progressive_distortions_for_source(self, num_levels=4):
        """
        Generate progressive distortions for one source image
        - Same distortion types for all 4 images
        - Increasing severity from level 1 to 4
        - SR only for level 4 (and only sometimes)
        """
        # Randomly select 2-4 distortion types that will be consistent for all levels
        distortion_types = ['contrast', 'brightness', 'blur', 'noise', 'jpeg']
        num_distortions = random.randint(2, 4)
        selected_distortions = random.sample(distortion_types, num_distortions)
        
        levels = []
        
        for level in range(num_levels):
            severity_base = (level + 1) / num_levels
            
            config = {'contrast': 0, 'brightness': 0, 'blur': 0, 'noise': 0, 'jpeg': 0, 'sr': False}
            
            # Assign progressive severity to the selected distortion types
            for dist_type in selected_distortions:
                # Vary severity slightly for each distortion type but maintain progression
                severity_variation = random.uniform(0.8, 1.2)
                config[dist_type] = min(1.0, severity_base * severity_variation)
            
            # Add SR distortion ONLY for level 4 (image_4) and only sometimes (50% chance)
            if level == 3 and random.random() < 0.5:
                config['sr'] = True
                # Random scale factor for SR
                config['sr_scale'] = random.choice([2, 4, 8])
            
            levels.append(config)
        
        return levels


def create_directories(base_path, splits):
    for split in splits:
        compressed_path = Path(base_path) / split / "compressed"
        extracted_path = Path(base_path) / split / "extracted"
        compressed_path.mkdir(parents=True, exist_ok=True)
        extracted_path.mkdir(parents=True, exist_ok=True)


def crop_image_center(image, crop_size=320):
    if isinstance(image, torch.Tensor):
        image_np = image.cpu().numpy()
        if image_np.shape[0] == 3:
            image_np = image_np.transpose(1, 2, 0)
        image_np = image_np.astype(np.uint8)
    else:
        image_np = image.astype(np.uint8)
    
    height, width = image_np.shape[:2]
    if height < crop_size or width < crop_size:
        return image_np

    y_start = (height - crop_size) // 2
    x_start = (width - crop_size) // 2
    return image_np[y_start:y_start + crop_size, x_start:x_start + crop_size]


def load_images_from_directory(directory_path):
    directory = Path(directory_path)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '.tiff', '.tif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(directory.glob(ext))
        image_files.extend(directory.glob(ext.upper()))
    return image_files


def init_models(sr_model_path='checkpoints'):
    global device, model_x4, model_x8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize x4 model
    x4_path = os.path.join(sr_model_path, '001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth')
    if os.path.exists(x4_path):
        try:
            model_x4 = SwinIR(
                upscale=4,  # This model is trained to upscale x4
                in_chans=3,
                img_size=48,
                window_size=8,
                img_range=1.0,
                depths=[6, 6, 6, 6, 6, 6],
                embed_dim=180,
                num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2,
                upsampler='pixelshuffle',
                resi_connection='1conv'
            )
            pretrained_state = torch.load(x4_path, map_location=device)
            if 'params' in pretrained_state:
                pretrained_state = pretrained_state['params']
            model_x4.load_state_dict(pretrained_state, strict=True)
            model_x4 = model_x4.to(device)
            model_x4.eval()
        except:
            model_x4 = None
    
    # Initialize x8 model
    x8_path = os.path.join(sr_model_path, '001_classicalSR_DIV2K_s48w8_SwinIR-M_x8.pth')
    if os.path.exists(x8_path):
        try:
            model_x8 = SwinIR(
                upscale=8,  # This model is trained to upscale x8
                in_chans=3,
                img_size=48,
                window_size=8,
                img_range=1.0,
                depths=[6, 6, 6, 6, 6, 6],
                embed_dim=180,
                num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2,
                upsampler='pixelshuffle',
                resi_connection='1conv'
            )
            pretrained_state = torch.load(x8_path, map_location=device)
            if 'params' in pretrained_state:
                pretrained_state = pretrained_state['params']
            model_x8.load_state_dict(pretrained_state, strict=True)
            model_x8 = model_x8.to(device)
            model_x8.eval()
        except:
            model_x8 = None
    
    return model_x4, model_x8


def process_single_image(args):
    image_file, output_path, split, crop_size, distortion_generator, model_x4, model_x8, device = args
    
    try:
        image_name = image_file.stem
        image_ext = image_file.suffix.lower() or '.jpg'
        
        image = cv2.imread(str(image_file))
        if image is None:
            return False
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cropped_image = crop_image_center(image_rgb, crop_size)
        
        if cropped_image.shape[0] < crop_size or cropped_image.shape[1] < crop_size:
            return False
        
        # Generate progressive distortions for this source image
        # Same distortion types for all 4 images, increasing severity
        distortion_levels = distortion_generator.generate_progressive_distortions_for_source(num_levels=4)
        
        # Apply each distortion level
        for level_num, config in enumerate(distortion_levels, 1):
            distorted_image = distortion_generator.apply_distortion_pipeline(
                cropped_image, config, model_x4, model_x8, device
            )
            
            if isinstance(distorted_image, torch.Tensor):
                distorted_np = distorted_image.cpu().numpy()
                if distorted_np.shape[0] == 3:
                    distorted_np = distorted_np.transpose(1, 2, 0)
                distorted_np = distorted_np.astype(np.uint8)
            else:
                distorted_np = distorted_image.astype(np.uint8)
            
            # Save with _number suffix
            output_filename = f"{image_name}_{level_num}{image_ext}"
            compressed_path = os.path.join(output_path, split, "compressed", output_filename)
            extracted_path = os.path.join(output_path, split, "extracted", output_filename)
            
            cv2.imwrite(compressed_path, cv2.cvtColor(distorted_np, cv2.COLOR_RGB2BGR))
            cv2.imwrite(extracted_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
        
        return True
        
    except:
        return False


def create_progressive_distorted_dataset(
    original_dataset_base_path,
    output_path="progressive_distorted_dataset",
    splits={"train": "train2017", "val": "val2017", "test": "test2017"},
    crop_size=320,
    num_workers=None,
    sr_model_path='checkpoints',
    test_sample_fraction=0.05
):
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    distortion_generator = ProgressiveDistortionGenerator()
    model_x4, model_x8 = init_models(sr_model_path=sr_model_path)
    create_directories(output_path, splits.keys())
    
    for split_name, split_folder in splits.items():
        split_path = os.path.join(original_dataset_base_path, split_folder)
        image_files = load_images_from_directory(split_path)
        
        if not image_files:
            continue
        
        if split_name == "test" and test_sample_fraction < 1.0:
            sample_size = max(1, int(len(image_files) * test_sample_fraction))
            image_files = random.sample(image_files, sample_size)
        
        tasks = []
        for image_file in image_files:
            tasks.append((image_file, output_path, split_name, crop_size, 
                         distortion_generator, model_x4, model_x8, device))
        
        with mp.Pool(processes=num_workers) as pool:
            list(tqdm(pool.imap(process_single_image, tasks), 
                     total=len(tasks), 
                     desc=f"Processing {split_name}"))


def main():
    ORIGINAL_DATASET_BASE_PATH = "/andromeda/personal/jdamerini"
    OUTPUT_PATH = "progressive_distorted_dataset"
    SPLITS = {"test": "test2017"}
    CROP_SIZE = 320
    SR_MODEL_PATH = 'checkpoints'
    TEST_SAMPLE_FRACTION = 0.05
    
    cv2.setNumThreads(0)
    random.seed(42)
    
    create_progressive_distorted_dataset(
        original_dataset_base_path=ORIGINAL_DATASET_BASE_PATH,
        output_path=OUTPUT_PATH,
        splits=SPLITS,
        crop_size=CROP_SIZE,
        num_workers=mp.cpu_count(),
        sr_model_path=SR_MODEL_PATH,
        test_sample_fraction=TEST_SAMPLE_FRACTION
    )


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    main()