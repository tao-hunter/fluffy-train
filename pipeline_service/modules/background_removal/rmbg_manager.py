from __future__ import annotations

import time
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, resized_crop

from config import Settings
from logger_config import logger


class BackgroundRemovalService:
    def __init__(self, settings: Settings):
        """
        Initialize the BackgroundRemovalService.
        """
        self.settings = settings

        # Set padding percentage, output size
        self.padding_percentage = self.settings.padding_percentage
        self.output_size = self.settings.output_image_size
        self.limit_padding = self.settings.limit_padding

        # Set device
        self.device = f"cuda:{settings.qwen_gpu}" if torch.cuda.is_available() else "cpu"

        # Set model
        self.model: AutoModelForImageSegmentation | None = None

        # Set transform
        self.transforms = transforms.Compose(
            [
                transforms.Resize(self.settings.input_image_size), 
                transforms.ToTensor(),
            ]
        )

        # Set normalize
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
       
    def _center_square_resize(self, image: Image.Image) -> Image.Image:
        """
        Fallback: center-crop to square and resize to configured output size.
        Keeps downstream geometry stable even if segmentation fails.
        """
        rgb = image.convert("RGB")
        w, h = rgb.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        cropped = rgb.crop((left, top, left + side, top + side))
        # output_image_size is (height, width)
        return cropped.resize((self.output_size[1], self.output_size[0]), Image.Resampling.LANCZOS)

    def _crop_from_mask_and_resize(self, rgba_tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Crop RGBA tensor around mask and resize to output size.

        Args:
            rgba_tensor: (4, H, W)
            mask: (H, W) in [0, 1]

        Returns:
            (4, out_h, out_w)
        """
        bbox_indices = torch.argwhere(mask > 0.8)  # (N, 2) of (y, x)
        H, W = mask.shape

        if bbox_indices.numel() == 0:
            # No foreground found: center-crop full image to square
            side = min(H, W)
            top = (H - side) // 2
            left = (W - side) // 2
            crop_args = dict(top=int(top), left=int(left), height=int(side), width=int(side))
            return resized_crop(rgba_tensor, **crop_args, size=self.output_size, antialias=False)

        y_min, y_max = torch.aminmax(bbox_indices[:, 0])
        x_min, x_max = torch.aminmax(bbox_indices[:, 1])

        width = (x_max - x_min).item()
        height = (y_max - y_min).item()
        cy = (y_max + y_min).item() / 2.0
        cx = (x_max + x_min).item() / 2.0

        size = max(width, height)
        size = int(size * (1.0 + self.padding_percentage))
        size = max(1, size)

        top = int(cy - size // 2)
        left = int(cx - size // 2)
        bottom = int(cy + size // 2)
        right = int(cx + size // 2)

        if self.limit_padding:
            top = max(0, top)
            left = max(0, left)
            bottom = min(H, bottom)
            right = min(W, right)

        crop_args = dict(
            top=int(top),
            left=int(left),
            height=int(max(1, bottom - top)),
            width=int(max(1, right - left)),
        )
        return resized_crop(rgba_tensor, **crop_args, size=self.output_size, antialias=False)

    async def startup(self) -> None:
        """
        Startup the BackgroundRemovalService.
        """
        logger.info(f"Loading {self.settings.background_removal_model_id} model...")

        # Load model
        try:
            self.model = AutoModelForImageSegmentation.from_pretrained(
                self.settings.background_removal_model_id,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            ).to(self.device)
            logger.success(f"{self.settings.background_removal_model_id} model loaded.")
        except Exception as e:
            logger.error(f"Error loading {self.settings.background_removal_model_id} model: {e}")
            raise RuntimeError(f"Error loading {self.settings.background_removal_model_id} model: {e}")

    async def shutdown(self) -> None:
        """
        Shutdown the BackgroundRemovalService.
        """
        self.model = None
        logger.info("BackgroundRemovalService closed.")

    def ensure_ready(self) -> None:
        """
        Ensure the BackgroundRemovalService is ready.
        """
        if self.model is None:
            raise RuntimeError(f"{self.settings.background_removal_model_id} model not initialized.")

    def remove_background(self, image: Image.Image) -> Image.Image:
        """
        Remove the background from the image.
        """
        try:
            self.ensure_ready()
            t1 = time.time()
            # Check if the image has alpha channel
            has_alpha = False
            
            if image.mode == "RGBA":
                # Get alpha channel
                alpha = np.array(image)[:, :, 3]
                if not np.all(alpha==255):
                    has_alpha=True
            
            if has_alpha:
                # Use alpha as a mask, crop + resize to standard output size
                rgba = np.array(image)
                alpha = rgba[:, :, 3].astype(np.float32) / 255.0  # (H, W)
                mask = torch.from_numpy(alpha).to(self.device).clamp(0, 1)

                rgb_image = image.convert("RGB")
                rgb_tensor = self.transforms(rgb_image).to(self.device)  # (3, H, W)
                rgba_tensor = torch.cat([rgb_tensor * mask.unsqueeze(0), mask.unsqueeze(0)], dim=0)  # (4,H,W)
                out_rgba = self._crop_from_mask_and_resize(rgba_tensor, mask)
                image_without_background = to_pil_image(out_rgba[:3].clamp(0, 1))
                
            else:
                # PIL.Image (H, W, C) C=3
                rgb_image = image.convert('RGB')
                
                # Tensor (H, W, C) -> (C, H',W')
                rgb_tensor = self.transforms(rgb_image).to(self.device)
                output = self._remove_background(rgb_tensor)

                image_without_background = to_pil_image(output[:3])

            removal_time = time.time() - t1
            logger.success(f"Background remove - Time: {removal_time:.2f}s - OutputSize: {image_without_background.size} - InputSize: {image.size}")

            return image_without_background
            
        except Exception as e:
            logger.error(f"Error removing background: {e}")
            try:
                return self._center_square_resize(image)
            except Exception:
                return image 

    def _remove_background(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Remove the background from the image.
        """
        # Normalize tensor value for background removal model, reshape for model batch processing (C=3, H, W) -> (1, C=3, H, W)
        input_tensor = self.normalize(image_tensor).unsqueeze(0)
                
        with torch.no_grad():
            # Get mask from model (1, 1, H, W)
            preds = self.model(input_tensor)[-1].sigmoid()
            # Reshape mask values: (1, 1, H, W) -> (H, W)
            mask = preds[0].squeeze().clamp(0, 1).float()

        # Concat mask with image and blacken the background: (C=3, H, W) | (1, H, W) -> (C=4, H, W)
        tensor_rgba = torch.cat([image_tensor * mask.unsqueeze(0), mask.unsqueeze(0)], dim=0)
        return self._crop_from_mask_and_resize(tensor_rgba, mask)

