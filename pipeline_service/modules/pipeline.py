from __future__ import annotations

import base64
import io
import time
from typing import Optional
import json

from PIL import Image
import pyspz
import torch
import gc

from config import Settings, settings
from logger_config import logger
from schemas import (
    GenerateRequest,
    GenerateResponse,
    TrellisParams,
    TrellisRequest,
    TrellisResult,
)
from modules.image_edit.qwen_edit_module import QwenEditModule
from modules.background_removal.rmbg_manager import BackgroundRemovalService
from modules.gs_generator.trellis_manager import TrellisService
from modules.gs_generator.reconviagen_manager import ReconViaGenManager
from modules.utils import (
    secure_randint,
    set_random_seed,
    decode_image,
    to_png_base64,
    save_files,
)


class GenerationPipeline:
    def __init__(self, settings: Settings = settings):
        self.settings = settings

        # Initialize modules
        self.qwen_edit = QwenEditModule(settings)
        self.rmbg = BackgroundRemovalService(settings)
        self.trellis = TrellisService(settings)
        self.reconviagen = ReconViaGenManager(settings)

    def _load_qwen_view_prompts(self) -> dict:
        """
        Load multi-view prompts from config.
        Expected format: { "<view_key>": { "positive": "..." } }
        """
        try:
            path = self.settings.qwen_view_prompts_path
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                return data
        except Exception as e:
            logger.warning(f"Failed to load qwen view prompts: {e}")
        return {}

    async def startup(self) -> None:
        """Initialize all pipeline components."""
        logger.info("Starting pipeline")
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)

        await self.qwen_edit.startup()
        await self.rmbg.startup()
        if self.settings.use_reconviagen:
            try:
                await self.reconviagen.startup()
                if not self.reconviagen.is_ready():
                    logger.warning("ReconViaGen failed to load, falling back to Trellis")
                    await self.trellis.startup()
            except Exception as e:
                logger.error(f"ReconViaGen startup failed: {e}, falling back to Trellis")
                await self.trellis.startup()
        else:
            await self.trellis.startup()

        logger.info("Warming up generator...")
        await self.warmup_generator()
        self._clean_gpu_memory()

        logger.success("Warmup is complete. Pipeline ready to work.")

    async def shutdown(self) -> None:
        """Shutdown all pipeline components."""
        logger.info("Closing pipeline")

        # Shutdown all modules
        await self.qwen_edit.shutdown()
        await self.rmbg.shutdown()
        if self.settings.use_reconviagen and self.reconviagen.is_ready():
            await self.reconviagen.shutdown()
        if self.trellis.is_ready():
            await self.trellis.shutdown()

        logger.info("Pipeline closed.")

    def _clean_gpu_memory(self) -> None:
        """
        Clean the GPU memory.
        """
        gc.collect()
        torch.cuda.empty_cache()

    async def warmup_generator(self) -> None:
        """Function for warming up the generator"""

        temp_image = Image.new("RGB", (64, 64), color=(128, 128, 128))
        buffer = io.BytesIO()
        temp_image.save(buffer, format="PNG")
        temp_imge_bytes = buffer.getvalue()
        await self.generate_from_upload(temp_imge_bytes, seed=42)

    async def generate_from_upload(self, image_bytes: bytes, seed: int) -> bytes:
        """
        Generate 3D model from uploaded image file and return PLY as bytes.

        Args:
            image_bytes: Raw image bytes from uploaded file

        Returns:
            PLY file as bytes
        """
        # Encode to base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Create request
        request = GenerateRequest(
            prompt_image=image_base64, prompt_type="image", seed=seed
        )

        # Generate
        response = await self.generate_gs(request)

        # Return binary PLY
        if not response.ply_file_base64:
            raise ValueError("PLY generation failed")

        return response.ply_file_base64  # bytes

    async def generate_gs(self, request: GenerateRequest) -> GenerateResponse:
        """
        Execute full generation pipeline.

        Args:
            request: Generation request with prompt and settings

        Returns:
            GenerateResponse with generated assets
        """
        t1 = time.time()
        logger.info(f"New generation request")

        # Set seed
        if request.seed < 0:
            request.seed = secure_randint(0, 10000)
            set_random_seed(request.seed)
        else:
            set_random_seed(request.seed)

        # Decode input image
        image = decode_image(request.prompt_image)

        # ---- Multi-view image preparation (quality-critical) ----
        # Default strategy: include original view (less hallucination) + a couple synthesized views (more coverage)
        use_qwen_views = request.use_qwen_views if request.use_qwen_views is not None else self.settings.use_qwen_views
        include_original_view = (
            request.include_original_view if request.include_original_view is not None else self.settings.include_original_view
        )
        view_keys = request.qwen_view_keys if request.qwen_view_keys is not None else self.settings.qwen_view_keys

        view_prompts = self._load_qwen_view_prompts()

        images_for_3d: list[Image.Image] = []
        # Always compute bg-removed original (even if we don't include it) because it's a good fallback
        image_without_background_original = self.rmbg.remove_background(image)
        if include_original_view:
            images_for_3d.append(image_without_background_original)

        edited_images: list[Image.Image] = []
        edited_no_bg_images: list[Image.Image] = []

        if use_qwen_views:
            for key in view_keys:
                prompt_obj = view_prompts.get(key, {})
                prompt = None
                if isinstance(prompt_obj, dict):
                    prompt = prompt_obj.get("positive")
                elif isinstance(prompt_obj, str):
                    prompt = prompt_obj

                if not prompt:
                    logger.warning(f"Missing prompt for view '{key}', skipping")
                    continue

                img_edit = self.qwen_edit.edit_image(
                    prompt_image=image,
                    seed=request.seed,
                    prompt=prompt,
                )
                img_no_bg = self.rmbg.remove_background(img_edit)
                edited_images.append(img_edit)
                edited_no_bg_images.append(img_no_bg)
                images_for_3d.append(img_no_bg)

        if not images_for_3d:
            images_for_3d = [image_without_background_original]

        # Keep backward-compatible debug saving (up to 3 synthesized views)
        image_edited = edited_images[0] if len(edited_images) > 0 else None
        image_without_background = edited_no_bg_images[0] if len(edited_no_bg_images) > 0 else image_without_background_original
        image_edited_2 = edited_images[1] if len(edited_images) > 1 else None
        image_without_background_2 = edited_no_bg_images[1] if len(edited_no_bg_images) > 1 else None
        image_edited_3 = edited_images[2] if len(edited_images) > 2 else None
        image_without_background_3 = edited_no_bg_images[2] if len(edited_no_bg_images) > 2 else None

        # save to debug
        # image_edited.save("image_edited.png")
        # image_edited_2.save("image_edited_2.png")
        # image_without_background.save("image_without_background.png")
        # image_without_background_2.save("image_without_background_2.png")

        trellis_result: Optional[TrellisResult] = None

        # Resolve Trellis parameters from request
        trellis_params: TrellisParams = request.trellis_params

        # 3. Generate the 3D model
        # Choose between Trellis and ReconViaGen based on settings
        use_reconviagen = getattr(self.settings, 'use_reconviagen', False)
        
        if use_reconviagen and self.reconviagen.is_ready():
            logger.info("Using ReconViaGen for multi-view 3D generation")
            
            # Extract parameters for ReconViaGen
            # Use TrellisParams defaults if not provided
            if trellis_params:
                ss_guidance_strength = trellis_params.sparse_structure_cfg_strength
                ss_sampling_steps = trellis_params.sparse_structure_steps
                slat_guidance_strength = trellis_params.slat_cfg_strength
                slat_sampling_steps = trellis_params.slat_steps
            else:
                # Use defaults from settings
                ss_guidance_strength = self.settings.trellis_sparse_structure_cfg_strength
                ss_sampling_steps = self.settings.trellis_sparse_structure_steps
                slat_guidance_strength = self.settings.trellis_slat_cfg_strength
                slat_sampling_steps = self.settings.trellis_slat_steps
            
            multiimage_algo = getattr(self.settings, 'reconviagen_multiimage_algo', 'multidiffusion')
            
            trellis_result = self.reconviagen.generate_multiview(
                images=images_for_3d,
                seed=request.seed,
                ss_guidance_strength=ss_guidance_strength,
                ss_sampling_steps=ss_sampling_steps,
                slat_guidance_strength=slat_guidance_strength,
                slat_sampling_steps=slat_sampling_steps,
                multiimage_algo=multiimage_algo
            )
        else:
            logger.info("Using standard Trellis for multi-view 3D generation")
            trellis_result = self.trellis.generate(
                TrellisRequest(
                    images=images_for_3d,
                    seed=request.seed,
                    params=trellis_params,
                )
            )

        # Save generated files
        if self.settings.save_generated_files:
            save_files(
                trellis_result, 
                image, 
                image_edited, 
                image_without_background,
                image_edited_2,
                image_without_background_2,
                image_edited_3,
                image_without_background_3
            )

        # Convert to PNG base64 for response (only if needed)
        image_edited_base64 = None
        image_without_background_base64 = None
        if self.settings.send_generated_files:
            image_edited_base64 = to_png_base64(image_edited)
            image_without_background_base64 = to_png_base64(image_without_background)

        t2 = time.time()
        generation_time = t2 - t1

        logger.info(f"Total generation time: {generation_time} seconds")
        # Clean the GPU memory
        self._clean_gpu_memory()

        response = GenerateResponse(
            generation_time=generation_time,
            ply_file_base64=trellis_result.ply_file if trellis_result else None,
            image_edited_file_base64=image_edited_base64
            if self.settings.send_generated_files
            else None,
            image_without_background_file_base64=image_without_background_base64
            if self.settings.send_generated_files
            else None,
        )
        return response
