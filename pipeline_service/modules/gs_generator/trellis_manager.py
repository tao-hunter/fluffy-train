from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Optional
import io

import torch
from PIL import Image, ImageStat

from config import Settings
from logger_config import logger
from libs.trellis.pipelines import TrellisImageTo3DPipeline
from schemas import TrellisResult, TrellisRequest, TrellisParams

class TrellisService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pipeline: Optional[TrellisImageTo3DPipeline] = None
        self.gpu = settings.trellis_gpu
        self.default_params = TrellisParams.from_settings(self.settings)

    async def startup(self) -> None:
        logger.info("Loading Trellis pipeline...")
        os.environ.setdefault("ATTN_BACKEND", "flash-attn")
        os.environ.setdefault("SPCONV_ALGO", "native")

        if torch.cuda.is_available():
            torch.cuda.set_device(self.gpu)

        self.pipeline = TrellisImageTo3DPipeline.from_pretrained(
            self.settings.trellis_model_id
        )
        self.pipeline.cuda()
        logger.success("Trellis pipeline ready.")

    async def shutdown(self) -> None:
        self.pipeline = None
        logger.info("Trellis pipeline closed.")

    def is_ready(self) -> bool:
        return self.pipeline is not None

    def post_process(self, trellis_result: dict) -> dict:
        """
        Post process the trellis result.
        """
        # Extract Gaussian and mesh
        gs = trellis_result['gaussian'][0]
        # mesh = outputs['mesh'][0]
        
        # Filter out low opacity splats
        opacity_threshold = 0.005  # Adjust threshold as needed
        opacity_mask = gs._opacity.squeeze() > opacity_threshold
        
        if opacity_mask.sum() < gs._opacity.shape[0]:
            remaining_points = opacity_mask.sum().item()
            total_points = gs._opacity.shape[0]
            removal_percentage = (total_points - remaining_points) / total_points * 100
            
            print(f"   🔍 Filtering splats: {total_points} -> {remaining_points} (removed {total_points - remaining_points} low opacity splats, {removal_percentage:.1f}%)")
            
            # Check if too many points were removed
            if remaining_points < 1000:  # Less than 1000 points left
                print(f"   ⚠️ Too few points remaining ({remaining_points}), keeping original Gaussian")
            elif removal_percentage > 90:  # More than 90% removed
                print(f"   ⚠️ Too many points removed ({removal_percentage:.1f}%), keeping original Gaussian")
            else:
                # Apply mask to all Gaussian splat parameters (check if they exist first)
                gs._xyz = gs._xyz[opacity_mask]
                gs._features_dc = gs._features_dc[opacity_mask]
                gs._scaling = gs._scaling[opacity_mask]
                gs._rotation = gs._rotation[opacity_mask]
                gs._opacity = gs._opacity[opacity_mask]
                
                # Only filter _features_rest if it exists and is not None
                if hasattr(gs, '_features_rest') and gs._features_rest is not None:
                    gs._features_rest = gs._features_rest[opacity_mask]
                
                print(f"   ✅ Applied opacity filtering successfully")

        else:
            print(f"   ✅ No points removed, keeping original Gaussian")
        trellis_result['gaussian'][0] = gs
            
        return trellis_result

    def generate(
        self,
        trellis_request: TrellisRequest,
    ) -> TrellisResult:
        if not self.pipeline:
            raise RuntimeError("Trellis pipeline not loaded.")

        images_rgb = [image.convert("RGB") for image in trellis_request.images]
        logger.info(f"Generating Trellis {trellis_request.seed=} and image size {trellis_request.images[0].size}")

        params = self.default_params.overrided(trellis_request.params)

        start = time.time()
        try:
            # Generate with voxel-aware texture steps
            outputs, num_voxels = self.pipeline.run_multi_image_with_voxel_count(
                images_rgb,
                seed=trellis_request.seed,
                sparse_structure_sampler_params={
                    "steps": params.sparse_structure_steps,
                    "cfg_strength": params.sparse_structure_cfg_strength,
                },
                slat_sampler_params={
                    "steps": params.slat_steps,
                    "cfg_strength": params.slat_cfg_strength,
                },
                preprocess_image=False,
                formats=["gaussian"],
                num_oversamples=params.num_oversamples,
                voxel_threshold=25000,
            )

            generation_time = time.time() - start

            outputs = self.post_process(outputs)
            gaussian = outputs["gaussian"][0]

            # Save ply to buffer
            buffer = io.BytesIO()
            gaussian.save_ply(buffer)
            buffer.seek(0)

            result = TrellisResult(
                ply_file=buffer.getvalue() if buffer else None # bytes
            )

            logger.success(f"Trellis finished generation in {generation_time:.2f}s with {num_voxels} occupied voxels.")
            return result
        finally:
            if buffer:
                buffer.close()

