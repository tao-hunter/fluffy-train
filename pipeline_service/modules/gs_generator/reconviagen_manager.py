"""
ReconViaGen Manager
Handles ReconViaGen pipeline for multi-view 3D generation
"""

from __future__ import annotations

import time
import gc
from typing import Optional, List
import torch
from PIL import Image

from config import Settings
from logger_config import logger
from schemas import TrellisResult, TrellisRequest
from libs.reconviagen.multiview_pipeline import ReconViaGenMultiViewPipeline


class ReconViaGenManager:
    """Manager for ReconViaGen multi-view 3D generation"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pipeline: Optional[ReconViaGenMultiViewPipeline] = None
        self.gpu = getattr(settings, 'reconviagen_gpu', 0)
        
    async def startup(self) -> None:
        """Initialize ReconViaGen pipeline"""
        logger.info("Loading ReconViaGen VGGT pipeline...")
        
        try:
            model_path = getattr(self.settings, 'reconviagen_model_id', "Stable-X/trellis-vggt-v0-1")
            self.pipeline = ReconViaGenMultiViewPipeline(model_path)
            
            # Load pipeline on startup for faster inference
            if not self.pipeline.load_pipeline():
                raise RuntimeError("Failed to load ReconViaGen VGGT pipeline")
                
            logger.success("ReconViaGen VGGT pipeline ready.")
            
        except Exception as e:
            logger.error(f"Failed to initialize ReconViaGen VGGT pipeline: {e}")
            self.pipeline = None
    
    async def shutdown(self) -> None:
        """Shutdown ReconViaGen pipeline"""
        if self.pipeline:
            self.pipeline.unload_pipeline()
            self.pipeline = None
        logger.info("ReconViaGen VGGT pipeline closed.")
    
    def is_ready(self) -> bool:
        """Check if pipeline is ready"""
        return self.pipeline is not None and self.pipeline.is_loaded
    
    def generate(self, request: TrellisRequest) -> TrellisResult:
        """
        Generate 3D model using ReconViaGen (single or multi-image)
        
        Args:
            request: TrellisRequest with images and parameters
            
        Returns:
            TrellisResult with generated PLY data
        """
        if not self.pipeline:
            raise RuntimeError("ReconViaGen pipeline not loaded.")
        
        # Handle both single and multi-image cases
        images = request.images if isinstance(request.images, list) else [request.images] if hasattr(request, 'images') else []
        if not images:
            raise ValueError("No images provided in request")
        
        logger.info(f"Generating ReconViaGen {request.seed=} with {len(images)} image(s), image size {images[0].size}")
        
        # Extract parameters from request
        params = request.params
        ss_guidance_strength = getattr(params, 'sparse_structure_cfg_strength', 7.5) if params else 7.5
        ss_sampling_steps = getattr(params, 'sparse_structure_steps', 30) if params else 30
        slat_guidance_strength = getattr(params, 'slat_cfg_strength', 3.0) if params else 3.0
        slat_sampling_steps = getattr(params, 'slat_steps', 12) if params else 12
        
        start = time.time()
        try:
            if len(images) > 1:
                # Multi-image generation
                multiimage_algo = getattr(self.settings, 'reconviagen_multiimage_algo', 'multidiffusion')
                ply_data = self.pipeline.generate_3d_from_multiview_images(
                    images=images,
                    seed=request.seed,
                    ss_guidance_strength=ss_guidance_strength,
                    ss_sampling_steps=ss_sampling_steps,
                    slat_guidance_strength=slat_guidance_strength,
                    slat_sampling_steps=slat_sampling_steps,
                    multiimage_algo=multiimage_algo,
                    preprocess_image=False
                )
            else:
                # Single image generation
                ply_data = self.pipeline.generate_3d_from_single_image(
                    image=images[0],
                    seed=request.seed,
                    ss_guidance_strength=ss_guidance_strength,
                    ss_sampling_steps=ss_sampling_steps,
                    slat_guidance_strength=slat_guidance_strength,
                    slat_sampling_steps=slat_sampling_steps,
                    preprocess_image=False  # We handle preprocessing in the main pipeline
                )
            
            if ply_data is None:
                raise RuntimeError("ReconViaGen generation failed")
            
            generation_time = time.time() - start
            
            # Force GPU cleanup after generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc; gc.collect()
            
            result = TrellisResult(
                ply_file=ply_data  # bytes
            )
            
            logger.success(f"ReconViaGen finished generation in {generation_time:.2f}s.")
            return result
            
        except Exception as e:
            logger.error(f"ReconViaGen generation failed: {e}")
            raise
    
    def generate_multiview(
        self, 
        images: List[Image.Image], 
        seed: int = 42,
        ss_guidance_strength: float = 7.5,
        ss_sampling_steps: int = 30,
        slat_guidance_strength: float = 3.0,
        slat_sampling_steps: int = 12,
        multiimage_algo: str = "multidiffusion"
    ) -> TrellisResult:
        """
        Generate 3D model using ReconViaGen (multi-view images)
        
        Args:
            images: List of PIL Images (multi-view)
            seed: Random seed
            ss_guidance_strength: Sparse structure guidance strength
            ss_sampling_steps: Sparse structure sampling steps
            slat_guidance_strength: SLat guidance strength
            slat_sampling_steps: SLat sampling steps
            multiimage_algo: Multi-image algorithm
            
        Returns:
            TrellisResult with generated PLY data
        """
        if not self.pipeline:
            raise RuntimeError("ReconViaGen pipeline not loaded.")
        
        logger.info(f"Generating ReconViaGen multiview with {len(images)} images, seed={seed}")
        
        start = time.time()
        try:
            ply_data = self.pipeline.generate_3d_from_multiview_images(
                images=images,
                seed=seed,
                ss_guidance_strength=ss_guidance_strength,
                ss_sampling_steps=ss_sampling_steps,
                slat_guidance_strength=slat_guidance_strength,
                slat_sampling_steps=slat_sampling_steps,
                multiimage_algo=multiimage_algo,
                preprocess_image=False  # We handle preprocessing in the main pipeline
            )
            
            if ply_data is None:
                raise RuntimeError("ReconViaGen multiview generation failed")
            
            generation_time = time.time() - start
            
            result = TrellisResult(
                ply_file=ply_data  # bytes
            )
            
            logger.success(f"ReconViaGen multiview finished generation in {generation_time:.2f}s.")
            return result
            
        except Exception as e:
            logger.error(f"ReconViaGen multiview generation failed: {e}")
            raise

