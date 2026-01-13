from typing import Optional, Literal

from pydantic import BaseModel
from fastapi import File, UploadFile

from schemas.trellis_schemas import TrellisParamsOverrides


class GenerateRequest(BaseModel):
    # Prompt data
    prompt_type: Literal["text", "image"] = "image"
    prompt_image: str 
    seed: int = -1

    # Multi-view strategy overrides (optional)
    use_qwen_views: Optional[bool] = None
    include_original_view: Optional[bool] = None
    qwen_view_keys: Optional[list[str]] = None

    # Trellis parameters
    trellis_params: Optional[TrellisParamsOverrides] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt_type": "text",
                "prompt_image": "file_name.jpg",
                "seed": 42,
                "use_qwen_views": True,
                "include_original_view": True,
                "qwen_view_keys": ["side_view", "back_view"],
                "trellis_params": {
                    "sparse_structure_steps": 8,
                    "sparse_structure_cfg_strength": 5.75,
                    "slat_steps": 20,
                    "slat_cfg_strength": 2.4,
                }
            }
        }

