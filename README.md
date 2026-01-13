# 3D Generation Pipeline

fast

## Requirements

- **Docker** and **Docker Compose**
- **NVIDIA GPU** with CUDA 12.x support
- At least **80GB VRAM** (61GB+ recommended)

## Installation

### Docker (building)
```bash
docker build -f docker/Dockerfile -t forge3d-pipeline:latest .
```

## Run pipeline

Copy `env.sample` to `.env` (or set env vars another way) and configure if needed

- Start with docker-compose 

```bash
cd docker
docker-compose up -d --build
```

- Start with docker run
```bash
docker run --gpus all -p 10006:10006 forge3d-pipeline:latest
```

- Start with docker run and env file
```bash
docker run --gpus all -p 10006:10006 --env-file .env forge3d-pipeline:latest
```

- Start with docker run and env file and bound directory (Useful for active development)
```bash
docker run --gpus all -v ./pipeline_service:/workspace/pipeline_service -p 10006:10006 --env-file .env forge3d-pipeline:latest
```

## API Usage

**Seed parameter:**
- `seed: 42` - Use specific seed for reproducible results
- `seed: -1` - Auto-generate random seed (default)

### Endpoint 1: File upload (returns binary PLY)

```bash
curl -X POST "http://localhost:10006/generate" \
  -F "prompt_image_file=@cr7.png" \
  -F "seed=42" \
  -o model.ply
```

### Endpoint 2: File upload (returns binary SPZ)

```bash
curl -X POST "http://localhost:10006/generate-spz" \
  -F "prompt_image_file=@image.png" \
  -F "seed=42" \
  -o model.spz
```

### Endpoint 3: Base64 (returns JSON)

```bash
curl -X POST "http://localhost:10006/generate_from_base64" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_type": "image",
    "prompt_image": "<base64_encoded_image>",
    "seed": 42
  }'
```

## Quality tuning (identity / detail matching)

This pipeline generates multi-view inputs for 3D by optionally using **Qwen image editing** to synthesize additional views.
If you see **distorted shapes**, that usually means the synthesized views hallucinated geometry. If you see **simplified carvings/symbols**, increase the Trellis detail knobs.

- **Reduce hallucination (recommended default)**:
  - Keep the original view: `INCLUDE_ORIGINAL_VIEW=true`
  - Generate only a couple extra views: `QWEN_VIEW_KEYS=["side_view","back_view"]`
- **Disable view synthesis entirely (debugging / max identity)**:
  - `USE_QWEN_VIEWS=false` (uses only the bg-removed original image for 3D)
- **Increase 3D detail (slower)**:
  - `TRELLIS_SPARSE_STRUCTURE_STEPS`, `TRELLIS_SLAT_STEPS` (more steps = more detail, slower)
  - `TRELLIS_NUM_OVERSAMPLES` (helps stability/detail, increases compute)

### Per-request overrides (JSON)
You can override the multi-view strategy per request to A/B test quickly:

```json
{
  "prompt_type": "image",
  "prompt_image": "<base64>",
  "seed": 42,
  "use_qwen_views": true,
  "include_original_view": true,
  "qwen_view_keys": ["side_view", "back_view"],
  "trellis_params": {
    "sparse_structure_steps": 12,
    "sparse_structure_cfg_strength": 6.5,
    "slat_steps": 32,
    "slat_cfg_strength": 2.8,
    "num_oversamples": 4
  }
}
```

### Endpoint 4: Health check (returns JSON)

```bash
curl -X GET "http://localhost:10006/health" \
  -H "Content-Type: application/json" 
```