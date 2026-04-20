# Stable Diffusion XL Inference Server

A high-performance SDXL image generation API built with **FastAPI**, optimized for AMD RDNA 4 GPUs. The server manages pipeline lifecycle, model caching, LoRA hot-loading, and prompt conditioning via [Compel](https://github.com/damian0815/compel), returning batched results as a ZIP archive with per-request performance metrics.

## Features

- **SDXL text-to-image and image-to-image** generation at 1024×1024
- **Lightning mode** — reduced-step inference (8 steps, low CFG) for near-real-time output
- **LoRA support** — load one or more LoRA adapters per request with independent scale control
- **Automatic model caching** — first load converts single-file `.safetensors` to diffusers format for ~3× faster subsequent loads
- **Pipeline warmup on startup** — MIOpen, Triton, and TunableOp caches are populated at boot, eliminating cold-start latency
- **Spatial transforms + intelligent void filling** — translate, scale, rotate images/masks; voids are automatically filled with LaMa inpainting (fast structural continuation)
- **Batched output** — multiple images per request, packaged as a ZIP with embedded `metrics.json`
- **CORS-configurable** — origin allowlist via environment variable

## Requirements

| Dependency                                                   | Purpose                         |
| ------------------------------------------------------------ | ------------------------------- |
| Python 3.10+                                                 | Runtime                         |
| PyTorch (ROCm)                                               | GPU compute                     |
| [Diffusers](https://github.com/huggingface/diffusers) ≥ 0.36 | SDXL pipeline                   |
| [Compel](https://github.com/damian0815/compel)               | Prompt weighting / conditioning |
| [FastAPI](https://fastapi.tiangolo.com/)                     | HTTP server                     |
| CUDA / ROCm compatible GPU                                   | Inference (RDNA 4 optimized)    |

## Project Structure

```
├── main.py                  # FastAPI application, endpoints, lifespan
├── requirements.txt
├── caches/
│   ├── warmed_configs.json  # Tracks which model/resolution combos are warmed
│   └── models/              # Diffusers-format model cache (auto-generated)
│       └── juggernaut/
└── src/
    ├── models.py            # Pydantic request/response schemas
    ├── pipeline.py          # Pipeline loading, caching, warmup, generation
    ├── loras.py             # LoRA loading and adapter management
    ├── prompt.py            # Prompt pre-processing and quality tags
    ├── controlnet.py        # ControlNet asset generation and spatial transform endpoint
    └── transform.py         # Spatial transforms and LaMa void-fill operations
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place models

Put SDXL `.safetensors` checkpoint files in `~/sd_models/`. The filename (without extension) becomes the model identifier:

```
~/sd_models/juggernaut.safetensors
~/sd_models/pony.safetensors
```

On first load, each model is converted to diffusers format and cached under `caches/models/<name>/` for faster subsequent starts.

### 3. Place LoRAs (optional)

Put LoRA `.safetensors` files in `~/sd_loras/`:

```
~/sd_loras/detail_enhancer.safetensors
```

### 4. Run the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Environment Variables

| Variable               | Default      | Description                        |
| ---------------------- | ------------ | ---------------------------------- |
| `DEFAULT_MODEL`        | `juggernaut` | Model loaded on startup            |
| `ORIGINS`              | `[]`         | JSON array of allowed CORS origins |
| `SKIP_PIPELINE_WARMUP` | `0`          | Set to `1` to skip warmup on boot  |

## API

### `POST /generate/image/`

Generate one or more images from a text prompt.

**Request body** (`application/json`):

| Field                | Type      | Default          | Description                                     |
| -------------------- | --------- | ---------------- | ----------------------------------------------- |
| `user_input`         | `string`  | _(required)_     | Text prompt                                     |
| `model`              | `string`  | `$DEFAULT_MODEL` | Model identifier                                |
| `loras`              | `array`   | `[]`             | LoRA adapters `[{"name": "...", "scale": 0.5}]` |
| `lightning`          | `boolean` | `false`          | Use lightning mode (8 steps, CFG 1.5)           |
| `batch_size`         | `integer` | `1`              | Number of images to generate                    |
| `reference`          | `bytes`   | `null`           | Reference image for img2img                     |
| `reference_strength` | `float`   | `null`           | Strength for img2img denoising                  |
| `image_seed`         | `integer` | `-1`             | Seed for generation (`-1` = random)             |
| `prompt_seed`        | `integer` | `-1`             | Seed for prompt processing                      |

**Response** (`application/zip`):

The ZIP archive contains:

- `metrics.json` — latency, throughput, and timing breakdown
- `image_0.png`, `image_1.png`, … — generated images

```json
// metrics.json
{
  "latency": 4.82,
  "throughput": 0.207,
  "breakdown": {
    "pipeline_load_time": 0.001,
    "lora_load_time": 0.0,
    "prompt_processing_time": 0.12,
    "generation_time": 4.7
  }
}
```

### `POST /spatial-assets/transform`

Spatially transform an image (scale, rotate, displace) with automatic void filling.

**Request body** (`multipart/form-data`):

| Field         | Type      | Default      | Description                          |
| ------------- | --------- | ------------ | ------------------------------------ |
| `input_image` | `file`    | _(required)_ | Image file to transform              |
| `dx`          | `integer` | `0`          | X-axis displacement in pixels        |
| `dy`          | `integer` | `0`          | Y-axis displacement in pixels        |
| `z`           | `float`   | `1.0`        | Zoom/scale factor (0.1–5.0)          |
| `r`           | `float`   | `0.0`        | Rotation angle in degrees (-360–360) |

**Response** (`image/png`):

The transformed image as a PNG. For RGB images, void regions are automatically
filled using LaMa inpainting (fast, deterministic structural continuation).
Non-RGB images (e.g. grayscale masks) are returned with black voids.

### `GET /models/`

List available model identifiers.

**Response** (`application/json`):

```json
["juggernaut", "pony"]
```

## Architecture Notes

- **Pipeline caching** — Only one pipeline is held in VRAM at a time. Switching models triggers a full cleanup (`gc.collect()` + `torch.cuda.empty_cache()`) to avoid OOM on consumer GPUs.
- **VAE tiling & slicing** — Enabled by default to keep peak VRAM usage low during decode.
- **Euler Ancestral scheduler** — Used for standard mode (25 steps, CFG 7.0). Lightning mode uses 8 steps at CFG 1.5.
- **Prompt enhancement** — Quality tags are automatically prepended; model-specific negative prompts are injected (e.g., Pony-style score tags).

## License

This project is provided as-is for personal and research use.
