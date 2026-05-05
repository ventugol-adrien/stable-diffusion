# Stable Diffusion XL Inference Server

A high-performance SDXL image generation API built with **FastAPI**, optimized for AMD RDNA 4 GPUs. The server manages pipeline lifecycle, model caching, LoRA hot-loading, multi-ControlNet spatial conditioning, and prompt conditioning via [Compel](https://github.com/damian0815/compel).

## Features

- **SDXL text-to-image and image-to-image** generation
- **Lightning mode** — reduced-step inference (8 steps, low CFG) for near-real-time output
- **LoRA support** — load one or more LoRA adapters per request with independent scale control
- **Multi-ControlNet** — depth map and canny edge conditioning via Xinsir SDXL ControlNets
- **IP-Adapter** — style/reference image conditioning
- **Divergent Spaces** — heterogeneous per-image control (different depth/canny/mask per batch item)
- **Inpainting** — mask-guided inpainting via `StableDiffusionXLControlNetInpaintPipeline`
- **ControlNet asset generation** — on-server depth map (DA3), edge map (PiDiNet), and compositing mask extraction
- **Node-based workflow engine** — composable pipelines for hi-res refinement, tiling, upscaling, and spatial transforms
- **Automatic model caching** — first load converts single-file `.safetensors` to diffusers format for ~3× faster subsequent loads
- **Pipeline warmup on startup** — MIOpen, Triton, and TunableOp caches are populated at boot, eliminating cold-start latency
- **Batched output** — multiple images per request; single PNG returned when `batch_size=1`
- **CORS-configurable** — origin allowlist via environment variable

## Requirements

| Dependency                                                      | Purpose                            |
| --------------------------------------------------------------- | ---------------------------------- |
| Python 3.10+                                                    | Runtime                            |
| PyTorch (ROCm)                                                  | GPU compute                        |
| [Diffusers](https://github.com/huggingface/diffusers) ≥ 0.36    | SDXL pipeline                      |
| [Compel](https://github.com/damian0815/compel)                  | Prompt weighting / conditioning    |
| [FastAPI](https://fastapi.tiangolo.com/)                        | HTTP server                        |
| [controlnet-aux](https://github.com/huggingface/controlnet_aux) | PiDiNet edge extraction            |
| [transformers](https://github.com/huggingface/transformers)     | Depth-Anything V2 depth estimation |
| CUDA / ROCm compatible GPU                                      | Inference (RDNA 4 optimized)       |

## Project Structure

```
├── main.py                  # FastAPI application, endpoints, lifespan
├── openapi.yaml             # OpenAPI 3.1 specification
├── requirements.txt
├── caches/
│   ├── warmed_configs.json  # Tracks which model/resolution combos are warmed
│   └── models/              # Diffusers-format model cache (auto-generated)
│       └── juggernaut/
└── src/
    ├── models.py            # Pydantic request/response schemas
    ├── pipeline.py          # Pipeline loading, caching, warmup, generation
    ├── loras.py             # LoRA loading and adapter management (/loras router)
    ├── prompt.py            # Prompt pre-processing and quality tags
    ├── controlnet.py        # ControlNet asset generation (/spatial-assets router)
    ├── executor.py          # DAG execution engine
    └── nodes/
        ├── base_node.py     # Abstract node base class
        ├── compel_node.py   # Prompt conditioning via Compel
        ├── text2image.py    # Text-to-image inference node
        ├── image2image.py   # Image-to-image inference node
        ├── hi_res_node.py   # Hi-res refinement pass node
        ├── tiling_node.py   # Tiling plan node
        ├── upscale_node.py  # Upscaling node
        ├── transform_node.py# Spatial transform node
        └── response_node.py # HTTP response packaging node
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

| Variable               | Default      | Description                                     |
| ---------------------- | ------------ | ----------------------------------------------- |
| `DEFAULT_MODEL`        | `juggernaut` | Model loaded on startup and used as the default |
| `ORIGINS`              | `[]`         | JSON array of allowed CORS origins              |
| `SKIP_PIPELINE_WARMUP` | `0`          | Set to `1` to skip warmup on boot               |

## API

All endpoints that accept image files use **`multipart/form-data`**.

---

### `POST /generate/image`

Generate one or more images from a text prompt. Supports text-to-image, image-to-image, multi-ControlNet, IP-Adapter, and inpainting via Divergent Spaces.

**Request** (`multipart/form-data`):

| Field                       | Type      | Default          | Description                                                   |
| --------------------------- | --------- | ---------------- | ------------------------------------------------------------- |
| `user_input`                | `string`  | _(required)_     | Text prompt                                                   |
| `model`                     | `string`  | `$DEFAULT_MODEL` | Model identifier                                              |
| `lightning`                 | `boolean` | `false`          | 8-step low-CFG mode                                           |
| `batch_size`                | `integer` | `1`              | Number of images to generate                                  |
| `resolution`                | `string`  | `480p`           | Output resolution: `360p`, `480p`, `720p`, `1080p`            |
| `image_seed`                | `integer` | `-1`             | Generation seed (`-1` = random)                               |
| `prompt_seed`               | `integer` | `-1`             | Prompt processing seed                                        |
| `loras.{n}.name`            | `string`  | —                | LoRA adapter name (from `~/sd_loras/`)                        |
| `loras.{n}.scale`           | `float`   | `0.5`            | LoRA influence weight                                         |
| `reference`                 | `file`    | —                | Reference image for img2img                                   |
| `strength`                  | `float`   | —                | Denoising strength for img2img (0.0–1.0)                      |
| `depth_map`                 | `file`    | —                | Depth map image for ControlNet depth conditioning             |
| `depth_scales`              | `float[]` | —                | ControlNet scale(s) for the depth map                         |
| `canny_edges`               | `file`    | —                | Canny edge map image for ControlNet edge conditioning         |
| `canny_edges_scales`        | `float[]` | —                | ControlNet scale(s) for the edge map                          |
| `ip_image`                  | `file`    | —                | Reference image for IP-Adapter style transfer                 |
| `ip_scale`                  | `float`   | —                | IP-Adapter influence weight                                   |
| `final_strength`            | `float`   | —                | Post-generation refinement strength (0.0–1.0)                 |
| `grain_intensity`           | `float`   | `0.020`          | Film grain intensity applied to output (0.0–0.10)             |
| `{n}.depth_map`             | `file`    | —                | Per-image depth map for Divergent Spaces batch item `n`       |
| `{n}.depth_map_scale`       | `float`   | `0.6`            | Depth scale for batch item `n`                                |
| `{n}.canny_edges`           | `file`    | —                | Per-image canny map for batch item `n`                        |
| `{n}.canny_edges_scale`     | `float`   | `0.2`            | Canny scale for batch item `n`                                |
| `{n}.mask`                  | `file`    | —                | Inpaint mask for batch item `n` (enables inpainting pipeline) |
| `{n}.reference`             | `file`    | —                | Base image for inpainting for batch item `n`                  |
| `{n}.strength`              | `float`   | `1.0`            | Inpaint denoising strength for batch item `n`                 |
| `{n}.ip_image`              | `file`    | —                | IP-Adapter image for batch item `n`                           |
| `{n}.ip_scale`              | `float`   | —                | IP-Adapter scale for batch item `n`                           |
| `{n}.transform_input_image` | `file`    | —                | Input image for spatial transform on batch item `n`           |
| `{n}.transform_dx`          | `int`     | —                | Horizontal pixel offset                                       |
| `{n}.transform_dy`          | `int`     | —                | Vertical pixel offset                                         |
| `{n}.transform_z`           | `float`   | —                | Zoom factor                                                   |
| `{n}.transform_r`           | `float`   | —                | Rotation in degrees                                           |
| `{n}.transform_strength`    | `float`   | —                | Transform blend strength                                      |

> **Divergent Spaces** (`{n}.*` fields): when any indexed field is present, each batch item receives its own set of spatial priors. The number of batch items must equal `batch_size`. Including a `{n}.mask` field switches the pipeline to `StableDiffusionXLControlNetInpaintPipeline`.

**Response:**

- **`image/png`** — when `batch_size=1`
- **`application/zip`** — when `batch_size>1`, containing `image_0.png … image_N.png` and `metrics.json`

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

---

### `GET /models`

List model identifiers available for generation (`.safetensors` stems from `~/sd_models/`).

**Response** (`application/json`):

```json
["juggernaut", "pony"]
```

---

### `GET /loras/`

List LoRA adapter names available for use (`.safetensors` stems from `~/sd_loras/`).

**Response** (`application/json`):

```json
["detail_enhancer", "style_anime"]
```

---

### `POST /spatial-assets/generate`

Generate ControlNet conditioning assets from a single input image: a DA3 depth map, a PiDiNet edge map, and five foreground/background mask pairs fused from both.

**Request** (`multipart/form-data`):

| Field         | Type   | Description               |
| ------------- | ------ | ------------------------- |
| `input_image` | `file` | _(required)_ Source image |

**Response** (`application/zip`) — `controlnet_priors.zip` containing:

| File                             | Description                  |
| -------------------------------- | ---------------------------- |
| `da3_depth_map.png`              | Depth-Anything V3 depth map  |
| `matched_edge_map.png`           | PiDiNet crisp edge map       |
| `masks/{variant}_foreground.png` | Foreground mask (5 variants) |
| `masks/{variant}_background.png` | Background mask (5 variants) |

Mask variants: `conservative_baseline`, `conservative_depth_push`, `conservative_tight_edge`, `conservative_deep_smooth`, `conservative_surgical`.

---

### `POST /workflows/`

Full hi-res workflow: text-to-image (or image-to-image) → 4× upscale → tiling → hi-res refinement pass → spatial transform → response.

**Request** (`multipart/form-data`):

| Field             | Type      | Default      | Description                             |
| ----------------- | --------- | ------------ | --------------------------------------- |
| `prompt`          | `string`  | _(required)_ | Text prompt                             |
| `negative_prompt` | `string`  | —            | Negative prompt                         |
| `model`           | `string`  | `juggernaut` | Model identifier                        |
| `lightning`       | `boolean` | `false`      | Lightning mode                          |
| `width`           | `integer` | `1024`       | Image width                             |
| `height`          | `integer` | `1024`       | Image height                            |
| `steps`           | `integer` | `50`         | Inference steps                         |
| `cfg_scale`       | `float`   | `7.5`        | Classifier-free guidance scale          |
| `batch_size`      | `integer` | `1`          | Number of images                        |
| `output_type`     | `string`  | `pil`        | `pil` or `pt`                           |
| `init_image`      | `file`    | —            | Init image (triggers image-to-image)    |
| `strength`        | `float`   | `0.75`       | Denoising strength for img2img          |
| `hires_strength`  | `float`   | `0.35`       | Strength for the hi-res refinement pass |

**Response:** `image/png` (single) or `application/zip` (batch).

---

### `POST /workflows/image/`

Simple workflow: text-to-image or image-to-image only, no upscaling or hi-res pass. Accepts the same form fields as `/workflows/` (except `hires_strength` is ignored).

**Response:** `image/png` (single) or `application/zip` (batch).

---

## Architecture Notes

- **Pipeline caching** — Only one pipeline is held in VRAM at a time. Switching models triggers a full cleanup (`gc.collect()` + `torch.cuda.empty_cache()`) to avoid OOM.
- **VAE** — Uses `madebyollin/sdxl-vae-fp16-fix` with tiling and slicing enabled to keep peak VRAM low during decode.
- **Schedulers** — Standard mode: Euler Ancestral (30 steps, CFG 7.0). Lightning mode: DPM Solver Multistep (8 steps, CFG 1.5).
- **Prompt enhancement** — Quality tags are automatically prepended; model-specific negative prompts are injected (e.g., Pony-style score tags).
- **cuDNN disabled** — Disabled to avoid `CUDNN_STATUS_NOT_INITIALIZED` conflicts with the system CUDA driver; PyTorch built-in CUDA kernels are used instead.
- **TunableOp** — Written to disk on exit as a cache warm-up safety net.
- **ControlNet models** — Loaded on demand per request; not cached between requests.

## License

This project is provided as-is for personal and research use.
