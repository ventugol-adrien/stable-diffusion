# Stable Diffusion XL Inference Server

A high-performance SDXL image generation API built with **FastAPI**, optimized for AMD RDNA 4 GPUs. The server manages pipeline lifecycle, model caching, LoRA hot-loading, and prompt conditioning via [Compel](https://github.com/damian0815/compel), returning batched results as a ZIP archive with per-request performance metrics.

## Features

- **SDXL text-to-image and image-to-image** generation at 1024×1024
- **Lightning mode** — reduced-step inference (8 steps, low CFG) for near-real-time output
- **LoRA support** — load one or more LoRA adapters per request with independent scale control
- **Automatic model caching** — first load converts single-file `.safetensors` to diffusers format for ~3× faster subsequent loads
- **Pipeline warmup on startup** — MIOpen, Triton, and TunableOp caches are populated at boot, eliminating cold-start latency
- **Spatial transforms + intelligent void filling** — translate, scale, rotate images/masks; voids are automatically filled with LaMa inpainting (fast structural continuation)
- **ControlNet guidance** — depth (Xinsir SDXL) and canny (Xinsir SDXL) priors, individually or combined
- **Divergent spaces** — per-image heterogeneous ControlNet inpainting within a single batch; each batch slot gets its own reference image, mask, depth map, and canny edges, with optional spatial transform pre-processing
- **IP-Adapter** — reference image style transfer via IP-Adapter-Plus-Face-SDXL
- **Perturbed-Attention Guidance (PAG)** — always-on quality enhancement on all pipeline types; falls back gracefully if unavailable
- **High-resolution fix** — true 4× Real-ESRGAN upscale followed by tiled SGM Uniform DPM++ 2M SDE img2img refinement with seam blending; triggered by `final_strength`
- **Offline-first artifact caching** — all HuggingFace repos and model weights are downloaded once and reused entirely from disk thereafter
- **Batched output** — multiple images per request, packaged as a ZIP with embedded `metrics.json`
- **CORS-configurable** — origin allowlist via environment variable

## Requirements

| Dependency                                                   | Purpose                         |
| ------------------------------------------------------------ | ------------------------------- |
| Python 3.10+                                                 | Runtime                         |
| PyTorch (ROCm)                                               | GPU compute                     |
| [Diffusers](https://github.com/huggingface/diffusers) ≥ 0.37 | SDXL + ControlNet pipelines     |
| [Compel](https://github.com/damian0815/compel)               | Prompt weighting / conditioning |
| [FastAPI](https://fastapi.tiangolo.com/)                     | HTTP server                     |
| [basicsr](https://github.com/XPixelGroup/BasicSR) 1.4.2      | Real-ESRGAN backbone            |
| [realesrgan](https://github.com/xinntao/Real-ESRGAN) 0.3.0   | High-resolution upscaling       |
| CUDA / ROCm compatible GPU                                   | Inference (RDNA 4 optimized)    |

## Project Structure

```
├── main.py                  # FastAPI application, endpoints, lifespan
├── requirements.txt
├── caches/
│   ├── warmed_configs.json  # Tracks which model/resolution combos are warmed
│   ├── models/              # Diffusers-format model cache (auto-generated)
│   │   └── juggernaut/
│   └── artifacts/           # Offline-first HuggingFace + weight cache
│       ├── huggingface/     # Cached HF repos (depth annotators, etc.)
│       ├── controlnet/      # Cached ControlNet model weights
│       ├── ip_adapter/      # Cached IP-Adapter weights
│       ├── vae_fp16_fix/    # Cached VAE fp16 fix weights
│       └── realesrgan/      # Cached Real-ESRGAN weights
└── src/
    ├── models.py            # Pydantic request/response schemas
    ├── pipeline.py          # Pipeline loading, caching, warmup, PAG, schedulers
    ├── loras.py             # LoRA loading and adapter management
    ├── prompt.py            # Prompt pre-processing and quality tags
    ├── controlnet.py        # ControlNet asset generation (depth + canny)
    ├── transform.py         # Spatial transforms and LaMa void-fill operations
    ├── upscaler.py          # Real-ESRGAN lazy singleton upscaler
    └── hr_fix.py            # Tiled 4x SDXL refinement and seam blending
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

The first request that uses ControlNet, IP-Adapter, depth annotation, or the HR fix will download and cache the required model artifacts under `caches/artifacts/`. All subsequent requests are fully offline.

## Environment Variables

| Variable               | Default      | Description                        |
| ---------------------- | ------------ | ---------------------------------- |
| `DEFAULT_MODEL`        | `juggernaut` | Model loaded on startup            |
| `ORIGINS`              | `[]`         | JSON array of allowed CORS origins |
| `SKIP_PIPELINE_WARMUP` | `0`          | Set to `1` to skip warmup on boot  |

## API

All endpoints that accept images use `multipart/form-data`.

---

### `POST /generate/image`

Generate one or more SDXL images. Returns a single `image/png` for `batch_size=1`, or a `application/zip` archive for larger batches.

**Request body** (`multipart/form-data`):

#### Core fields

| Field         | Type      | Default          | Description                                             |
| ------------- | --------- | ---------------- | ------------------------------------------------------- |
| `user_input`  | `string`  | _(required)_     | Text prompt                                             |
| `model`       | `string`  | `$DEFAULT_MODEL` | Model identifier                                        |
| `loras`       | `string`  | `[]`             | JSON array of LoRA objects `[{"name":"…","scale":0.5}]` |
| `lightning`   | `boolean` | `false`          | 8-step, CFG 1.5 mode for fast previews                  |
| `batch_size`  | `integer` | `1`              | Number of images to generate                            |
| `image_seed`  | `integer` | `-1`             | Generation RNG seed (`-1` = random)                     |
| `prompt_seed` | `integer` | `-1`             | Prompt processing RNG seed (`-1` = random)              |

#### Image-to-image (global)

| Field       | Type    | Default | Description                                                   |
| ----------- | ------- | ------- | ------------------------------------------------------------- |
| `reference` | `file`  | `null`  | Reference image for global img2img (LANCZOS-fit to 1024×1024) |
| `strength`  | `float` | `null`  | Denoising strength for the global img2img pass (0.0–1.0)      |

#### Global ControlNet priors

These apply the same prior to every image in the batch.

| Field                | Type    | Default | Description                                              |
| -------------------- | ------- | ------- | -------------------------------------------------------- |
| `depth_map`          | `file`  | `null`  | Depth map image (Xinsir SDXL depth ControlNet)           |
| `depth_scales`       | `float` | `[]`    | Weight for depth ControlNet (repeat for multiple values) |
| `canny_edges`        | `file`  | `null`  | Canny edge map image (Xinsir SDXL canny ControlNet)      |
| `canny_edges_scales` | `float` | `[]`    | Weight for canny ControlNet                              |

#### IP-Adapter (global)

| Field      | Type    | Default | Description                                   |
| ---------- | ------- | ------- | --------------------------------------------- |
| `ip_image` | `file`  | `null`  | Reference image for IP-Adapter style transfer |
| `ip_scale` | `float` | `null`  | IP-Adapter influence scale (0.0–1.0)          |

#### High-resolution fix

| Field            | Type    | Default | Description                                                                                                    |
| ---------------- | ------- | ------- | -------------------------------------------------------------------------------------------------------------- |
| `final_strength` | `float` | `null`  | Trigger a true 4× tiled HR fix after generation. Value is clamped to [0.15, 0.40]. Recommended: **0.20–0.30**. |

When `final_strength` is set, every generated image goes through:

1. **4× Real-ESRGAN upscale** (pixel space, RealESRGAN_x4plus, tile=512)
2. **Tile the 4× image** into overlapping 1024×1024 crops
3. **Run SGM Uniform DPM++ 2M SDE img2img** on each tile at the clamped strength (30 steps)
4. **Blend the refined tiles** back together with feathered seam masks

#### Divergent spaces (per-image heterogeneous ControlNet inpainting)

Divergent spaces let each slot in the batch have a different reference, mask, and guidance image. They are sent as indexed form fields: `0.<field>`, `1.<field>`, etc. The number of spaces must equal `batch_size`.

| Field (prefixed with `<index>.`) | Type      | Default | Description                                                                                  |
| -------------------------------- | --------- | ------- | -------------------------------------------------------------------------------------------- |
| `reference`                      | `file`    | `null`  | Base image for this batch slot (inpainting target)                                           |
| `mask`                           | `file`    | `null`  | Grayscale inpainting mask (white = inpaint, black = keep)                                    |
| `strength`                       | `float`   | `1.0`   | Inpainting denoising strength for this slot                                                  |
| `depth_map`                      | `file`    | `null`  | Per-slot depth prior                                                                         |
| `depth_map_scale`                | `float`   | `0.6`   | Weight for this slot's depth ControlNet                                                      |
| `canny_edges`                    | `file`    | `null`  | Per-slot canny prior                                                                         |
| `canny_edges_scale`              | `float`   | `0.2`   | Weight for this slot's canny ControlNet                                                      |
| `ip_image`                       | `file`    | `null`  | Per-slot IP-Adapter reference image                                                          |
| `ip_scale`                       | `float`   | `null`  | Per-slot IP-Adapter scale                                                                    |
| `transform.input_image`          | `file`    | `null`  | Source image for spatial transform (generates depth + canny)                                 |
| `transform.dx`                   | `integer` | `0`     | X-axis displacement in pixels                                                                |
| `transform.dy`                   | `integer` | `0`     | Y-axis displacement in pixels                                                                |
| `transform.z`                    | `float`   | `1.0`   | Zoom/scale factor                                                                            |
| `transform.r`                    | `float`   | `0.0`   | Rotation angle in degrees                                                                    |
| `transform_strength`             | `float`   | `null`  | How strongly the transform fill is pre-blended into the reference before denoising (0.0–1.0) |

When `transform.input_image` is provided:

1. The image is spatially transformed (dx, dy, z, r).
2. Voids are filled with LaMa inpainting.
3. Depth and canny priors are extracted from the filled image automatically.
4. The void region becomes the inpainting mask (white = transform void, black = keep original).
5. If `transform_strength` > 0, the LaMa fill is blended into the `reference` image inside the mask before denoising begins.

**Response:**

- `batch_size=1` → `image/png` with `X-Metrics-*` headers
- `batch_size>1` → `application/zip` containing `image_0.png … image_N.png` and `metrics.json`

Add `?returnAssets=true` to include generated depth maps, canny edges, and masks inside the ZIP under `assets/`.

---

### Examples

#### Simple text-to-image

```bash
curl -X POST http://localhost:8000/generate/image \
  -F "user_input=a photorealistic portrait of a woman in golden hour light"
```

#### With HR fix (recommended: `final_strength=0.25`)

```bash
curl -X POST http://localhost:8000/generate/image \
  -F "user_input=a photorealistic portrait of a woman in golden hour light" \
  -F "final_strength=0.25" \
  --output result.png
```

The HR fix pipeline:

1. Generates at 1024×1024 (30 steps, DPM++ 2M SDE)
2. Upscales to 4096×4096 with Real-ESRGAN (pixel space)
3. Splits the 4096×4096 image into overlapping 1024×1024 tiles
4. Runs a 30-step SGM Uniform img2img pass on each tile at strength=0.25
5. Feather-blends the refined tiles into the final 4× image

#### With spatial transform divergent space (batch_size=1)

```bash
curl -X POST http://localhost:8000/generate/image \
  -F "user_input=cinematic interior, cozy living room" \
  -F "batch_size=1" \
  -F "0.reference=@room.png" \
  -F "0.transform.input_image=@room.png" \
  -F "0.transform.dx=80" \
  -F "0.transform.z=0.95" \
  -F "0.transform_strength=0.7" \
  -F "0.strength=0.85" \
  -F "final_strength=0.25" \
  --output shifted.png
```

#### Two-slot divergent batch

```bash
curl -X POST http://localhost:8000/generate/image \
  -F "user_input=cinematic living room" \
  -F "batch_size=2" \
  -F "0.reference=@room_a.png" \
  -F "0.mask=@mask_a.png" \
  -F "0.depth_map=@depth_a.png" \
  -F "0.depth_map_scale=0.6" \
  -F "1.reference=@room_b.png" \
  -F "1.mask=@mask_b.png" \
  -F "1.depth_map=@depth_b.png" \
  -F "1.depth_map_scale=0.6" \
  --output results.zip
```

---

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

**Response** (`image/png`): The transformed image. Voids in RGB images are filled with LaMa inpainting. Non-RGB images (e.g. masks) return with black voids.

---

### `GET /models`

Returns an array of available model identifiers.

```json
["juggernaut", "pony"]
```

---

## Architecture Notes

- **Pipeline caching** — Only one pipeline is held in VRAM at a time. Switching models triggers a full cleanup (`gc.collect()` + `torch.cuda.empty_cache()`) to avoid OOM.
- **VAE tiling & slicing** — Enabled by default to keep peak VRAM usage low during decode.
- **Scheduler** — DPM++ 2M SDE (`sde-dpmsolver++`) for base generation. SGM Uniform trailing timestep spacing for the refinement pass.
- **PAG** — Perturbed-Attention Guidance is attempted on every pipeline type (base, ControlNet, img2img). Falls back silently if the diffusers build does not support it.
- **Real-ESRGAN** — Lazy-loaded on first HR fix request. Weights are downloaded once to `caches/artifacts/realesrgan/` and reused offline.
- **Prompt enhancement** — Quality tags are automatically prepended; model-specific negative prompts are injected (e.g., Pony-style score tags).
- **Offline-first caching** — `snapshot_download` fetches each HF repo once; thereafter the local directory is used directly without any network calls.

## License

This project is provided as-is for personal and research use.
