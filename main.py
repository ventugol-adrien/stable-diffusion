import base64
from contextlib import asynccontextmanager
import io, os, json, time
from random import randint
from pathlib import Path
import zipfile
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import FileResponse, Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from src.models import ImageRequest
from compel import CompelForSDXL
from diffusers import (
    AutoPipelineForImage2Image,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLControlNetInpaintPipeline,
    ControlNetModel,
)
from PIL import Image, ImageOps

from src.pipeline import (
    get_pipe,
    get_fast_pipe,
    warmup_pipeline,
    generate_image,
    shutdown,
    MODEL_CACHE_DIR,
)
from src.loras import add_loras, record_lora_config, router as loras_router
from src.prompt import process_prompt

from src.controlnet import router as depthmap_router, get_asset_generator
from src.transform import TransformParams, apply_transforms, lama_fill

import torch

MODELS_DIR = Path.home() / "sd_models"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: Load the default pipeline and run a warmup inference to populate
    MIOpen, Triton, and TunableOp runtime caches. This moves the ~250s cold
    start from the first user request to server boot.

    Shutdown: Flush TunableOp cache to disk and release VRAM.
    """
    # --- STARTUP ---

    skip_warmup = os.environ.get("SKIP_PIPELINE_WARMUP", "0") == "1"
    if skip_warmup:
        print("🚀 Lifespan startup: skipping pipeline warmup due to -ollama flag...")
    else:
        print("🚀 Lifespan startup: loading default pipeline + warmup...")
        # Enable TunableOp write-on-exit as a safety net (in addition to explicit flush)
        try:
            if torch.cuda.tunable.is_enabled():
                torch.cuda.tunable.write_file_on_exit(True)
        except Exception:
            pass
        pipe = get_pipe(os.environ.get("DEFAULT_MODEL", "juggernaut"))
        warmup_pipeline(pipe)
        del pipe  # Don't hold a reference — cleanup_resources() must be able to free it
        print("✅ Server ready. First user request will be fast.")

    yield  # --- Server runs here ---

    # --- SHUTDOWN ---
    print("🛑 Lifespan shutdown: flushing caches and releasing VRAM...")
    shutdown()


app = FastAPI(lifespan=lifespan)

origins = json.loads(os.environ.get("ORIGINS", "[]"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specific origins (use ["*"] to allow all)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

app.include_router(depthmap_router)
app.include_router(loras_router)


@app.post("/generate/image")
def handle_generate_image(
    request: ImageRequest = Depends(ImageRequest.as_form),
    returnAssets: bool = False,
):
    breakdown = {}
    start_time = time.monotonic()

    pipe = (
        get_fast_pipe(request.model) if request.lightning else get_pipe(request.model)
    )

    t_to_pipeline = time.monotonic() - start_time
    breakdown["pipeline_load_time"] = t_to_pipeline

    # Track which LoRA combos are used (currently none here)
    if request.loras:
        add_loras(pipe, request.loras)

    t_to_loras = time.monotonic() - t_to_pipeline
    breakdown["lora_load_time"] = t_to_loras

    positive_prompt, negative_prompt = process_prompt(request.user_input)

    compel_proc = CompelForSDXL(pipe=pipe, device="cuda")
    conditioning = compel_proc(positive_prompt, negative_prompt=negative_prompt)

    t_to_prompt = time.monotonic() - t_to_loras
    breakdown["prompt_processing_time"] = t_to_prompt

    init_image = None
    if request.reference:
        print("🖼️ Reference image provided, preparing for img2img generation...")
        init_image = Image.open(request.reference.file).convert("RGB")

        init_image = ImageOps.fit(init_image, (1024, 1024), method=Image.LANCZOS)
        pipe = AutoPipelineForImage2Image.from_pipe(pipe)

    # 1. Initialize dynamic lists for Multi-ControlNet
    controlnets = []
    control_images = []
    control_scales = []

    # 2. Process Depth Prior (Xinsir SOTA)
    if request.depthmap:
        print("🕳️ Depth map provided, loading Xinsir Depth ControlNet...")
        depthmap_img = Image.open(request.depthmap.file).convert("RGB")
        depthmap_img = ImageOps.fit(depthmap_img, (1024, 1024), method=Image.LANCZOS)
        control_images.append(depthmap_img)
        controlnets.append(
            ControlNetModel.from_pretrained(
                "xinsir/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16
            )
        )
        control_scales.append(
            request.depth_scales[0] if request.depth_scales else 0.5
        )  # Default weight for structural depth

    # 3. Process Canny Prior (Xinsir SOTA)
    if request.canny_edges:
        print("✏️ Canny map provided, loading Xinsir Canny ControlNet...")
        canny_img = Image.open(request.canny_edges.file).convert("RGB")
        canny_img = ImageOps.fit(canny_img, (1024, 1024), method=Image.LANCZOS)
        control_images.append(canny_img)
        controlnets.append(
            ControlNetModel.from_pretrained(
                "xinsir/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
            )
        )
        control_scales.append(
            request.edges_scales[0] if request.edges_scales else 0.4
        )  # Default weight for fine edge details

    # 4. Process Divergent Spaces (Heterogeneous Control Batching)
    has_mask_in_divergent = False
    reference_tensor = None
    mask_tensor = None
    active_mask_strength = 1.0

    if request.divergent_spaces:
        if len(request.divergent_spaces) != request.batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Number of divergent spaces ({len(request.divergent_spaces)}) must match batch size ({request.batch_size}).",
            )
        print(
            f"🌌 Divergent Spaces provided. Pre-computing sparse tensors for batch size {request.batch_size}..."
        )
        import torchvision.transforms.functional as TF

        batch_size = request.batch_size

        # --- Pre-process spatial transforms ---
        # When a space has transform params, we transform the input image,
        # LaMa-fill voids, extract depth + canny from the clean result,
        # transform the mask with black void fill, and inject the results
        # back into the space so the existing tensor-building code picks
        # them up seamlessly.
        for space in request.divergent_spaces:
            if space.transform_input_image is None:
                continue

            t_params = TransformParams(
                dx=space.transform_dx or 0,
                dy=space.transform_dy or 0,
                z=space.transform_z or 1.0,
                r=space.transform_r or 0.0,
            )
            print(f"🔄 Applying spatial transform: {t_params}")

            # Transform the input image and LaMa-fill voids
            input_img = Image.open(space.transform_input_image.file).convert("RGB")
            transformed_rgb, void_mask = apply_transforms(input_img, t_params)
            has_voids = void_mask.getbbox() is not None
            if has_voids:
                filled_rgb = lama_fill(transformed_rgb, void_mask)
            else:
                filled_rgb = transformed_rgb

            # Extract depth + canny from the clean filled image
            generator = get_asset_generator()
            space._generated_depth = generator.generate_depth(filled_rgb)
            space._generated_canny = generator.generate_canny(filled_rgb)

            # Store the LaMa-filled result for optional blending with reference
            space._transform_fill = filled_rgb

            # Discard uploaded depth/canny images — transform-generated ones take over.
            # Scales are preserved so the existing tensor-building code uses them.
            space.depthmap = None
            space.canny_edges = None

            # Transform the mask with black (0) void fill
            if space.mask:
                mask_img = Image.open(space.mask.file).convert("L")
                transformed_mask, _ = apply_transforms(mask_img, t_params)
                space._generated_mask = transformed_mask
            else:
                # No mask provided — create one where voids = black (don't inpaint)
                space._generated_mask = Image.eval(
                    void_mask, lambda v: 0 if v > 0 else 255
                ).convert("L")

        has_depth = any(
            ds.depthmap or hasattr(ds, "_generated_depth")
            for ds in request.divergent_spaces
        )
        has_canny = any(
            ds.canny_edges or hasattr(ds, "_generated_canny")
            for ds in request.divergent_spaces
        )
        has_mask_in_divergent = any(
            ds.mask or hasattr(ds, "_generated_mask") for ds in request.divergent_spaces
        )

        target_width, target_height = 1024, 1024
        if has_mask_in_divergent:
            for ds in request.divergent_spaces:
                # Get dimensions from generated reference (transform) or uploaded reference
                ref_src = (
                    Image.open(ds.reference.file).convert("RGB")
                    if ds.reference
                    else None
                )
                if ref_src is not None:
                    target_width = ref_src.width - (ref_src.width % 8)
                    target_height = ref_src.height - (ref_src.height % 8)
                    break

        if has_depth:
            depth_tensor = torch.zeros(
                (batch_size, 3, target_height, target_width),
                device="cuda",
                dtype=torch.float16,
            )
            active_depth_scale = 0.5
            for i in range(batch_size):
                space = request.divergent_spaces[i]
                if space.depthmap or hasattr(space, "_generated_depth"):
                    # Use generated depth (from transform) or open from UploadFile
                    if hasattr(space, "_generated_depth"):
                        img = space._generated_depth.convert("RGB")
                    else:
                        img = Image.open(space.depthmap.file).convert("RGB")
                    if has_mask_in_divergent:
                        img = ImageOps.fit(
                            img, (target_width, target_height), method=Image.LANCZOS
                        )
                    else:
                        img = ImageOps.fit(img, (1024, 1024), method=Image.LANCZOS)
                    img_tensor = TF.to_tensor(img).to(
                        device="cuda", dtype=torch.float16
                    )
                    depth_tensor[i] = img_tensor
                    if space.depthmap_scale is not None:
                        active_depth_scale = space.depthmap_scale
            controlnets.append(
                ControlNetModel.from_pretrained(
                    "xinsir/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16
                )
            )
            control_images.append(depth_tensor)
            control_scales.append(active_depth_scale)

        if has_canny:
            canny_tensor = torch.zeros(
                (batch_size, 3, target_height, target_width),
                device="cuda",
                dtype=torch.float16,
            )
            active_canny_scale = 0.4
            for i in range(batch_size):
                space = request.divergent_spaces[i]
                if space.canny_edges or hasattr(space, "_generated_canny"):
                    # Use generated canny (from transform) or open from UploadFile
                    if hasattr(space, "_generated_canny"):
                        img = space._generated_canny.convert("RGB")
                    else:
                        img = Image.open(space.canny_edges.file).convert("RGB")
                    if has_mask_in_divergent:
                        img = ImageOps.fit(
                            img, (target_width, target_height), method=Image.LANCZOS
                        )
                    else:
                        img = ImageOps.fit(img, (1024, 1024), method=Image.LANCZOS)
                    img_tensor = TF.to_tensor(img).to(
                        device="cuda", dtype=torch.float16
                    )
                    canny_tensor[i] = img_tensor
                    if space.edges_scale is not None:
                        active_canny_scale = space.edges_scale
            controlnets.append(
                ControlNetModel.from_pretrained(
                    "xinsir/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
                )
            )
            control_images.append(canny_tensor)
            control_scales.append(active_canny_scale)

        if has_mask_in_divergent:
            mask_tensor = torch.ones(
                (batch_size, 1, target_height, target_width),
                device="cuda",
                dtype=torch.float16,
            )
            reference_tensor = torch.zeros(
                (batch_size, 3, target_height, target_width),
                device="cuda",
                dtype=torch.float16,
            )
            for i in range(batch_size):
                space = request.divergent_spaces[i]
                has_generated = hasattr(space, "_generated_mask")
                has_uploaded = space.mask and space.reference
                if has_generated or has_uploaded:
                    # Mask: use generated (from transform) or open from UploadFile
                    if has_generated:
                        mask_img = space._generated_mask.convert("L")
                    else:
                        mask_img = Image.open(space.mask.file).convert("L")
                    mask_img = ImageOps.fit(
                        mask_img, (target_width, target_height), method=Image.LANCZOS
                    )
                    mask_tensor[i] = TF.to_tensor(mask_img).to(
                        device="cuda", dtype=torch.float16
                    )

                    # Reference: open from UploadFile
                    if space.reference:
                        ref_img = Image.open(space.reference.file).convert("RGB")
                    else:
                        continue
                    ref_img = ImageOps.fit(
                        ref_img, (target_width, target_height), method=Image.LANCZOS
                    )

                    # Pre-blend: mix the transform fill into the reference so
                    # the pipeline denoises from something that already resembles
                    # the input image.  At transform_strength=1.0, the reference
                    # becomes the transform image in masked areas.
                    if hasattr(space, "_transform_fill") and space.transform_strength:
                        fill_img = ImageOps.fit(
                            space._transform_fill.convert("RGB"),
                            (target_width, target_height),
                            method=Image.LANCZOS,
                        )
                        blend_mask = mask_img.point(
                            lambda p: int(p * space.transform_strength)
                        )
                        ref_img = Image.composite(fill_img, ref_img, blend_mask)

                    reference_tensor[i] = TF.to_tensor(ref_img).to(
                        device="cuda", dtype=torch.float16
                    )

                    if space.mask_strength is not None:
                        active_mask_strength = space.mask_strength

    # 5. Initialize Pipeline if any spatial priors exist
    if controlnets:
        print(f"🚀 Initializing SDXL Pipeline with {len(controlnets)} ControlNet(s)...")

        # The Diffusers pipeline natively accepts a list of ControlNet models
        if has_mask_in_divergent:
            pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
                MODEL_CACHE_DIR / request.model,
                controlnet=controlnets,
                torch_dtype=torch.float16,
            ).to("cuda")
        else:
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                MODEL_CACHE_DIR / request.model,
                controlnet=controlnets,
                torch_dtype=torch.float16,
            ).to("cuda")

        # Note: During the actual generation call (pipe(...)), you MUST pass:
        # image=control_images
        # controlnet_conditioning_scale=control_scales

        # 5. IP-Adapter Integration
        if request.ip_adapter_image and request.ip_adapter_scale:
            print("🧩 IP-Adapter image and scale provided, adding to pipeline...")
            ip_adapter_image = Image.open(request.ip_adapter_image.file).convert("RGB")
            pipe.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="sdxl_models",
                weight_name="ip-adapter_sdxl.bin",
            )
            pipe.set_ip_adapter_scale(request.ip_adapter_scale)

    # When ControlNets are active with batch_size > 1, the control image tensors
    # already carry the batch dimension. Expand prompt embeds to match and set
    # num_images_per_prompt=1 so diffusers' check_inputs won't reject the mismatch.
    prompt_embeds = conditioning.embeds
    pooled_prompt_embeds = conditioning.pooled_embeds
    negative_prompt_embeds = conditioning.negative_embeds
    negative_pooled_prompt_embeds = conditioning.negative_pooled_embeds
    num_images = request.batch_size
    if control_images and batch_size > 1:
        prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(batch_size, 1)
        negative_prompt_embeds = negative_prompt_embeds.repeat(batch_size, 1, 1)
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(
            batch_size, 1
        )
        num_images = 1

    gen_kwargs = {
        "pipe": pipe,
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
        "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
        "controlnet_conditioning_scale": control_scales if control_scales else None,
        "num_inference_steps": 8 if request.lightning else 30,
        "guidance_scale": 1.5 if request.lightning else 5.0,  # Fixed from 'cfg'
        "height": target_height if "target_height" in locals() else 1024,
        "width": target_width if "target_width" in locals() else 1024,
        "num_images_per_prompt": num_images,
        "ip_adapter_image": ip_adapter_image if request.ip_adapter_image else None,
        "control_guidance_end_step": 0.5,
        "seed": request.image_seed,
    }

    if has_mask_in_divergent:
        gen_kwargs["image"] = reference_tensor
        gen_kwargs["mask_image"] = mask_tensor
        gen_kwargs["control_image"] = control_images if control_images else None
        gen_kwargs["strength"] = active_mask_strength
    else:
        gen_kwargs["image"] = control_images if control_images else None

    images = generate_image(**gen_kwargs)

    # Optional img2img refinement pass
    if request.final_strength and request.final_strength > 0:
        print(f"🎨 Running img2img refinement (strength={request.final_strength})...")
        base_pipe = get_pipe(request.model)
        img2img_pipe = AutoPipelineForImage2Image.from_pipe(base_pipe)
        for i, img in enumerate(images):
            images[i] = img2img_pipe(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                image=img,
                strength=request.final_strength,
                num_inference_steps=8 if request.lightning else 30,
                guidance_scale=1.5 if request.lightning else 7.0,
                generator=torch.Generator(device="cuda").manual_seed(
                    randint(0, 2**32 - 1)
                ),
            ).images[0]
        del img2img_pipe

    t_to_generation = time.monotonic() - t_to_prompt
    breakdown["generation_time"] = t_to_generation
    # record_lora_config(request.model, request.loras)

    latency = time.monotonic() - start_time
    throughput = request.batch_size / latency if latency > 0 else 0

    metrics = {"latency": latency, "throughput": throughput, "breakdown": breakdown}

    if len(images) == 1 and not returnAssets:
        img_buffer = io.BytesIO()
        images[0].save(img_buffer, format="PNG")
        return Response(
            content=img_buffer.getvalue(),
            media_type="image/png",
            headers={
                "Content-Disposition": "inline; filename=result.png",
                "X-Metrics-Latency": str(latency),
                "X-Metrics-Throughput": str(throughput),
                "X-Metrics-Breakdown": json.dumps(breakdown),
            },
        )

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("metrics.json", json.dumps(metrics))
        for i, img in enumerate(images):
            img_buffer = io.BytesIO()
            img.save(img_buffer, format="PNG")
            zip_file.writestr(f"image_{i}.png", img_buffer.getvalue())

        # Include generated spatial assets when requested
        if returnAssets and request.divergent_spaces:
            for i, space in enumerate(request.divergent_spaces):
                if hasattr(space, "_generated_depth"):
                    buf = io.BytesIO()
                    space._generated_depth.save(buf, format="PNG")
                    zip_file.writestr(f"assets/{i}_depth_map.png", buf.getvalue())
                if hasattr(space, "_generated_canny"):
                    buf = io.BytesIO()
                    space._generated_canny.save(buf, format="PNG")
                    zip_file.writestr(f"assets/{i}_canny_edges.png", buf.getvalue())
                if hasattr(space, "_generated_mask"):
                    buf = io.BytesIO()
                    space._generated_mask.save(buf, format="PNG")
                    zip_file.writestr(f"assets/{i}_mask.png", buf.getvalue())

    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": "attachment; filename=results.zip",
        },
    )


@app.get("/models")
def get_models():
    """Return a list of available models and their active LoRAs."""
    model_safetensors = [f.stem for f in MODELS_DIR.glob("*.safetensors")]

    return JSONResponse(content=model_safetensors, media_type="application/json")
