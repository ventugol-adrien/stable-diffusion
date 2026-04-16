"""
Generate two images based on depth map, edge map, reference image, mask, and prompt.

Usage:
    python generate.py --prompt "your prompt" --depthmap depth.png --canny canny.png --mask mask.png --reference ref.png
"""

import argparse
import time
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from compel import CompelForSDXL
from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetInpaintPipeline,
)
from PIL import Image, ImageOps

from src.pipeline import (
    get_pipe,
    generate_image,
    cleanup_resources,
    MODEL_CACHE_DIR,
    DTYPE,
)
from src.prompt import process_prompt

MODEL = "juggernaut"


def load_and_prep_image(path: str, target_size=None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if target_size:
        return ImageOps.fit(img, target_size, method=Image.LANCZOS)
    return ImageOps.fit(img, (1024, 1024), method=Image.LANCZOS)


def load_and_prep_mask(path: str, target_size=None) -> Image.Image:
    img = Image.open(path).convert("L")
    if target_size:
        return ImageOps.fit(img, target_size, method=Image.LANCZOS)
    return ImageOps.fit(img, (1024, 1024), method=Image.LANCZOS)


def generate_two(
    prompt: str,
    depthmap_path: str,
    canny_path: str,
    mask_path: str,
    reference_path: str,
    output_dir: str = "outputs",
):
    cleanup_resources()

    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    print("Loading input images...")

    # Read reference image first to establish target dimensions
    ref_pil = Image.open(reference_path).convert("RGB")

    target_width = ref_pil.width - (ref_pil.width % 8)
    target_height = ref_pil.height - (ref_pil.height % 8)
    target_size = (target_width, target_height)

    depthmap_img = load_and_prep_image(depthmap_path, target_size=target_size)
    canny_img = load_and_prep_image(canny_path, target_size=target_size)
    mask_img = load_and_prep_mask(mask_path, target_size=target_size)
    ref_img = load_and_prep_image(reference_path, target_size=target_size)

    # Pre-compute base image tensors on GPU once
    depth_base = TF.to_tensor(depthmap_img).to(device="cuda", dtype=DTYPE)
    canny_base = TF.to_tensor(canny_img).to(device="cuda", dtype=DTYPE)
    mask_base = TF.to_tensor(mask_img).to(device="cuda", dtype=DTYPE)
    ref_base = TF.to_tensor(ref_img).to(device="cuda", dtype=DTYPE)

    # Load base pipe once for prompt encoding, then free VRAM
    print("Encoding prompt...")
    base_pipe = get_pipe(MODEL)
    positive_prompt, negative_prompt = process_prompt(prompt)
    compel_proc = CompelForSDXL(pipe=base_pipe, device="cuda")
    conditioning = compel_proc(positive_prompt, negative_prompt=negative_prompt)
    del compel_proc
    cleanup_resources()

    # Expand prompt embeddings for the batch dimension
    batch_size = 2
    prompt_embeds = conditioning.embeds.repeat(batch_size, 1, 1)
    pooled_prompt_embeds = conditioning.pooled_embeds.repeat(batch_size, 1)
    negative_prompt_embeds = conditioning.negative_embeds.repeat(batch_size, 1, 1)
    negative_pooled_prompt_embeds = conditioning.negative_pooled_embeds.repeat(
        batch_size, 1
    )

    # Load ControlNet models
    print("Loading ControlNet models...")
    depth_cn = ControlNetModel.from_pretrained(
        "xinsir/controlnet-depth-sdxl-1.0", torch_dtype=DTYPE
    )
    canny_cn = ControlNetModel.from_pretrained(
        "xinsir/controlnet-canny-sdxl-1.0", torch_dtype=DTYPE
    )

    # Build ControlNet pipeline
    print("Building Inpaint Pipeline...")
    cn_pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        MODEL_CACHE_DIR / MODEL,
        controlnet=[depth_cn, canny_cn],
        torch_dtype=DTYPE,
    ).to("cuda", memory_format=torch.channels_last)
    cn_pipe.enable_vae_slicing()
    cn_pipe.enable_vae_tiling()

    print(f"Batching {batch_size} images...")
    depth_tensor = depth_base.unsqueeze(0).expand(batch_size, -1, -1, -1)
    canny_tensor = canny_base.unsqueeze(0).expand(batch_size, -1, -1, -1)
    mask_tensor = mask_base.unsqueeze(0).expand(batch_size, -1, -1, -1)
    ref_tensor = ref_base.unsqueeze(0).expand(batch_size, -1, -1, -1)

    gen_kwargs = {
        "pipe": cn_pipe,
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
        "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
        "image": ref_tensor,
        "mask_image": mask_tensor,
        "control_image": [depth_tensor, canny_tensor],
        "controlnet_conditioning_scale": [0.5, 0.4],  # Default scales from main.py
        "strength": 0.,
        "num_inference_steps": 30,
        "guidance_scale": 7.0,
        "height": target_height,
        "width": target_width,
        "num_images_per_prompt": 1,
    }

    t0 = time.monotonic()
    images = generate_image(**gen_kwargs)
    latency = time.monotonic() - t0

    for i, img in enumerate(images):
        out_name = f"generated_{i}.png"
        out_path = out_dir / out_name
        img.save(out_path)
        print(f"Saved: {out_path}")

    print(f"\nDone! Latency: {latency:.1f}s ({latency / batch_size:.1f}s per image)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate two images from priors and mask"
    )
    parser.add_argument("--prompt", required=True, help="Prompt for generation")
    parser.add_argument("--depthmap", required=True, help="Path to depth map image")
    parser.add_argument("--canny", required=True, help="Path to canny edges image")
    parser.add_argument("--mask", required=True, help="Path to mask image")
    parser.add_argument("--reference", required=True, help="Path to reference image")
    parser.add_argument(
        "--output_dir", required=False, default="outputs", help="Output directory"
    )
    args = parser.parse_args()

    generate_two(
        prompt=args.prompt,
        depthmap_path=args.depthmap,
        canny_path=args.canny,
        mask_path=args.mask,
        reference_path=args.reference,
        output_dir=args.output_dir,
    )
