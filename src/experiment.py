"""
Experiment endpoint: sweep hyperparameters, generate all combinations on GPU
in parallel batches, stream results to /scratch/<id>/, and serve a zip with
an HTML gallery when complete.

Axis split
----------
Control-image axes  (inner batch — stacked tensors, one pipe() call per sub-batch)
  transform_dx, transform_dy, transform_z, transform_r, transform_strength

Scalar axes  (outer loop — must be uniform within a batch)
  depthmap_scale, edge_map_scale, ip_scale, strength, final_strength

Execution
---------
for scalar_combo in product(scalar axes):
    for sub_batch of ≤10 control combos (dx × dy × z × r × transform_strength):
        → stack B control tensors
        → one pipe() call → B images
        → if final_strength > 0: tiled_refine_images(B images)
        → save B PNGs
    every 10 total saves: flush manifest JSON + VRAM hygiene
zip + HTML gallery on completion.
"""

from __future__ import annotations

import gc
import io
import itertools
import json
import os
import shutil
import tempfile
import threading
import time
import traceback
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from random import randint
from typing import Optional

import torch
import torchvision.transforms.functional as TF
from compel import CompelForSDXL
from diffusers.pipelines.pag import StableDiffusionXLPAGImg2ImgPipeline
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps

from src.controlnet import get_asset_generator
from src.hr_fix import tiled_refine_images
from src.loras import add_loras
from src.models import LoRA
from src.pipeline import (
    MODEL_CACHE_DIR,
    apply_filmic_finish,
    configure_sgm_uniform_scheduler,
    generate_image,
    get_controlnet_model,
    get_pipe,
    load_ip_adapter_local,
    try_enable_pag,
)
from src.prompt import process_prompt
from src.transform import TransformParams, apply_transforms, lama_fill
from src.upscaler_gpu import upscale_images_gpu

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vram(label: str) -> None:
    """Print current CUDA VRAM usage at a labelled checkpoint."""
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    free = (
        torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()
    ) / 1024**3
    print(
        f"[VRAM] {label}: {alloc:.2f} GB alloc / {reserved:.2f} GB reserved / {free:.2f} GB free"
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCRATCH = Path("/scratch")
_SUB_BATCH = 10  # max control-image combos per pipe() call

_REFINE_POSITIVE = (
    "highly detailed, accurate textures, sharp focus, intricate surface detail, "
    "photorealistic++, crisp edges, fine grain, high frequency detail, realistic lighting++"
)
_REFINE_NEGATIVE = (
    "blurry, soft focus, low detail, low resolution, smeared, watercolor, "
    "painterly, plastic look, bad lighting"
)

# ---------------------------------------------------------------------------
# ParamRange
# ---------------------------------------------------------------------------


@dataclass
class ParamRange:
    start: float
    end: float
    step: float

    def values(self) -> list[float]:
        """Return evenly-spaced values from start to end (inclusive) by step.
        Uses a manual loop to avoid floating-point overshoot from arange."""
        result: list[float] = []
        v = self.start
        while v <= self.end + 1e-9:
            result.append(round(v, 10))
            v += self.step
        return result if result else [self.start]


# ---------------------------------------------------------------------------
# ExperimentRequest  (assembled from raw multipart form data)
# ---------------------------------------------------------------------------


@dataclass
class ExperimentRequest:
    # Fixed fields
    user_input: str
    model: str
    seed: int

    # LoRAs
    loras: list[LoRA]

    # Eagerly-read image bytes (UploadFile is gone after request returns)
    transform_input_image_bytes: Optional[bytes]
    mask_bytes: Optional[bytes]
    reference_bytes: Optional[bytes]
    ip_adapter_image_bytes: Optional[bytes]

    # Control-image sweep axes (inner batch)
    transform_dx: Optional[ParamRange]
    transform_dy: Optional[ParamRange]
    transform_z: Optional[ParamRange]
    transform_r: Optional[ParamRange]
    transform_strength: Optional[ParamRange]

    # Scalar sweep axes (outer loop)
    depthmap_scale: Optional[ParamRange]
    edge_map_scale: Optional[ParamRange]
    ip_scale: Optional[ParamRange]
    strength: Optional[ParamRange]
    final_strength: Optional[ParamRange]

    @classmethod
    async def as_form(cls, request: Request) -> "ExperimentRequest":
        form = await request.form()

        def _fv(key: str) -> Optional[str]:
            v = form.get(key)
            return str(v) if v is not None else None

        def _ff(key: str) -> Optional[float]:
            v = _fv(key)
            return float(v) if v is not None else None

        def _parse_range(name: str) -> Optional[ParamRange]:
            """
            Reads <name>_min, <name>_max, <name>_step from form.
            Returns None if _min is absent.
            If only _min is present: fixed single value (range of length 1).
            """
            mn = _ff(f"{name}_min")
            if mn is None:
                return None
            mx = _ff(f"{name}_max")
            st = _ff(f"{name}_step")
            if mx is None or st is None or st == 0:
                return ParamRange(mn, mn, 1.0)
            return ParamRange(mn, mx, st)

        async def _read_file(key: str) -> Optional[bytes]:
            f = form.get(key)
            if f is None:
                return None
            try:
                return await f.read()
            except Exception:
                return None

        # Loras: support JSON array or indexed form fields loras.0.name / loras.0.scale
        loras: list[LoRA] = []
        raw_loras = form.get("loras")
        if raw_loras:
            try:
                parsed = json.loads(str(raw_loras))
                loras = [LoRA(**l) if isinstance(l, dict) else l for l in parsed]
            except Exception:
                pass
        # Indexed style
        lora_map: dict[int, dict] = {}
        for key, value in form.multi_items():
            if key.startswith("loras."):
                parts = key.split(".")
                if len(parts) == 3 and parts[1].isdigit():
                    idx = int(parts[1])
                    lora_map.setdefault(idx, {})[parts[2]] = str(value)
        for idx in sorted(lora_map):
            loras.append(LoRA(**lora_map[idx]))

        return cls(
            user_input=str(form.get("user_input", "")),
            model=str(form.get("model", os.environ.get("DEFAULT_MODEL", "juggernaut"))),
            seed=int(form.get("seed", -1)),
            loras=loras,
            transform_input_image_bytes=await _read_file("transform_input_image"),
            mask_bytes=await _read_file("mask"),
            reference_bytes=await _read_file("reference"),
            ip_adapter_image_bytes=await _read_file("ip_adapter_image"),
            transform_dx=_parse_range("transform_dx"),
            transform_dy=_parse_range("transform_dy"),
            transform_z=_parse_range("transform_z"),
            transform_r=_parse_range("transform_r"),
            transform_strength=_parse_range("transform_strength"),
            depthmap_scale=_parse_range("depthmap_scale"),
            edge_map_scale=_parse_range("edge_map_scale"),
            ip_scale=_parse_range("ip_scale"),
            strength=_parse_range("strength"),
            final_strength=_parse_range("final_strength"),
        )


# ---------------------------------------------------------------------------
# ExperimentState
# ---------------------------------------------------------------------------


@dataclass
class ExperimentState:
    id: str
    status: str  # "pending" | "running" | "complete" | "error"
    total: int = 0
    completed: int = 0
    started_at: float = field(default_factory=time.monotonic)
    work_dir: Optional[Path] = None
    result_zip_path: Optional[Path] = None
    error: Optional[str] = None


# In-memory registry — keyed by experiment id
_experiments: dict[str, ExperimentState] = {}

# ---------------------------------------------------------------------------
# Transform cache
# ---------------------------------------------------------------------------


@dataclass
class _TransformResult:
    depth_pil: Image.Image
    canny_pil: Image.Image
    transform_fill_pil: Image.Image
    mask_pil: Image.Image  # L-mode, white = inpaint region


def _run_transform(
    input_bytes: bytes,
    mask_bytes: Optional[bytes],
    dx: int,
    dy: int,
    z: float,
    r: float,
) -> _TransformResult:
    """Run the full transform → lama → depth/canny pipeline for one (dx,dy,z,r)."""
    input_img = Image.open(io.BytesIO(input_bytes)).convert("RGB")
    t_params = TransformParams(dx=dx, dy=dy, z=z, r=r)
    transformed_rgb, void_mask = apply_transforms(input_img, t_params)

    has_voids = void_mask.getbbox() is not None
    filled_rgb = lama_fill(transformed_rgb, void_mask) if has_voids else transformed_rgb

    generator = get_asset_generator()
    depth_pil = generator.generate_depth(filled_rgb)
    canny_pil = generator.generate_canny(filled_rgb)

    if mask_bytes is not None:
        mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
        transformed_mask, _ = apply_transforms(mask_img, t_params)
        final_mask = transformed_mask.convert("L")
    else:
        final_mask = Image.eval(void_mask, lambda v: 0 if v > 0 else 255).convert("L")

    return _TransformResult(
        depth_pil=depth_pil.convert("RGB"),
        canny_pil=canny_pil.convert("RGB"),
        transform_fill_pil=filled_rgb.convert("RGB"),
        mask_pil=final_mask,
    )


# ---------------------------------------------------------------------------
# HTML gallery builder
# ---------------------------------------------------------------------------


def _build_gallery_html(
    manifest: list[dict],
    prompt: str,
    model: str,
    date_str: str,
    swept_params: list[str],
    fixed_params: dict,
) -> str:
    fixed_rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in fixed_params.items()
    )
    cards = []
    for entry in manifest:
        fname = entry["filename"]
        params = entry["params"]
        rows = "".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>"
            for k, v in params.items()
            if k in swept_params
        )
        cards.append(
            f'<div class="card">'
            f'<img src="images/{fname}" loading="lazy" alt="{fname}">'
            f'<table class="ptable">{rows}</table>'
            f"</div>"
        )
    cards_html = "\n".join(cards)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Experiment Gallery — {model}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:system-ui,sans-serif;background:#111;color:#eee;padding:16px}}
header{{margin-bottom:20px;padding:16px;background:#1e1e1e;border-radius:8px}}
header h1{{font-size:1.2rem;margin-bottom:8px}}
header .meta{{font-size:.85rem;color:#aaa;margin-bottom:8px}}
header table td:first-child{{color:#aaa;padding-right:12px;white-space:nowrap}}
header table td{{padding:2px 0;font-size:.82rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px}}
.card{{background:#1e1e1e;border-radius:8px;overflow:hidden}}
.card img{{width:100%;display:block;object-fit:cover}}
.ptable{{width:100%;border-collapse:collapse;font-size:.78rem;padding:8px}}
.ptable td{{padding:3px 8px;border-bottom:1px solid #333}}
.ptable td:first-child{{color:#aaa;white-space:nowrap}}
.ptable tr:last-child td{{border-bottom:none}}
</style>
</head>
<body>
<header>
  <h1>Experiment Gallery</h1>
  <div class="meta">Model: <b>{model}</b> &nbsp;|&nbsp; Generated: {date_str} &nbsp;|&nbsp; {len(manifest)} image(s)</div>
  <div class="meta" style="margin-bottom:4px">Prompt: <i>{prompt}</i></div>
  <table>{fixed_rows}</table>
</header>
<div class="grid">
{cards_html}
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------


def _run_experiment(
    state: ExperimentState, req: ExperimentRequest
) -> None:  # noqa: C901
    try:
        state.status = "running"
        state.started_at = time.monotonic()

        work_dir = state.work_dir
        images_dir = work_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # --- Load pipeline ---
        pipe = get_pipe(req.model)
        if req.loras:
            add_loras(pipe, req.loras)

        positive_prompt, negative_prompt = process_prompt(req.user_input, req.model)
        compel_proc = CompelForSDXL(pipe=pipe, device="cuda")
        conditioning = compel_proc(positive_prompt, negative_prompt=negative_prompt)
        del compel_proc

        # --- IP-Adapter ---
        ip_adapter_pil: Optional[Image.Image] = None
        if req.ip_adapter_image_bytes:
            ip_adapter_pil = Image.open(io.BytesIO(req.ip_adapter_image_bytes)).convert(
                "RGB"
            )
            # Load onto base pipe now — cn_pipe will share the same UNet
            # (via from_pipe) and inherits the IP-Adapter processors.
            load_ip_adapter_local(pipe)
            _vram("after IP-Adapter load")

        # --- Reference image (global img2img) ---
        reference_pil: Optional[Image.Image] = None
        if req.reference_bytes:
            reference_pil = Image.open(io.BytesIO(req.reference_bytes)).convert("RGB")

        # --- Refine pipe (built once, shared for all combos) ---
        refine_pipe = None
        refine_embeds = None
        needs_refine = req.final_strength is not None and any(
            v > 0 for v in req.final_strength.values()
        )
        if needs_refine:
            refine_pipe = StableDiffusionXLPAGImg2ImgPipeline.from_pipe(pipe)
            _vram("after refine_pipe from_pipe")
            refine_pipe.vae.enable_tiling()
            refine_pipe.vae.enable_slicing()
            try_enable_pag(refine_pipe, context="experiment refinement")
            try:
                refine_pipe.set_ip_adapter_scale(0.0)
            except Exception:
                pass
            configure_sgm_uniform_scheduler(refine_pipe)
            refine_compel = CompelForSDXL(pipe=refine_pipe, device="cuda")
            refine_cond = refine_compel(
                _REFINE_POSITIVE, negative_prompt=_REFINE_NEGATIVE
            )
            refine_embeds = {
                "prompt_embeds": refine_cond.embeds,
                "pooled_prompt_embeds": refine_cond.pooled_embeds,
                "negative_prompt_embeds": refine_cond.negative_embeds,
                "negative_pooled_prompt_embeds": refine_cond.negative_pooled_embeds,
            }
            del refine_compel

        # --- ControlNets (depth + canny) ---
        cn_depth = None
        cn_canny = None
        if req.transform_input_image_bytes:
            cn_depth = get_controlnet_model(
                "xinsir/controlnet-depth-sdxl-1.0", "depth_sdxl"
            ).to("cuda")
            cn_canny = get_controlnet_model(
                "xinsir/controlnet-canny-sdxl-1.0", "canny_sdxl"
            ).to("cuda")
            _vram("after ControlNet load (depth + canny)")

        # --- Build sweep axes ---
        def _axis(pr: Optional[ParamRange]) -> list[float]:
            return pr.values() if pr is not None else [None]

        # Control-image axes
        ctrl_axes = {
            "transform_dx": _axis(req.transform_dx),
            "transform_dy": _axis(req.transform_dy),
            "transform_z": _axis(req.transform_z),
            "transform_r": _axis(req.transform_r),
            "transform_strength": _axis(req.transform_strength),
        }
        # Scalar axes
        scalar_axes = {
            "depthmap_scale": _axis(req.depthmap_scale),
            "edge_map_scale": _axis(req.edge_map_scale),
            "ip_scale": _axis(req.ip_scale),
            "strength": _axis(req.strength),
            "final_strength": _axis(req.final_strength),
        }

        ctrl_keys = list(ctrl_axes.keys())
        ctrl_combos = list(itertools.product(*ctrl_axes.values()))
        scalar_keys = list(scalar_axes.keys())
        scalar_combos = list(itertools.product(*scalar_axes.values()))

        state.total = len(scalar_combos) * len(ctrl_combos)

        # Identify which params are actually swept (have >1 value)
        swept_params = [
            k for k, v in {**ctrl_axes, **scalar_axes}.items() if len(v) > 1
        ]

        # Transform result cache — keyed by (dx, dy, z, r)
        transform_cache: dict[tuple, _TransformResult] = {}

        manifest: list[dict] = []
        global_idx = 0

        # Target image size (from reference if available, else 1024×1024)
        if reference_pil is not None:
            target_w = reference_pil.width - (reference_pil.width % 8)
            target_h = reference_pil.height - (reference_pil.height % 8)
        else:
            target_w, target_h = 1024, 1024

        # --- Outer loop: scalar combos ---
        for s_combo in scalar_combos:
            s = dict(zip(scalar_keys, s_combo))
            depth_scale = s["depthmap_scale"] or 0.5
            edge_scale = s["edge_map_scale"] or 0.4
            ip_scale_val = s["ip_scale"]
            strength_val = s["strength"]
            final_strength_val = s["final_strength"]
            # Minimum of 0.05 ensures int(30 * strength) >= 1 so the scheduler
            # never produces empty timesteps (int(30 * 0.01) == 0 → crash).
            safe_final = (
                min(max(float(final_strength_val), 0.05), 0.40)
                if final_strength_val and final_strength_val > 0
                else None
            )

            # --- Inner loop: control-image combos, chunked into sub-batches ---
            for batch_start in range(0, len(ctrl_combos), _SUB_BATCH):
                sub_batch = ctrl_combos[batch_start : batch_start + _SUB_BATCH]
                B = len(sub_batch)

                # Per-slot tensors
                depth_tensors: list[torch.Tensor] = []
                canny_tensors: list[torch.Tensor] = []
                mask_tensors: list[torch.Tensor] = []
                ref_tensors: list[torch.Tensor] = []
                slot_params: list[dict] = []
                has_transform = req.transform_input_image_bytes is not None

                for slot_combo in sub_batch:
                    c = dict(zip(ctrl_keys, slot_combo))
                    dx = int(c["transform_dx"] or 0)
                    dy = int(c["transform_dy"] or 0)
                    z = float(c["transform_z"] or 1.0)
                    r = float(c["transform_r"] or 0.0)
                    ts = float(c["transform_strength"] or 1.0)

                    slot_params.append({**c, **s})

                    if has_transform:
                        cache_key = (dx, dy, z, r)
                        if cache_key not in transform_cache:
                            print(
                                f"🔄 Transform cache miss for (dx={dx}, dy={dy}, z={z}, r={r}) — running pipeline..."
                            )
                            transform_cache[cache_key] = _run_transform(
                                req.transform_input_image_bytes,
                                req.mask_bytes,
                                dx,
                                dy,
                                z,
                                r,
                            )
                        tr = transform_cache[cache_key]

                        # Resize to target
                        depth_img = ImageOps.fit(
                            tr.depth_pil, (target_w, target_h), Image.LANCZOS
                        )
                        canny_img = ImageOps.fit(
                            tr.canny_pil, (target_w, target_h), Image.LANCZOS
                        )
                        mask_img = ImageOps.fit(
                            tr.mask_pil, (target_w, target_h), Image.LANCZOS
                        )

                        # Blend transform fill into reference using transform_strength
                        if reference_pil is not None:
                            ref_base = ImageOps.fit(
                                reference_pil, (target_w, target_h), Image.LANCZOS
                            )
                            fill_img = ImageOps.fit(
                                tr.transform_fill_pil,
                                (target_w, target_h),
                                Image.LANCZOS,
                            )
                            blend_mask = mask_img.point(lambda p: int(p * ts))
                            ref_img = Image.composite(fill_img, ref_base, blend_mask)
                        else:
                            ref_img = ImageOps.fit(
                                tr.transform_fill_pil,
                                (target_w, target_h),
                                Image.LANCZOS,
                            )

                        depth_tensors.append(
                            TF.to_tensor(depth_img).to("cuda", dtype=torch.float16)
                        )
                        canny_tensors.append(
                            TF.to_tensor(canny_img).to("cuda", dtype=torch.float16)
                        )
                        mask_tensors.append(
                            TF.to_tensor(mask_img).to("cuda", dtype=torch.float16)
                        )
                        ref_tensors.append(
                            TF.to_tensor(ref_img).to("cuda", dtype=torch.float16)
                        )

                # --- Stack into batch tensors ---
                if has_transform and depth_tensors:
                    depth_batch = torch.stack(depth_tensors)  # (B,3,H,W)
                    canny_batch = torch.stack(canny_tensors)
                    mask_batch = torch.stack(
                        mask_tensors
                    )  # (B,1,H,W) — but to_tensor gives (1,H,W)
                    ref_batch = torch.stack(ref_tensors)

                    # Expand prompt embeds ×B
                    pe = conditioning.embeds.repeat(B, 1, 1)
                    ppe = conditioning.pooled_embeds.repeat(B, 1)
                    npe = conditioning.negative_embeds.repeat(B, 1, 1)
                    nppe = conditioning.negative_pooled_embeds.repeat(B, 1)

                    from diffusers import (
                        StableDiffusionXLControlNetInpaintPipeline,
                    )

                    # Share UNet, VAE, text-encoders with the cached base pipe.
                    # from_pipe avoids loading a second full SDXL copy into VRAM
                    # (~10 GB saved vs from_pretrained).
                    cn_pipe = StableDiffusionXLControlNetInpaintPipeline.from_pipe(
                        pipe,
                        controlnet=[cn_depth, cn_canny],
                    )
                    _vram("after cn_pipe from_pipe")
                    cn_pipe.vae.enable_tiling()
                    cn_pipe.vae.enable_slicing()
                    try_enable_pag(cn_pipe, context="experiment controlnet")

                    # IP-Adapter already loaded on the shared UNet — no reload needed.

                    gen_kwargs: dict = {
                        "pipe": cn_pipe,
                        "prompt_embeds": pe,
                        "pooled_prompt_embeds": ppe,
                        "negative_prompt_embeds": npe,
                        "negative_pooled_prompt_embeds": nppe,
                        "image": ref_batch,
                        "mask_image": mask_batch,
                        "control_image": [depth_batch, canny_batch],
                        "controlnet_conditioning_scale": [depth_scale, edge_scale],
                        "control_guidance_end": 0.5,
                        "num_inference_steps": 30,
                        "guidance_scale": 5.0,
                        "height": target_h,
                        "width": target_w,
                        "num_images_per_prompt": 1,
                        "seed": req.seed,
                    }
                    if strength_val is not None:
                        gen_kwargs["strength"] = float(strength_val)
                    if ip_adapter_pil:
                        gen_kwargs["ip_adapter_image"] = ip_adapter_pil
                        gen_kwargs["ip_adapter_scale"] = (
                            float(ip_scale_val) if ip_scale_val is not None else 0.5
                        )

                    images = generate_image(**gen_kwargs)
                    del cn_pipe
                    _vram("after cn_pipe del")

                else:
                    # No transform input — plain img2img or text2img, B=1 per combo
                    # (scalar combos only; run each independently)
                    images = []
                    for _sp in slot_params:
                        _gen_kwargs: dict = {
                            "pipe": pipe,
                            "prompt_embeds": conditioning.embeds,
                            "pooled_prompt_embeds": conditioning.pooled_embeds,
                            "negative_prompt_embeds": conditioning.negative_embeds,
                            "negative_pooled_prompt_embeds": conditioning.negative_pooled_embeds,
                            "num_inference_steps": 30,
                            "guidance_scale": 5.0,
                            "height": target_h,
                            "width": target_w,
                            "num_images_per_prompt": 1,
                            "seed": req.seed,
                        }
                        if reference_pil is not None:
                            _gen_kwargs["image"] = ImageOps.fit(
                                reference_pil, (target_w, target_h), Image.LANCZOS
                            )
                            _gen_kwargs["strength"] = (
                                float(strength_val) if strength_val else 0.75
                            )
                        if ip_adapter_pil:
                            _gen_kwargs["ip_adapter_image"] = ip_adapter_pil
                            _gen_kwargs["ip_adapter_scale"] = (
                                float(ip_scale_val) if ip_scale_val is not None else 0.5
                            )
                        images.extend(generate_image(**_gen_kwargs))

                # --- Optional batch refinement ---
                if safe_final and refine_pipe is not None:
                    print(
                        f"🎨 Refining batch of {len(images)} image(s) at strength={safe_final}..."
                    )
                    # Free fragmented VRAM from generation before upscaling on CPU.
                    gc.collect()
                    torch.cuda.empty_cache()
                    _vram("before upscale (post empty_cache)")
                    upscaled = upscale_images_gpu(images, outscale=4)
                    _vram("after upscale")
                    # Detect full upscale failure (all returned original size — OOM).
                    all_failed = all(
                        u.size == img.size for img, u in zip(images, upscaled)
                    )
                    if all_failed:
                        print(
                            "⚠️ All upscales failed (VRAM pressure) — skipping refinement, saving originals."
                        )
                        del upscaled
                    else:
                        refine_pag_scale = (
                            3.0
                            if getattr(refine_pipe, "pag_attn_processors", None)
                            else None
                        )
                        extra_refine_kwargs = {}
                        if ip_adapter_pil is not None:
                            # UNet has IP-Adapter loaded (shared via from_pipe).
                            # Even at scale=0.0 it requires image_embeds in
                            # added_cond_kwargs — provide the real image to satisfy it.
                            extra_refine_kwargs["ip_adapter_image"] = ip_adapter_pil
                        _vram("before tiled_refine_images")
                        images = tiled_refine_images(
                            upscaled,
                            refine_pipe,
                            strength=safe_final,
                            num_inference_steps=30,
                            guidance_scale=5.0,
                            pag_scale=refine_pag_scale,
                            **refine_embeds,
                            **extra_refine_kwargs,
                        )
                        del upscaled
                        _vram("after tiled_refine_images")

                # --- Save images ---
                for img, sp in zip(images, slot_params):
                    fname = f"{global_idx:04d}.png"
                    img.save(images_dir / fname, format="PNG")
                    manifest.append(
                        {
                            "filename": fname,
                            "params": {
                                k: (round(float(v), 4) if v is not None else None)
                                for k, v in sp.items()
                            },
                        }
                    )
                    del img
                    global_idx += 1
                    state.completed += 1

                    # Flush every 10 saves
                    if state.completed % 10 == 0:
                        (work_dir / "combos.json").write_text(
                            json.dumps(manifest, indent=2)
                        )
                        gc.collect()
                        torch.cuda.empty_cache()
                        _vram(f"periodic cleanup @ {state.completed} images")

                del images

        # --- Final manifest flush ---
        (work_dir / "combos.json").write_text(json.dumps(manifest, indent=2))

        # --- Build HTML gallery ---
        import datetime

        date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Fixed params = axes with exactly one value
        fixed_params: dict = {}
        for k, v in {**ctrl_axes, **scalar_axes}.items():
            if len(v) == 1 and v[0] is not None:
                fixed_params[k] = round(float(v[0]), 4)

        gallery_html = _build_gallery_html(
            manifest=manifest,
            prompt=req.user_input,
            model=req.model,
            date_str=date_str,
            swept_params=swept_params,
            fixed_params=fixed_params,
        )
        (work_dir / "gallery.html").write_text(gallery_html, encoding="utf-8")

        # --- Zip ---
        zip_path = work_dir / "experiment.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(work_dir / "gallery.html", "gallery.html")
            zf.write(work_dir / "combos.json", "combos.json")
            for png in sorted(images_dir.glob("*.png")):
                zf.write(png, f"images/{png.name}")

        state.result_zip_path = zip_path
        state.status = "complete"
        print(f"✅ Experiment {state.id} complete — {state.completed} image(s)")

    except Exception:
        state.error = traceback.format_exc()
        state.status = "error"
        print(f"❌ Experiment {state.id} failed:\n{state.error}")


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/experiment", tags=["experiment"])


@router.post("")
async def start_experiment(request: Request):
    req = await ExperimentRequest.as_form(request)

    if not req.user_input:
        raise HTTPException(status_code=400, detail="user_input is required")

    exp_id = str(uuid.uuid4())
    work_dir = _SCRATCH / exp_id
    work_dir.mkdir(parents=True, exist_ok=True)

    state = ExperimentState(id=exp_id, status="pending", work_dir=work_dir)
    _experiments[exp_id] = state

    t = threading.Thread(
        target=_run_experiment, args=(state, req), daemon=True, name=f"exp-{exp_id[:8]}"
    )
    t.start()

    return JSONResponse({"id": exp_id})


@router.get("/{exp_id}")
def get_experiment_status(exp_id: str):
    state = _experiments.get(exp_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    elapsed = time.monotonic() - state.started_at
    etd: Optional[float] = None
    if state.completed > 0 and state.total > 0 and state.status == "running":
        rate = state.completed / elapsed
        remaining = state.total - state.completed
        etd = remaining / rate if rate > 0 else None

    return JSONResponse(
        {
            "id": exp_id,
            "status": state.status,
            "total": state.total,
            "completed": state.completed,
            "elapsed_seconds": round(elapsed, 1),
            "etd_seconds": round(etd, 1) if etd is not None else None,
            "error": state.error,
        }
    )


@router.get("/{exp_id}/download")
def download_experiment(exp_id: str):
    state = _experiments.get(exp_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    if state.status == "complete":
        return FileResponse(
            path=str(state.result_zip_path),
            media_type="application/zip",
            filename=f"experiment_{exp_id[:8]}.zip",
        )

    if state.status == "error":
        raise HTTPException(status_code=500, detail=state.error or "Experiment failed")

    elapsed = time.monotonic() - state.started_at
    etd: Optional[float] = None
    if state.completed > 0 and state.total > 0:
        rate = state.completed / elapsed
        remaining = state.total - state.completed
        etd = remaining / rate if rate > 0 else None

    from fastapi.responses import JSONResponse as _JR

    return _JR(
        status_code=202,
        content={
            "id": exp_id,
            "status": state.status,
            "total": state.total,
            "completed": state.completed,
            "elapsed_seconds": round(elapsed, 1),
            "etd_seconds": round(etd, 1) if etd is not None else None,
        },
    )
