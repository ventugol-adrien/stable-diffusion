# **Outpainting Noise/Blur: Root Cause Analysis and Resolution**

## **1. Summary**

The research document *Architectural and Pipeline Analysis of Degradation in Outpainting Workflows Under Memory Constraints* correctly identified the theoretical mechanisms behind outpainting noise and blur. However, it did not trace any of those mechanisms to specific lines of code. This document records the exact bugs found in the implementation, how they map to the research, what was changed, and one additional finding (R1) that corrected a subtle error in the fix itself.

The fix reduced outpainting artifacts from severe/persistent to resolved across the three primary failure modes: plasticky over-smooth surfaces, flat blurry voids, and variance-crushed textures.

---

## **2. Confirmed Root Causes**

Three independent bugs acted in concert. Each contributed a distinct artifact type and was present simultaneously, making the degradation appear as a single undifferentiated "blur and noise" problem.

### **Bug 1 — `guidance_rescale=0.7` applied unconditionally to epsilon-prediction models**

**File:** `src/nodes/outpainting_node.py`

**What was wrong:**
```python
# BEFORE — applied to every model unconditionally
if is_cn_pipe:
    pipe_kwargs["guidance_rescale"] = 0.7
```

`guidance_rescale` is a variance-normalization operation designed exclusively for models trained with a **v-prediction objective** and a **zero terminal SNR** noise schedule (Lin et al., 2023). It rescales the CFG-amplified noise residual back to the original standard deviation to prevent exposure blowout.

When applied to a standard **epsilon-prediction** model (juggernaut, master, master_inpaint), the operation has the inverse effect: it forcefully suppresses the variance of the noise residual toward a mean that the epsilon model was never trained to expect. In outpainting, where the UNet must invent high-variance textures from a structureless void, this variance suppression manifests as:

- Plasticky, over-smooth surfaces with no grain or pore detail
- Accurate large-scale structure (low-frequency semantics are unaffected) next to flatly rendered fill zones
- Output that looks "correct at arm's length but blurry up close"

This matched the research document's §8 analysis precisely and was the highest-impact single bug.

**What was changed:**
```python
# AFTER — only applied when the scheduler declares v_prediction
is_vpred = (
    pipe.scheduler.config.get("prediction_type", "epsilon") == "v_prediction"
)
if is_vpred:
    pipe_kwargs["guidance_rescale"] = 0.7
```

**R1 finding — why the gate reads `scheduler.config`, not `unet.config`:**
The initial fix read `pipe.unet.config.get("prediction_type", ...)`. Inspection of every cached model's `unet/config.json` showed that `prediction_type` is **absent** from the UNet config for all SDXL checkpoints — diffusers does not write it there. The authoritative source is the **scheduler config**, where `pipeline.get_pipe()` already injects `"prediction_type": "v_prediction"` for `vpred/noob/illustrious` model names. The gate was updated to read from `pipe.scheduler.config` instead.

Result per model:

| Model | `prediction_type` in unet/config.json | Source of truth | `guidance_rescale` applied |
|---|---|---|---|
| juggernaut | absent (KEY_ABSENT) | scheduler config: `epsilon` (default) | No ✓ |
| master | absent (KEY_ABSENT) | scheduler config: `epsilon` (default) | No ✓ |
| master_inpaint | absent (KEY_ABSENT) | scheduler config: `epsilon` (default) | No ✓ |
| illustrious | absent (KEY_ABSENT) | scheduler config: `v_prediction` (injected by `get_pipe`) | Yes ✓ |

---

### **Bug 2 — `strength` defaulted to `0.85` instead of `1.0` in the outpaint workflow**

**File:** `main.py`, line 689

**What was wrong:**
```python
# BEFORE
strength=request.strength or 0.85,
```

`strength` in a diffusers inpainting pipeline controls how much noise is injected into the latent before the denoising schedule begins. At `strength=0.85`, only 85% noise is added; the remaining 15% of the original encoded content persists in the latent.

For **internal inpainting** (replacing a region within an existing image), a sub-1.0 strength is intentional — it preserves the image structure while modifying it.

For **outpainting** (generating into a void padded with white pixels), it is destructive. The fill zone is encoded by the VAE as a flat, near-zero-variance latent (white pixels → uniform signal). At `strength=0.85`, this flat structure is never fully replaced with Gaussian noise, so the UNet is denoising a biased latent that already contains the "memory" of the white padding. The result matches the research document's §5.2 "White Padding Fallacy": the void area generates smudgy, low-contrast content that looks like the model tried to colorize white fog.

**What was changed:**
```python
# AFTER — explicit None check; 0.0 is a valid strength value and must not trigger the fallback
strength=request.strength if request.strength is not None else 1.0,
```

The `or 0.85` form was also semantically wrong for a different reason: if a caller explicitly passed `strength=0.0`, `or` would silently substitute `0.85`.

---

### **Bug 3 — Deterministic ODE sampler used for outpainting**

**File:** `src/nodes/outpainting_node.py`, `_get_cn_pipe()`

**What was wrong:**

The outpainting pipeline inherited its scheduler from the base pipeline, which uses `DPMSolverMultistepScheduler` — a deterministic ODE solver. Deterministic samplers do not inject stochastic noise at intermediate timesteps; they converge on the mathematically exact solution implied by the initial noise state and the prompt conditioning.

For text2image, this is desirable: it produces consistent, reproducible outputs. For outpainting a structureless void, it is harmful. When the model lacks a strong structural signal in the early noise state (which it will, because the void is pure random Gaussian), a deterministic solver defaults to the **mean of the probability distribution** over plausible textures. The mean of a texture distribution is a blurry, averaged, featureless surface — exactly the "plasticky" artifact described in the research document's §7.

**What was changed:**
```python
# In _get_cn_pipe(), after building cn_pipe and before caching:
cn_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    cn_pipe.scheduler.config,
    use_karras_sigmas=True,
    algorithm_type="sde-dpmsolver++",
)
```

DPM++ 2M SDE (Stochastic Differential Equation variant) injects a calculated amount of Gaussian noise back into the latent at every denoising step. This continuous variance injection prevents the solver from locking into a blurry average, forcing the UNet to actively resolve micro-textures at each step rather than smoothing them away.

This scheduler override is scoped to `_cn_pipe_cache` only. The base pipeline used by text2image retains its original deterministic scheduler.

---

### **Bug 4 — Post-generation image compositing always active**

**File:** `src/nodes/outpainting_node.py`, `OutpaintingNode.__call__()`

**What was wrong:**

After the pipeline returned its output, the code unconditionally pasted the original pixel content back over the preserved region (outside the fill zone):

```python
# BEFORE — always ran
inv_mask = Image.fromarray(255 - np.array(mask), mode="L")
result.paste(orig_img, mask=inv_mask)
```

The rationale was to correct colour drift in the preserved region caused by the VAE round-trip. The side effect was a hard seam at the mask boundary: the generated content was softly blended by the pipeline's mask blur, then a sharp paste edge was stamped on top of it, negating the blend.

**What was changed:**

The compositing block now runs only when `composite_original=True` (default: `False`) is set in `OutpaintingInputs`. The pipeline's own latent-space blending is trusted for the seam.

---

## **3. What the Research Document Got Right**

| Research Claim | Verdict |
|---|---|
| §8: `guidance_rescale=0.7` is catastrophic on epsilon models | **Confirmed.** Was the single highest-impact bug. |
| §7: DPM++ 2M Karras converges to blurry averages in void space | **Confirmed.** Switching to SDE variant resolved the micro-texture flatness. |
| §5.2: Sub-1.0 denoising strength causes white padding to persist | **Confirmed.** The `or 0.85` default in `main.py` was exactly this failure mode. |
| §4.1: VAE tiling causes seam blurring | **Conditionally true.** Only active on the ROCm <24 GB path; not the cause in the primary test environment. |
| §6.2: FreeU skip dampening can suppress micro-texture | **Plausible but not confirmed.** FreeU remains active at the original values; the other fixes resolved the observed symptoms. |

## **4. What the Research Document Got Wrong or Missed**

| Research Claim | Actual Finding |
|---|---|
| §4.2: qfloat8 quantization is a risk factor | **Not applicable.** This pipeline uses neither `enable_sequential_cpu_offload()` nor 8-bit quantization. Section was theoretical noise. |
| §5.2: "White padding" is the mask initialization problem | **Partially wrong.** The TransformNode generates a correct binary fill mask; the pipeline zeros the fill zone before conditioning. The actual bug was the `strength=0.85` default, not mask content. |
| Implicit: `prediction_type` can be read from `unet/config.json` | **Wrong for SDXL.** The key is absent from all cached UNet configs. The scheduler config is the authoritative source. |
| No section: compositing paste creates a hard seam | **Missed entirely.** The post-generation pixel paste was cancelling the pipeline's own soft blend. |

## **5. Files Changed**

| File | Change |
|---|---|
| `src/nodes/outpainting_node.py` | Gate `guidance_rescale` on `pipe.scheduler.config.get("prediction_type")` |
| `src/nodes/outpainting_node.py` | Patch `cn_pipe.scheduler` to `DPMSolverMultistepScheduler` with `algorithm_type="sde-dpmsolver++"` |
| `src/nodes/outpainting_node.py` | Add `composite_original: bool = False` field; compositing block now conditional |
| `main.py` | `strength=request.strength or 0.85` → `strength=request.strength if request.strength is not None else 1.0` |
