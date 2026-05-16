# **Architectural and Pipeline Analysis of Degradation in Outpainting Workflows Under Memory Constraints**

## **1\. Introduction to the Generative Outpainting Paradigm**

Generative outpainting utilizing latent diffusion models represents a highly complex implementation of computational image synthesis. Unlike conventional text-to-image generation, which constructs a composition globally from a uniform field of Gaussian noise, outpainting operates as a highly localized boundary-value problem. The neural network is tasked with hallucinating contextually coherent, structurally accurate, and texturally consistent extensions to an existing image boundary, necessitating a seamless extrapolation of semantic patterns into an adjacent void.1

A frequently observed and persistently frustrating phenomenon in high-resolution outpainting pipelines—particularly within consumer or mid-tier hardware environments constrained to less than 24 gigabytes of Video Random Access Memory (VRAM)—is the generation of spatial extensions that exhibit precise macro-structural accuracy but suffer from severe micro-textural degradation. These localized regions often present as overwhelmingly noisy, artificially plasticky, or distinctly blurred, compromising the photorealism and cohesion of the final composite image.1

Based on a specific set of operational parameters—including the utilization of DPM++ 2M Karras sampling over 40-50 steps, the active engagement of FreeU architectural adjustments, the strict use of 9-channel inpainting models over standard 4-channel models, and the application of Classifier-Free Guidance (CFG) reguidance set to 0.7—this report provides a granular examination of the computational, mathematical, and architectural factors driving noise and blur artifacts. By systematically dissecting memory offloading mechanics, Variational Autoencoder (VAE) limitations, deterministic sampling algorithms, and the underlying mathematics of terminal Signal-to-Noise Ratio (SNR) schedules, this analysis establishes a comprehensive framework for diagnosing and correcting textural failures in memory-constrained diffusion pipelines.

## **2\. Phenomenological Analysis of Community Troubleshooting Data**

The symptoms of outpainting degradation are not isolated anomalies; they represent a systemic pattern of pipeline behavior documented extensively across developer forums, GitHub repositories, and community discussion platforms. An analysis of these communal troubleshooting efforts reveals critical insights into the mechanical triggers of textural degradation.

### **2.1 The "Blurry Mess" and the Img2Img Iteration Loop**

A predominant theme across community forums is the rapid deterioration of image quality during iterative outpainting workflows. Users utilizing custom scripts based on the Hugging Face diffusers library frequently report that outpainting an image across four to five consecutive passes results in the outer boundaries collapsing into a "blurry mess".3

This degradation is fundamentally tied to the inherent lossiness of the image-to-image (img2img) translation process.3 Diffusion models do not operate in pixel space; they operate in a highly compressed latent space. When a newly outpainted image is sent back into the pipeline for further expansion, the pixel-space image must be encoded back into the latent domain using the VAE encoder. This latent representation is highly compressed, operating at an 8x spatial reduction.3

Community experts have mathematically likened this iterative encoding and decoding process to continuously converting a lossless PNG image into a lossy JPEG file and back again.3 With each consecutive outpainting pass, the high-frequency details—such as sharp architectural edges, skin pores, and foliage textures—are systematically smoothed out by the VAE's compression algorithms. By the third iteration, the accumulated degradation guarantees a soft, undefined texture, regardless of the prompt accuracy.3 Furthermore, the noise injected into the latent space during subsequent generation passes interacts unpredictably with these compression artifacts, creating localized clusters of visual static.3 Advanced users recommend preserving the generation entirely within the latent space across the entire workflow, decoding to pixel space only once the final outpainting mask has been completely traversed.3

### **2.2 Grey Blobs and Mid-Generation Collapse**

Another frequently documented symptom within \<24GB VRAM environments is the sudden appearance of grey, smudgy blobs, particularly when attempting to restore or alter small details within larger outpainted fields.4 Developers operating on consumer GPUs (such as AMD 7800s or lower-tier NVIDIA cards) observe that the inpainting process appears to function correctly up until approximately 75% of the generation schedule, at which point the hallucinated region suddenly reverts to a blurry, pixelated mass.4

This specific symptom is highly diagnostic of tensor precision overflow. In constrained VRAM environments, scripts aggressively deploy fp16 (16-bit floating point) quantization to halve the memory footprint. While the UNet backbone is generally resilient to half-precision computation, the VAE and certain cross-attention mechanisms are profoundly fragile.5 When the latent activations transition between the heavily conditioned original image and the unconditioned outpainting void, the numerical values can spike, causing fp16 tensors to output Not a Number (NaN) values. The pipeline attempts to average these failures out, resulting directly in the "grey blob" or "fried" static observed by developers.4

### **2.3 The Dissonance Between Prompt Fidelity and Texture**

Further analysis of user experiences highlights a persistent dissonance between generative structures and final textures. Users repeatedly report that while outpainting scripts correctly understand the prompt (e.g., correctly placing a mountain or a building in the expanded void), the resulting rendering lacks all inherent detail, appearing overly soft or marred by a continuous, granular noise floor.6 This points definitively toward an issue not with the text encoder's ability to condition the generation, but with the sampling scheduler's ability to resolve micro-variance during the final denoising steps.

| Observed Symptom in Community Forums | Technical Origin | Pipeline Location |
| :---- | :---- | :---- |
| Cumulative "Blurry Mess" across multiple passes.3 | Lossy VAE encoding/decoding loops acting as low-pass filters.3 | Iterative Pipeline Architecture |
| "Fried" noise or grey blobs at high step counts.4 | fp16 numerical overflow in the VAE or attention blocks.5 | Precision Quantization |
| Accurate macro-structure with "poo" or blurred textures.6 | Mismatched denoising strength on absolute white/black padding.6 | Mask Initialization |

## **3\. The Architecture of Inpainting Models vs. Standard Base Models**

The diagnostic data indicates that swapping the pipeline from a dedicated inpainting model to a standard base model temporarily resolves the noise and blur, yielding "clear" results, but entirely fails to create an accurate structural continuation of the outpaint. This behavioral divergence is deeply rooted in the tensor architecture of the UNet's input layer, defining exactly why dedicated models are necessary, yet prone to artifact propagation.

### **3.1 The 4-Channel Standard UNet Constraint**

Standard latent diffusion models, which are explicitly designed for txt2img generation, possess an input convolution layer restricted to exactly 4 channels. These 4 channels correspond entirely to the encoded, noisy latent representation of the image being generated.1

When forced into an outpainting workflow, a standard base model relies on a crude computational hack: the unmasked area is protected by forcing the original pixels back into the latent space after every single denoising step, while the masked area is allowed to denoise freely. Because the standard UNet physically cannot "see" the mask as an input parameter, it attempts to denoise the entire canvas as a single, global entity, guided exclusively by the text prompt.1

This explains the observed phenomenon perfectly. Because the model is unconstrained by boundary mathematics, it generates a mathematically pristine, clear, and sharp texture.1 However, because it is blind to the seam, the semantic transition across the boundary is highly erratic. The model might generate a clear sky right next to an outpainted brick wall because it has no architectural mechanism to force contextual continuation.

### **3.2 The 9-Channel Inpainting UNet Dynamics**

To resolve the boundary blindness of standard models, dedicated inpainting models are surgically modified at the architectural level. The input convolution layer is expanded from 4 channels to 9 channels.2 These channels consist of:

1. **4 Channels:** The standard encoded, noisy latent tensor.  
2. **4 Channels:** The encoded latent representation of the unmasked original image context.  
3. **1 Channel:** The binary mask defining the strict boundary between the original image and the void.

This 9-channel architecture forces the network to rigorously abide by the structural logic of the unmasked image, explicitly referencing the existing pixels to inform the generation of the void.2 This guarantees macro-structural accuracy, fulfilling the prompt while seamlessly continuing existing geometries.

However, this hyper-awareness of the boundary is a double-edged sword. Because the network is heavily biased toward analyzing the adjacent context, it becomes highly susceptible to perpetuating any existing degradation found at the absolute edge of the expanded canvas.2 If the outpainting pipeline utilizes discrete resizing operations on the image or the mask to fit within the \<24GB VRAM constraint, these resizing operations introduce jagged, aliased, or stair-step artifacts along the 1-channel mask boundary.8 The 9-channel UNet interprets these aliased edges as literal structural commands, diffusing micro-noise and blur outward from the seam. The strict adherence to the context channels is exactly why the structure is accurate, but the texture becomes blurred and noisy if the mask initialization is mathematically imperfect.7

## **4\. The VAE Bottleneck: Memory Optimization and Textural Degradation**

Operating high-resolution diffusion models on hardware constrained to less than 24GB of VRAM introduces severe memory bottlenecks, the most prominent of which occurs during the final VAE decoding stage. To prevent catastrophic CUDA Out-Of-Memory exceptions, diffusion libraries employ spatial division strategies that fundamentally compromise image fidelity.

### **4.1 The Mechanics of enable\_vae\_tiling()**

The VAE is tasked with taking the final 4-channel latent representation and projecting it back into a massive 3-channel (RGB) pixel space array. For large outpainted canvases, this tensor projection exceeds available memory buffers. To circumvent this, scripts universally implement memory-saving interventions such as enable\_vae\_tiling().9

When VAE tiling is engaged, the algorithm partitions the vast latent tensor into smaller, manageable, overlapping matrices (tiles) and processes the decoding sequentially, stitching the results back together in system RAM.9 While highly efficient for memory preservation, VAE tiling intrinsically compromises spatial coherence and textural continuity.11

The decoding process in modern autoencoders relies heavily on self-attention mechanisms embedded within its middle blocks. These attention blocks analyze the entirety of the latent space to maintain global color normalization, contrast scaling, and high-frequency textural continuity across the image. When the latent tensor is cleaved into tiles, the receptive field of the VAE's attention blocks is brutally truncated. The network becomes completely blind to the global context of the image.

The visual manifestation of this architectural blindspot aligns seamlessly with the blur and noise observed in the outpainted regions.11 Because the UNet operates globally on the latent space, the structural geometry remains accurate. However, as the VAE decodes the tiles independently, it fails to match the high-frequency textural statistics of the original image core with the newly generated outer bounds. To hide the seams between the decoded tiles, the decoder mathematically averages out pixel values near the boundaries. This averaging inherently acts as a severe low-pass filter, resulting in a region that accurately represents the prompted structure but is covered in a soft, blurry haze or plagued by disassociated, tile-specific noise patterns.11

### **4.2 CPU Offloading and Quantization Artifacts**

Beyond the VAE, VRAM constraints force the deployment of aggressive CPU offloading models. Functions such as enable\_model\_cpu\_offload() and enable\_sequential\_cpu\_offload() dynamically move weights between system RAM and VRAM to keep peak memory usage below the hardware ceiling.13

While enable\_model\_cpu\_offload() maintains the structural integrity of the neural network by moving whole discrete models (maintaining precision), enable\_sequential\_cpu\_offload() pushes the optimization further by offloading the network on a layer-by-layer basis.15 This continuous shuttling of individual neural layers across the PCIe bus can disrupt the continuity of cross-attention states.

Furthermore, memory-constrained pipelines often combine sequential offloading with dynamic quantization, scaling the transformer blocks to 8-bit floats (qfloat8) to reduce the memory footprint further.15 Quantizing an inpainting UNet to 8-bit precision annihilates the fine-grained gradient nuances required for smooth blending. In generative outpainting, the transition from the hard edge of the original image into the hallucinated void requires incredibly precise decimal representations in the attention matrices. Truncating these decimals forces the network to round out features arbitrarily, generating harsh boundaries and injecting coarse, blocky noise into the expanded regions.14

| Memory Optimization | Operational Mechanism | Artifact Risk Profile in Outpainting Workflows |
| :---- | :---- | :---- |
| enable\_vae\_tiling() | Slices latent tensor into independent blocks for sequential VAE decoding.9 | **High.** Truncated self-attention causes contrast flattening, seam blurring, and textural inconsistency.11 |
| enable\_vae\_slicing() | Computes batch decodes sequentially.19 | **Low.** Only affects batch processing, leaving individual image spatial coherence largely intact.9 |
| Sequential CPU Offload | Shuttles individual UNet layers between CPU and GPU.16 | **Moderate.** PCIe bus latency and attention state disruption can cause micro-stutters in feature generation.15 |
| qfloat8 Quantization | Truncates neural weights to 8-bit precision.15 | **Severe.** Loss of gradient precision causes mathematical rounding errors, manifesting as coarse noise.18 |

## **5\. Denoising Strength, Mask Initialization, and the "White Area" Paradox**

A critical diagnostic anchor in the provided data is the observation that reducing the denoising strength immediately restores "white areas" in the output. This behavioral observation conclusively proves that the outpainting extension matrix is being initialized incorrectly, creating a mathematical paradox that the diffusion model resolves by blurring the output.

### **5.1 The Mechanics of Denoising Strength in Outpainting**

In standard image-to-image (img2img) generation, the denoising strength operates as a percentage of the diffusion schedule.6 A denoising strength of 0.5 injects 50% noise into the existing pixel structure and denoises backward, maintaining the underlying composition while altering surface details. A strength of 1.0 (100%) entirely replaces the latent space with pure Gaussian noise, destroying the original image and generating an entirely new composition from scratch.6

Generative outpainting is not structurally synonymous with internal inpainting; it is not replacing existing pixels with new ones, it is fabricating a universe where previously there was an absolute void. Therefore, when setting up an outpainting mask, the newly expanded canvas area must be filled with pure, uncorrelated Gaussian noise (often labeled "Latent Noise" in community interfaces).1 Consequently, the denoising strength over this void must be set to 1.0. The UNet requires a full schedule of pure noise to completely forge new semantic concepts within that boundary.

### **5.2 The White Padding Fallacy**

The fact that reducing the denoising strength reveals white areas indicates that the pipeline is padding the expanded outpainting canvas with absolute white pixels, rather than initializing it with latent noise.1 Furthermore, it implies the mask content setting is likely defaulting to "Original" or "Fill" instead of "Latent Noise."

If the script pads the expanded borders with empty white pixels, setting a denoising strength of anything less than 1.0 (e.g., 0.7) creates an unresolvable computational trap.6 The VAE encodes the stark white border into a flat, highly uniform latent vector. The 0.7 denoising strength then injects only partial noise into this uniform white flatland. Because the noise is only partial, the mathematical structure of the white padding persists in the latent space.

When the UNet attempts to generate the outpainted prompt, it lacks the total variance required to hallucinate complex shapes. It is forced to conform to the underlying flat structure. The result is that the model simply blurs the existing white pixels with vague, smudgy colors derived from the prompt's conditioning. Thus, lowering the denoising strength does not fix the blur; it is the explicit cause of it.6 It halts the model's ability to overwrite the boundary padding, presenting as a soft, unstructured haze. Outpainting initialization demands a 1.0 denoising strength over a pure noise initialization to generate crisp, high-frequency textures.

## **6\. The FreeU Intervention and Structural Integrity**

The application of FreeU is another critical parameter in the user's implementation. The observation that disabling FreeU results in "worse structure" perfectly aligns with the architectural mechanisms of the UNet and highlights why it must remain active, despite its complex interaction with image noise.

### **6.1 The Role of UNet Skip Connections**

The UNet architecture utilized in Latent Diffusion Models is composed of a contracting path (the backbone), which compresses the image into high-level, low-frequency semantic representations, and an expanding path, which rebuilds the spatial resolution. These two paths are bridged by "skip connections".21 Skip connections bypass the deep backbone, feeding high-frequency, fine-grained details directly from the early layers to the late layers.

While skip connections are vital for image sharpness, they possess an architectural flaw: they can cause the model to over-index on high-frequency noise and overlook the global, low-frequency semantics generated by the backbone.21 In outpainting, where the model must invent global structures (like mountains or buildings) in the void, an over-reliance on skip connections leads to catastrophic structural collapse. The network generates highly detailed but semantically meaningless textures that fail to cohere into recognizable objects.

### **6.2 FreeU Rebalancing and Noise Interaction**

FreeU intervenes by mathematically rebalancing the weights between the UNet's backbone and its skip connections during inference, without requiring any additional training or fine-tuning.21 It dynamically amplifies the backbone feature maps while slightly attenuating the skip connection feature maps. This forces the model to prioritize global semantics, which directly explains the user's observation: disabling FreeU causes the outpainted structures to degrade or fail entirely.

However, while FreeU secures structural integrity, its attenuation of the skip connections can inadvertently suppress the generation of desirable micro-textures (like film grain or natural noise), pushing the image toward a smoother, slightly more artificial appearance. If other parameters in the pipeline (such as the sampler or CFG rescale) are already contributing to textural blur, the FreeU rebalancing can magnify the perception of that blur by enforcing rigid structural gradients over organic noise. FreeU must remain engaged to hold the hallucinated structure together, but the textural artifacts must be mitigated via sampler and guidance optimization.21

## **7\. Sampler Dynamics: Deterministic vs. Stochastic Solvers**

The configuration relies heavily on the DPM++ 2M Karras sampler executed over 40 to 50 inference steps. While this specific Ordinary Differential Equation (ODE) solver is widely regarded as a superior, highly efficient configuration for standard txt2img generation, its intrinsic mathematical characteristics actively hinder the generation of crisp textures in outpainting tasks.22

### **7.1 The Blurring Effect of Deterministic Solvers**

DPM++ 2M Karras belongs to a class of solvers known as deterministic multistep samplers.23 Deterministic algorithms converge to a mathematically exact solution based entirely on the initial noise state and the prompt conditioning. Crucially, they do not inject any additional stochastic noise into the latent space during the step-by-step sampling process.22 The "Karras" designation refers to a specific noise schedule that concentrates computational steps at the critical low-noise end of the generation cycle to rapidly refine details.

In generative outpainting, the network is fundamentally deprived of context. It is looking into an abyss of noise and attempting to extrapolate a continuation of a complex scene. Because deterministic samplers like DPM++ 2M do not introduce new variance at intermediate timesteps, if the model cannot find a strong structural pattern in the pure noise early in the generation, it defaults to calculating the mathematical mean of the probability distribution.23

Visually, generating the "mean" of a texture results directly in flat, blurry, plasticky, and overly-smooth surfaces.23 The lack of continuous noise injection means the model has no chaotic "material" to work with if it gets stuck attempting to hallucinate highly complex, high-variance textures like foliage, skin pores, or intricate architectural weathering. Increasing the step count to 40 or 50 does not solve this; it merely gives the deterministic solver more time to perfect the mathematically smooth average, locking the blur in permanently.22

### **7.2 The Ancestral and SDE Imperative for Outpainting**

To counteract this deterministic smoothing, community consensus and diffusion mathematics heavily point toward the utilization of Ancestral samplers or Stochastic Differential Equation (SDE) solvers for heavy inpainting and outpainting.23

Algorithms such as Euler a (Ancestral) or DPM++ 2M SDE Karras operate stochastically.24 At every single inference step, after removing a calculated amount of noise, these algorithms deliberately inject a calculated amount of new, random Gaussian noise back into the latent space.24 This continuous stochastic injection prevents the model from ever settling into a blurry mathematical average. The added variance constantly forces the UNet to interpret, reinterpret, and resolve micro-structures, resulting in outputs that are significantly sharper, highly detailed, and perceptually matched to the photographic or illustrative noise floor of the original image.23

Transitioning the outpainting script from a deterministic solver to a stochastic solver is one of the most mechanically sound methods for resolving "smooth" or "blurry" hallucinations without altering the underlying model architecture.

| Sampler / Scheduler Category | Primary Mechanism | Suitability for Boundary Outpainting | Textural Artifact Risk |
| :---- | :---- | :---- | :---- |
| **DPM++ 2M Karras** | Deterministic ODE. No noise injected during steps.22 | **Poor.** Tends to converge to averages in unconditioned void space.23 | High risk of over-smoothed, blurry, or plasticky textures. |
| **Euler a (Ancestral)** | Stochastic. Injects variance at every generation step.24 | **Excellent.** Continuous variance forces UNet to hallucinate crisp micro-details. | Low. Excellent at mimicking natural image grain. |
| **DPM++ 2M SDE Karras** | Stochastic Differential Equation solver. Highly detailed.24 | **Excellent.** Balances deterministic convergence with stochastic noise injection.26 | Low. Produces sharp textures, though at the cost of inference speed. |

## **8\. Classifier-Free Guidance (CFG) Rescale at 0.7: A Mathematical Misalignment**

The absolute most critical optimization failure in the provided configuration is the manipulation of the guidance\_rescale parameter. Setting the CFG reguidance (rescale) to 0.7 represents a highly specific mathematical operation designed for a distinct subset of diffusion models. Applying it broadly to standard inpainting pipelines results in catastrophic textural degradation.27

### **8.1 The Mathematics of CFG Rescale and Terminal SNR**

Classifier-Free Guidance (CFG) operates by extrapolating the difference between a conditioned prediction (derived from the text prompt) and an unconditioned prediction (derived from an empty prompt), pushing the generated latent vector further in the direction of the text prompt. Mathematically, this subtraction and multiplication process drastically inflates the standard deviation of the noise residual, which can lead to severe color saturation and overexposure in the final image.29

According to the foundational paper *Common Diffusion Noise Schedules and Sample Steps are Flawed* (Lin et al., 2023), traditional diffusion schedules fail to reach a terminal Signal-to-Noise Ratio (SNR) of zero.28 Because standard models are never exposed to pure Gaussian noise during training, inference generation from pure noise results in exposure clipping, forcing outputs into a medium-brightness, washed-out spectrum.28

To correct this, researchers developed models trained on v\_prediction objective functions paired with a guidance\_rescale multiplier. The rescale algorithm calculates the standard deviation of both the original unguided noise and the CFG-inflated noise, scaling the final output to match the original variance distribution. The formula applied within the pipeline is: noise\_cfg \= guidance\_rescale \* noise\_pred\_rescaled \+ (1 \- guidance\_rescale) \* noise\_cfg.29

### **8.2 The Variance-Crushing Effect on Standard Models**

The documentation for guidance\_rescale explicitly dictates that it must be used *exclusively* with models trained specifically on v\_prediction objective functions and paired with a schedule enforcing zero terminal SNR.21

If a standard epsilon-prediction model (which encompasses the vast majority of standard SD 1.5 and SDXL inpainting checkpoints) is subjected to a guidance\_rescale of 0.7, the mathematical alignment completely fractures.21 The rescale operation forcefully suppresses the variance of the generated noise back toward a theoretical mean that the epsilon model does not natively understand or expect.

In generative outpainting, the network relies entirely on high-variance noise to invent complex textures from nothing. When guidance\_rescale is forcefully applied to a standard epsilon inpainting model, the macro-structure (driven by the low-frequency prompt conditioning) forms accurately, but the micro-variance (the texture) is artificially crushed by the rescale multiplier.21 The pixels fail to differentiate smoothly, resulting directly in the symptom described: an outpainted region that looks compositionally correct but texturally blurred, noisy, and artificially flat. Removing the guidance\_rescale parameter entirely is computationally mandatory unless the developer is explicitly utilizing a custom v\_prediction fine-tuned checkpoint.21

| Pipeline Parameter | Intended Function | Effect on Standard epsilon Inpainting Models | Effect on v\_prediction Models |
| :---- | :---- | :---- | :---- |
| guidance\_scale (CFG) | Amplifies text prompt adherence. | Increases contrast; high values cause color burning/saturation. | Increases contrast; requires rescale to prevent blowout. |
| guidance\_rescale=0.7 | Normalizes standard deviation to fix exposure.29 | **Catastrophic Texture Loss.** Induces blurring and high-frequency static noise by crushing variance.21 | Balances exposure; enables extreme dark/light dynamic range.28 |
| Terminal Zero SNR | Forces pure Gaussian noise at final timestep ![][image1]. | Invalidates model prior; generates gray/black blobs and destroys output.28 | Accurate pure-noise generation for deep blacks and bright whites. |

## **9\. Synthesis of Corrective Frameworks**

The presence of accurate macroscopic structure juxtaposed against degraded, noisy, or blurred microscopic textures in low-VRAM outpainting pathways indicates a confluence of optimization bottlenecks and mathematical misconfigurations. The artifacts are not the result of a single failure, but a cascade of computational compromises required to run generative AI under 24GB of VRAM.

By synthesizing the phenomenological evidence from community forums with the underlying mathematics of latent diffusion, the corrective framework becomes clear. The textural blur driven by VAE tiling 11 must be mitigated by minimizing the number of pixel-to-latent encoding loops, preserving the workflow entirely in the latent space until the final step.3 The paradoxical "white area" restoration proves that the outpainting mask is being initialized with absolute pixel values rather than pure latent noise, trapping the UNet when denoising strengths below 1.0 are applied.6 This necessitates forcing the pipeline to initialize the boundary void with pure Gaussian noise and clamping the denoising strength strictly to 1.0.

Furthermore, the deterministic smoothing introduced by the DPM++ 2M Karras sampler 23 must be counteracted by transitioning to stochastic, noise-injecting solvers like Euler a or DPM++ 2M SDE.24 Finally, the misapplication of guidance\_rescale=0.7 on standard epsilon models acts as the primary catalyst for granular noise and variance crushing.21 Disabling this rescale operation will immediately restore the high-frequency variance required for the network to render sharp, cohesive, and photorealistic textures across the outpainted boundary.

#### **Works cited**

1. Inpainting and Outpainting with Stable Diffusion \- MachineLearningMastery.com, accessed May 16, 2026, [https://machinelearningmastery.com/inpainting-and-outpainting-with-stable-diffusion/](https://machinelearningmastery.com/inpainting-and-outpainting-with-stable-diffusion/)  
2. A Generative Image Inpainting Model Based on Edge and Feature Self-Arrangement Constraints \- PMC, accessed May 16, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9581602/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9581602/)  
3. how to prevent the image quality from deteriorating with each inpaint ? : r/StableDiffusion, accessed May 16, 2026, [https://www.reddit.com/r/StableDiffusion/comments/17rdu4r/how\_to\_prevent\_the\_image\_quality\_from/](https://www.reddit.com/r/StableDiffusion/comments/17rdu4r/how_to_prevent_the_image_quality_from/)  
4. Inpainting produces grey blob regardless of settings : r/StableDiffusion \- Reddit, accessed May 16, 2026, [https://www.reddit.com/r/StableDiffusion/comments/15qf5c5/inpainting\_produces\_grey\_blob\_regardless\_of/](https://www.reddit.com/r/StableDiffusion/comments/15qf5c5/inpainting_produces_grey_blob_regardless_of/)  
5. Bad outputs from inpainting : r/StableDiffusion \- Reddit, accessed May 16, 2026, [https://www.reddit.com/r/StableDiffusion/comments/1ea6ptv/bad\_outputs\_from\_inpainting/](https://www.reddit.com/r/StableDiffusion/comments/1ea6ptv/bad_outputs_from_inpainting/)  
6. NEWBIE HERE: why are my outpainting renders such blurry messes? my settings are attached along : r/StableDiffusion \- Reddit, accessed May 16, 2026, [https://www.reddit.com/r/StableDiffusion/comments/1anau9o/newbie\_here\_why\_are\_my\_outpainting\_renders\_such/](https://www.reddit.com/r/StableDiffusion/comments/1anau9o/newbie_here_why_are_my_outpainting_renders_such/)  
7. SONIC: Spectral Optimization of Noise for Inpainting with Consistency \- arXiv, accessed May 16, 2026, [https://arxiv.org/html/2511.19985v2](https://arxiv.org/html/2511.19985v2)  
8. Aligned Stable Inpainting: Mitigating Unwanted Object Insertion and Preserving Color Consistency \- arXiv, accessed May 16, 2026, [https://arxiv.org/html/2601.15368v1](https://arxiv.org/html/2601.15368v1)  
9. Flux \- Hugging Face, accessed May 16, 2026, [https://huggingface.co/docs/diffusers/en/api/pipelines/flux](https://huggingface.co/docs/diffusers/en/api/pipelines/flux)  
10. Flux \- 文档- Hugging Face 文档, accessed May 16, 2026, [https://hugging-face.cn/docs/diffusers/api/pipelines/flux](https://hugging-face.cn/docs/diffusers/api/pipelines/flux)  
11. Anyone got any idea what might be causing the tiling artifacts on the top? : r/StableDiffusion, accessed May 16, 2026, [https://www.reddit.com/r/StableDiffusion/comments/1lp166h/anyone\_got\_any\_idea\_what\_might\_be\_causing\_the/](https://www.reddit.com/r/StableDiffusion/comments/1lp166h/anyone_got_any_idea_what_might_be_causing_the/)  
12. Help implementing Tiled Diffusion and Tiled VAE with Diffusers \- Hugging Face Forums, accessed May 16, 2026, [https://discuss.huggingface.co/t/help-implementing-tiled-diffusion-and-tiled-vae-with-diffusers/150354](https://discuss.huggingface.co/t/help-implementing-tiled-diffusion-and-tiled-vae-with-diffusers/150354)  
13. Stable Diffusion Image Generation | Hermes Agent \- nous research, accessed May 16, 2026, [https://hermes-agent.nousresearch.com/docs/user-guide/skills/optional/mlops/mlops-stable-diffusion](https://hermes-agent.nousresearch.com/docs/user-guide/skills/optional/mlops/mlops-stable-diffusion)  
14. Wan2.1 Full Parameter Training Guide \- VideoX-Fun \- GitHub, accessed May 16, 2026, [https://github.com/aigc-apps/VideoX-Fun/blob/main/scripts/wan2.1/README\_TRAIN.md](https://github.com/aigc-apps/VideoX-Fun/blob/main/scripts/wan2.1/README_TRAIN.md)  
15. Francis-Rings/StableAvatar: We present StableAvatar, the first end-to-end video diffusion transformer, which synthesizes infinite-length high-quality audio-driven avatar videos without any post-processing, conditioned on a reference image and audio. · GitHub, accessed May 16, 2026, [https://github.com/Francis-Rings/StableAvatar](https://github.com/Francis-Rings/StableAvatar)  
16. VideoX-Fun \- ComfyUI Cloud \- Comfy.ICU, accessed May 16, 2026, [https://comfy.icu/extension/aigc-apps\_\_VideoX-Fun](https://comfy.icu/extension/aigc-apps__VideoX-Fun)  
17. transformers.generator\_utils函数源码解析之RepetitionPenaltyLogitsProcessor-CSDN博客, accessed May 16, 2026, [https://blog.csdn.net/yangyanbao8389/article/details/121651056](https://blog.csdn.net/yangyanbao8389/article/details/121651056)  
18. GitHub \- aigc-apps/EasyAnimate: An End-to-End Solution for High-Resolution and Long Video Generation Based on Transformer Diffusion, accessed May 16, 2026, [https://github.com/aigc-apps/EasyAnimate](https://github.com/aigc-apps/EasyAnimate)  
19. diffusers/src/diffusers/pipelines/flux/pipeline\_flux\_fill.py at main \- GitHub, accessed May 16, 2026, [https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline\_flux\_fill.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux_fill.py)  
20. How to Inpaint and Mask \- Stable Diffusion AI | Fix Bad Hands\! \- YouTube, accessed May 16, 2026, [https://www.youtube.com/watch?v=Tcslol6Qiis](https://www.youtube.com/watch?v=Tcslol6Qiis)  
21. Controlling image quality \- Hugging Face, accessed May 16, 2026, [https://huggingface.co/docs/diffusers/v0.28.2/using-diffusers/image\_quality](https://huggingface.co/docs/diffusers/v0.28.2/using-diffusers/image_quality)  
22. The Stable Diffusion Dictionary: Every Term You'll Hit in Your First, accessed May 16, 2026, [https://dev.to/minatoplanb/the-stable-diffusion-dictionary-every-term-youll-hit-in-your-first-month-nj](https://dev.to/minatoplanb/the-stable-diffusion-dictionary-every-term-youll-hit-in-your-first-month-nj)  
23. Random document on Subject-Oriented Inpainting and Detailing in Stable Diffusion \+ my personal workflow. \- GitHub Gist, accessed May 16, 2026, [https://gist.github.com/DarkStoorM/4b1684e5d42532e8d55517e61001d97a](https://gist.github.com/DarkStoorM/4b1684e5d42532e8d55517e61001d97a)  
24. Sharing the experience of using DirectML for the new users. · lshqqytiger stable-diffusion-webui-amdgpu · Discussion \#84 \- GitHub, accessed May 16, 2026, [https://github.com/lshqqytiger/stable-diffusion-webui-amdgpu/discussions/84](https://github.com/lshqqytiger/stable-diffusion-webui-amdgpu/discussions/84)  
25. I made a comparison between the different samples. all the settings are the same foreach image except the sampler : r/StableDiffusion \- Reddit, accessed May 16, 2026, [https://www.reddit.com/r/StableDiffusion/comments/13lwlu7/i\_made\_a\_comparison\_between\_the\_different\_samples/](https://www.reddit.com/r/StableDiffusion/comments/13lwlu7/i_made_a_comparison_between_the_different_samples/)  
26. Every midjourney user after they see what can be done for free locally with SDXL. \- Reddit, accessed May 16, 2026, [https://www.reddit.com/r/StableDiffusion/comments/15h7ndw/every\_midjourney\_user\_after\_they\_see\_what\_can\_be/](https://www.reddit.com/r/StableDiffusion/comments/15h7ndw/every_midjourney_user_after_they_see_what_can_be/)  
27. batched sdxl \- Github-Gist, accessed May 16, 2026, [https://gist.github.com/CoffeeVampir3/3065491a77ef6c8d3c57953d94601e2c](https://gist.github.com/CoffeeVampir3/3065491a77ef6c8d3c57953d94601e2c)  
28. Stable diffusion 2 \- Hugging Face, accessed May 16, 2026, [https://huggingface.co/docs/diffusers/v0.18.0/en/api/pipelines/stable\_diffusion/stable\_diffusion\_2](https://huggingface.co/docs/diffusers/v0.18.0/en/api/pipelines/stable_diffusion/stable_diffusion_2)  
29. faceswap with inpaint \#89 \- instantX-research/InstantID \- GitHub, accessed May 16, 2026, [https://github.com/instantX-research/InstantID/pull/89/files](https://github.com/instantX-research/InstantID/pull/89/files)  
30. diffusers/src/diffusers/pipelines/ltx2/pipeline\_ltx2\_image2video.py at main \- GitHub, accessed May 16, 2026, [https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ltx2/pipeline\_ltx2\_image2video.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ltx2/pipeline_ltx2_image2video.py)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAVCAYAAAByrA+0AAAApElEQVR4XmNgGD7ACYjvAvEjIjADIxBPAeKVQKwA5YPAHCD+B8QeUD4zENuDGOJAvAqIxaASICAIxKeB+AEQSyOJ84AIFyAuRBIEAX0g/gTEa4CYBUkcZBBDKBCrIQmCQDQQ/wficjRxYTQ+HIDc/xuIbdAlsAFc7scJjIH4KwOm+3ECXO7HCkBxMJ9hwN0PioNzQPyOAeJ2GP4CxNcZIIYMKwAAzdQn+1ncQ04AAAAASUVORK5CYII=>