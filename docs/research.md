# Z-Image-Turbo local inference research

Reviewed 2026-11-20. These are implementation choices, not measurements on the RTX 3060.

## Model behavior

[Tongyi-MAI's official model card](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) describes Z-Image-Turbo as a 6B distilled text-to-image model, designed for eight neural function evaluations, with bilingual English and Chinese text rendering. Turbo is a generation model; its separate Z-Image-Edit sibling handles editing. The official Diffusers example uses 1024 x 1024, `num_inference_steps=9` (eight diffusion forwards), and `guidance_scale=0.0`. This is a Diffusers-specific convention, not a universal requirement for other runtimes.

[stable-diffusion.cpp's Z-Image guide](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/z_image.md) documents the runtime's separate diffusion model, Qwen3 4B encoder, and VAE, and demonstrates Z-Image-Turbo with `--steps 8 --cfg-scale 1.0`. It lists compatible GGUF sources and also documents CPU offload for smaller GPUs. The exact 12 GB full-residency fit has to be verified for the selected model weights, resolution, prompts, runtime, and driver.

## Files to evaluate

- [leejet/Z-Image-Turbo-GGUF](https://huggingface.co/leejet/Z-Image-Turbo-GGUF/tree/main): `z_image_turbo-Q6_K.gguf` is 5.26 GB; `z_image_turbo-Q8_0.gguf` is 6.58 GB; `z_image_turbo-Q4_K.gguf` is 3.86 GB. These are quantizations of the Turbo transformer. The Q6_K candidate is a quality/size compromise, not an externally validated optimal quantization.
- [unsloth/Qwen3-4B-Instruct-2507-GGUF](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/tree/main): `Qwen3-4B-Instruct-2507-Q8_0.gguf` is 4.28 GB; `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` is 2.50 GB. The stable-diffusion.cpp guide explicitly uses this Q4 encoder. The upstream source for this quantized encoder is third-party; its use can change output compared with the original BF16 encoder.
- [Comfy-Org/Z-Image-Turbo](https://huggingface.co/Comfy-Org/Z-Image-Turbo/tree/main/split_files/vae): `split_files/vae/ae.safetensors` is 335 MB. The stable-diffusion.cpp Z-Image guide also lists the FLUX.1 VAE.

As a starting estimate, Q6 diffusion + Q8 encoder + VAE occupy about 9.88 GB on disk, leaving nominal but unverified space for inference buffers on a 12 GB card. Q6 + Q4 encoder + VAE occupy about 8.10 GB on disk, leaving more headroom. File sizes are not a VRAM guarantee: model tensor layout, CUDA contexts, allocations and generation dimensions also matter. Use explicit GPU parameter and computation placement, inspect the engine's per-component placement and driver memory metrics, and test at intended resolutions.

## Runtime and deployment

The [stable-diffusion.cpp server API](https://github.com/leejet/stable-diffusion.cpp/blob/master/examples/server/api.md) exposes synchronous `POST /sdapi/v1/txt2img` and native asynchronous `POST /sdcpp/v1/img_gen`, plus job status and cancellation. A persistent server can retain a loaded context between requests; verify measured warm performance and actual resident allocation rather than assuming this. Bind the engine to loopback, pin the intended RTX 3060 with its CUDA UUID, serialize jobs, and ensure shutdown and cancellation release child processes. Since the selected runtime's GPU controls are not a proof against every driver/shared-memory allocation, verify peak memory experimentally and preserve a safe margin.
