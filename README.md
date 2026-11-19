# Z-Image-Turbo local CLI and MCP

Generate images locally with Z-Image-Turbo, a persistent stable-diffusion.cpp CUDA engine and a single queued service. The selected image model is `z_image_turbo-Q6_K.gguf`; the text encoder is Qwen3-4B-Instruct-2507 `Q4_K_M` and the VAE is `ae.safetensors`. The image model and text encoder are quantized. The engine is configured to put all three model components on the RTX 3060, and retains them between jobs until stopped.

The service pins `CUDA_VISIBLE_DEVICES` to the RTX 3060 UUID `GPU-1fbff0da-1530-2c25-e1df-4f00ecd1601b`; set `ZIMAGE_GPU_UUID` before launch if your 3060 has a different UUID. It refuses to start the engine if less than 10,500 MiB is free. The engine receives a 10.5 GiB budget, and the worker samples total card usage during jobs and stops the engine if it exceeds 11,500 MiB. Sampling is not a hard allocator limit and can miss short spikes or shared memory allocations. Stop other GPU services before loading this model.

## Use

The launcher is `E:\TOOLS\ZIMAGE-TURBO-MCP\bin\zimage.cmd`; it is added to the user PATH for new terminals. `local-zimage` is registered in Codex; start a new Codex session to load it. Other MCP clients can use `mcp-config.example.json`.

```powershell
zimage doctor
zimage generate --prompt "A red ceramic teapot on a white table, soft daylight" --width 1024 --height 1024 --steps 8 --seed 42 --wait --out teapot.png
zimage generate --prompt "A mountain at sunrise"
zimage status JOB_ID
zimage wait JOB_ID
zimage cancel JOB_ID
zimage download JOB_ID --out image.png
zimage stop
```

The default settings are 1024x1024, 8 steps, and CFG 1.0, following the stable-diffusion.cpp Z-Image-Turbo example. `--wait` waits for a result; otherwise generation returns a job ID. `--request-id` makes retries idempotent. MCP offers `generate_image`, `get_image_job` (optional `include_preview`), `cancel_image_job`, and `image_service_status`.

The service listens on `127.0.0.1:8773` using the bearer token stored in `data/token.txt`. The internal engine listens on an unauthenticated random loopback port and is intended for trusted local users. Do not share the token. Jobs, images, and engine logs are under `data/outputs` and `data/engine.log`. Interrupted jobs are marked failed after restart; cancellation of a running job unloads the engine. The resident weights consume GPU memory until `zimage stop` or cancellation.

## Files and setup

The model downloads are pinned to Hugging Face revisions and checked against SHA-256 digests in `assets.py`. Run `zimage setup-models` to download or verify them. The current machine has a private copy of the CUDA stable-diffusion.cpp build from MOSS-MCP under `vendor/sd`, so MOSS is not a runtime dependency. To recreate the Python environment: `uv venv --python 3.12 .venv`, then `uv pip install --python .venv\Scripts\python.exe -r requirements.txt`. A compatible CUDA build of stable-diffusion.cpp containing `sd-server.exe`, `sd-cli.exe` and its CUDA DLLs is also required under `vendor/sd`.

See [research.md](docs/research.md) for the upstream sources and the differences between the Diffusers and stable-diffusion.cpp sampling conventions. Tests: `.venv\Scripts\python.exe -m unittest discover -s tests -v`.

## Verified on this machine

On the RTX 3060, an eight-step 1024x1024 test succeeded. The cold engine job took about 33 seconds with the files in the OS disk cache; the following 1024x1024 job took about 20 seconds. The sampled GPU peak was 8,255 MiB across the two runs. Engine logs report 7,555 MiB of parameters in VRAM: 5,019 MiB diffusion, 2,376 MiB text encoder, and 160 MiB VAE, with 0 MiB model parameters in RAM. A cold filesystem cache may take longer. Prompt length and resolution can change speed and memory use.
