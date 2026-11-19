"""Download the pinned model components and check their SHA-256 digests."""
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSETS = [
    ('z_image_turbo-Q6_K.gguf', 'https://huggingface.co/leejet/Z-Image-Turbo-GGUF/resolve/c61c0e422dc8b541b7548cf33a4ef8302b0f8085/z_image_turbo-Q6_K.gguf', 5263239104, '319f627beac8059b7546f36a7b4d5097b7f4ee6a1fc37585d0f75ca1d12d01af'),
    ('Qwen3-4B-Instruct-2507-Q4_K_M.gguf', 'https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/a06e946bb6b655725eafa393f4a9745d460374c9/Qwen3-4B-Instruct-2507-Q4_K_M.gguf', 2497281120, '3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597'),
    ('ae.safetensors', 'https://huggingface.co/Comfy-Org/Z-Image-Turbo/resolve/6fc90a3b1b653e935a0d175e260736de25b84df5/split_files/vae/ae.safetensors', 335304388, 'afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38'),
]

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def check(path, size, expected):
    return path.is_file() and path.stat().st_size == size and digest(path) == expected

def setup():
    directory = ROOT / 'models'
    directory.mkdir(exist_ok=True)
    for name, url, size, expected in ASSETS:
        dst = directory / name
        part = directory / (name + '.part')
        if check(dst, size, expected):
            print(f'Verified {name}', flush=True)
            continue
        print(f'Downloading {name}', flush=True)
        subprocess.run(['curl.exe', '-L', '--fail', '--retry', '10', '--retry-delay', '5', '-C', '-', '-o', str(part), url], check=True)
        if not check(part, size, expected):
            raise RuntimeError(f'SHA-256 or size mismatch: {part}')
        part.replace(dst)
        print(f'Verified {name}', flush=True)

if __name__ == '__main__':
    setup()
