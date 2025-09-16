
# comfyui_modal_app.py
#
# ComfyUI (headless) on Modal — Qwen Image Edit ready
#
# Usage (local dev):
#   pip install modal
#   modal token new
#   modal run comfyui_modal_app.py::main --workflow-file="/mnt/data/1shotDatasetQWEN ICekiub FREE.json"
#
# Deploy:
#   modal deploy comfyui_modal_app.py
#   modal call comfyui_modal_app::ComfyUIRunner.run_workflow --workflow-file="/mnt/data/1shotDatasetQWEN ICekiub FREE.json"
#
# Notes:
# - Caches models in a Modal Volume "comfyui-models" to avoid repeated downloads.
# - Tries to clone a few common Qwen/Miaoshouai custom nodes; failures are non-fatal.
# - Focuses on executing a workflow JSON via the HTTP API (no web UI exposure).
# - Default GPU: L4. Change to A10G/A100 as desired.
#
# Environment variables you can override on call:
#   HF_TOKEN (optional, only if the repo requires it)
#
from __future__ import annotations

import os
import time
import json
import re
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, Optional, List

import modal

APP_NAME = "comfyui-qwen-serverless"
app = modal.App(APP_NAME)

# ---------- Persistent volumes ----------
MODELS_VOL = modal.Volume.from_name("comfyui-models", create_if_missing=True)
OUTPUTS_VOL = modal.Volume.from_name("comfyui-outputs", create_if_missing=True)

# ---------- Image ----------
# CUDA 12.1 PyTorch wheels; ComfyUI deps; git; ffmpeg; build tools
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "ffmpeg",
        "wget",
        "curl",
        "ca-certificates",
        "libgl1",
        "libglib2.0-0",
        "build-essential",
        "python3-dev",
        "pkg-config",
    )
    # Torch stack (CUDA 12.1)
    .pip_install(
        "torch==2.4.0+cu124",
        "torchvision==0.19.0+cu124",
        "torchaudio==2.4.0+cu124",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    # Core deps
    .pip_install(
        "xformers==0.0.28",
        "huggingface_hub>=0.23.0",
        "requests>=2.32.0",
        "uvicorn",
        "fastapi",
        "pillow",
        "einops",
        "safetensors",
    )
    # Clone ComfyUI + some helpful nodes
    .run_commands(
        # workspace
        "mkdir -p /opt && cd /opt",
        # ComfyUI core
        "git clone --depth=1 https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI || true",
        # Common nodes (best-effort; don't fail the build if repo changes name/moves)
        # Qwen nodes (names may change; non-fatal if they fail)
        "git clone --depth=1 https://github.com/Comfy-Org/ComfyUI_Qwen.git /opt/ComfyUI/custom_nodes/ComfyUI_Qwen || true",
        # Miaoshouai Tagger (used in the provided workflow JSON)
        "git clone --depth=1 https://github.com/cubiq/ComfyUI-Miaoshouai-Tagger.git /opt/ComfyUI/custom_nodes/ComfyUI-Miaoshouai-Tagger || true",
        # Impact Pack is common for samplers/utilities in many graphs
        "git clone --depth=1 https://github.com/ltdrdata/ComfyUI-Impact-Pack.git /opt/ComfyUI/custom_nodes/ComfyUI-Impact-Pack || true",
        # RealESRGAN upscaler is often used
        "git clone --depth=1 https://github.com/sczhou/CodeFormer.git /opt/CodeFormer || true",
        # Install python deps for ComfyUI (safe to run multiple times)
        "pip install --no-cache-dir -r /opt/ComfyUI/requirements.txt || true",
    )
)

# ---------- Helper: ensure models present ----------
def _download_cmd(url: str, dest: str) -> str:
    # Attempt download with wget; retry a bit; support HF redirects
    return (
        f'mkdir -p "$(dirname \\"{dest}\\")" && '
        f'echo "Downloading: {url} -> {dest}" && '
        f'curl -L --retry 5 --retry-all-errors -o "{dest}" "{url}" || '
        f'wget --tries=5 -O "{dest}" "{url}"'
    )

QWEN_ASSETS = [
    # diffusion model for image edit
    {
        "url": "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_fp8_e4m3fn.safetensors",
        "path": "/models/diffusion_models/qwen_image_edit_fp8_e4m3fn.safetensors",
    },
    # Lightning 4-steps (often used as unet/diffusion model variant)
    {
        "url": "https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-4steps-V1.0.safetensors",
        "path": "/models/loras/Qwen-Image-Lightning-4steps-V1.0.safetensors",
    },
    # text encoder
    {
        "url": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "path": "/models/clip/qwen_2.5_vl_7b_fp8_scaled.safetensors",
    },
    # VAE
    {
        "url": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors",
        "path": "/models/vae/qwen_image_vae.safetensors",
    },
]

# Lenovo Qwen LoRA / checkpoint (user-specified repo + filename)
LENOVO_REPO = "Danrisi/Lenovo_Qwen"
LENOVO_FILENAME = "lenovo.safetensors"
LENOVO_DEST = "/models/loras/lenovo.safetensors"

# ---------- Class ----------
@app.cls(
    gpu=modal.gpu.L40S(count=1),
    image=image,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
    volumes={"/models": MODELS_VOL, "/outputs": OUTPUTS_VOL},
    timeout=60 * 30,  # 30 minutes per call max
)
class ComfyUIRunner:
    def _spawn_server(self) -> None:
        """Boot ComfyUI in a background thread."""
        def _run():
            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            # Perf-friendly defaults for L40S / Inductor
            env.setdefault("TORCH_LOGS", "error")
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
            env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
            # Prefer /models for model dirs
            env["COMFYUI_MODEL_DIR"] = "/models"
            os.makedirs("/outputs", exist_ok=True)
            # Start ComfyUI headless; bind localhost
            cmd = [
                "python", "/opt/ComfyUI/main.py",
                "--disable-auto-launch",
                "--listen", "127.0.0.1",
                "--port", "8188",
                "--output-directory", "/outputs",
            ]
            proc = subprocess.Popen(cmd, env=env)
            proc.wait()

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    @modal.enter()
    def setup(self):
        # Ensure required folders
        for p in [
            "/models/checkpoints",
            "/models/loras",
            "/models/vae",
            "/models/clip",
            "/models/diffusion_models",
            "/outputs",
        ]:
            os.makedirs(p, exist_ok=True)

        # Best-effort: download required Qwen assets
        for item in QWEN_ASSETS:
            dest = item["path"]
            if not os.path.exists(dest):
                os.system(_download_cmd(item["url"], dest))

        # Best-effort: Lenovo_Qwen file via huggingface_hub (supports auth if needed)
        try:
            from huggingface_hub import hf_hub_download
            if not os.path.exists(LENOVO_DEST):
                tmp_path = hf_hub_download(
                    repo_id=LENOVO_REPO,
                    filename=LENOVO_FILENAME,
                    token=os.environ.get("HF_TOKEN") or None,
                    local_dir="/models/loras",
                    local_dir_use_symlinks=False,
                )
                # Ensure final name at expected path
                if tmp_path != LENOVO_DEST and os.path.exists(tmp_path):
                    os.rename(tmp_path, LENOVO_DEST)
        except Exception as e:
            print(f"[WARN] Could not fetch {LENOVO_REPO}:{LENOVO_FILENAME} — {e}")

        # Start the server
        self._spawn_server()

        # Wait for readiness
        import requests
        for _ in range(120):
            try:
                r = requests.get("http://127.0.0.1:8188/system_stats", timeout=1.5)
                if r.ok:
                    print("[ComfyUI] Ready.")
                    return
            except Exception:
                pass
            time.sleep(1.0)
        raise RuntimeError("ComfyUI did not become ready in time.")

    @modal.method()
    def run_workflow(self, workflow_file: str, save_prefix: str = "ComfyUI") -> Dict[str, Any]:
        """Execute a given workflow JSON and return result info.

        Args:
            workflow_file: Path accessible in the container/volume, e.g., "/mnt/data/…json"
            save_prefix: Filename prefix for SaveImage nodes

        Returns:
            Dict with prompt_id and list of saved image files found under /outputs
        """
        import requests
        # Load workflow JSON
        with open(workflow_file, "r", encoding="utf-8") as f:
            workflow = json.load(f)

        # Optionally enforce a filename_prefix for SaveImage nodes
        try:
            for node_id, node in workflow.items():
                if isinstance(node, dict) and node.get("class_type") == "SaveImage":
                    w = node.setdefault("inputs", {})
                    w["filename_prefix"] = save_prefix
        except Exception as e:
            print(f"[WARN] Could not set filename_prefix: {e}")

        # Submit prompt
        payload = {"prompt": workflow}
        resp = requests.post("http://127.0.0.1:8188/prompt", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        prompt_id = data.get("prompt_id")

        # Poll for outputs
        outputs: List[str] = []
        start = time.time()
        while time.time() - start < 900:  # up to 15 mins
            time.sleep(2)
            # scan outputs dir
            for p in sorted(Path("/outputs").glob(f"{save_prefix}*.png")):
                s = str(p)
                if s not in outputs:
                    outputs.append(s)
            # Heuristic: at least one image produced & no change for 6s
            if outputs:
                time.sleep(6)
                more = sorted([str(p) for p in Path("/outputs").glob(f"{save_prefix}*.png")])
                if len(more) == len(outputs):
                    break
                else:
                    outputs = more

        return {"prompt_id": prompt_id, "images": outputs}

# ---------- Entrypoint for local dev ----------
@app.function(gpu=modal.gpu.L40S(count=1), image=image, volumes={"/models": MODELS_VOL, "/outputs": OUTPUTS_VOL})
def main(workflow_file: str = "/mnt/data/1shotDatasetQWEN ICekiub FREE.json"):
    """Convenience runner for `modal run` during development."""
    runner = ComfyUIRunner()
    return runner.run_workflow.remote(workflow_file)
