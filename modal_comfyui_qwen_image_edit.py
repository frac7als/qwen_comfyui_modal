import os
import subprocess
import modal

# ------------------------------------------------------------
# Modal app: ComfyUI (Qwen Image Edit stack)
# - Minimal, clean setup focused on Qwen-Image edit/inference
# - Uses a persistent HF cache volume to avoid re-downloads
# ------------------------------------------------------------

# Base image with essentials
image = (
    modal.Image
    .debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "ffmpeg",
        "build-essential",
        "cmake",
        "wget",
    )
    .pip_install(
        # Core runtime utilities
        "imageio[ffmpeg]",
        "moviepy",
        # Web server used by ComfyUI launcher
        "fastapi[standard]==0.115.4",
        # Comfy CLI pinned for reproducibility
        "comfy-cli==1.5.1",
    )
    # Install ComfyUI framework with NVIDIA support
    .run_commands(
        "comfy --skip-prompt install --fast-deps --nvidia --version 0.3.47"
    )
)

# ------------------------------
# Custom Nodes (git-cloned; idempotent)
# ------------------------------
image = image.run_commands(
    "mkdir -p /root/comfy/ComfyUI/custom_nodes",
    # rgthree-comfy
    "bash -lc 'cd /root/comfy/ComfyUI/custom_nodes && if [ ! -d rgthree-comfy ]; then git clone https://github.com/rgthree/rgthree-comfy.git; fi'",
    # efficiency-nodes-comfyui (jags111 fork)
    "bash -lc 'cd /root/comfy/ComfyUI/custom_nodes && if [ ! -d efficiency-nodes-comfyui ]; then git clone https://github.com/jags111/efficiency-nodes-comfyui.git; fi'",
    # CG Use Everywhere
    "bash -lc 'cd /root/comfy/ComfyUI/custom_nodes && if [ ! -d cg-use-everywhere ]; then git clone https://github.com/chrisgoringe/cg-use-everywhere.git; fi'",
    # ComfyUI-MultiGPU
    "bash -lc 'cd /root/comfy/ComfyUI/custom_nodes && if [ ! -d ComfyUI-MultiGPU ]; then git clone https://github.com/pollockjj/ComfyUI-MultiGPU.git; fi'",
    # ComfyUI-Miaoshouai-Tagger
    "bash -lc 'cd /root/comfy/ComfyUI/custom_nodes && if [ ! -d ComfyUI-Miaoshouai-Tagger ]; then git clone https://github.com/miaoshouai/ComfyUI-Miaoshouai-Tagger.git; fi'",
    # ComfyLiterals
    "bash -lc 'cd /root/comfy/ComfyUI/custom_nodes && if [ ! -d ComfyLiterals ]; then git clone https://github.com/M1kep/ComfyLiterals.git; fi'",
    # KJNodes (adds Set/Get utilities e.g., Set_FACE)
    "bash -lc 'cd /root/comfy/ComfyUI/custom_nodes && if [ ! -d ComfyUI-KJNodes ]; then git clone https://github.com/kijai/ComfyUI-KJNodes.git; fi'",
)

# ------------------------------
# Model downloads from Hugging Face
# ------------------------------

def hf_download():
    from huggingface_hub import hf_hub_download

    # Ensure model directories exist
    diffusion_dir = "/root/comfy/ComfyUI/models/diffusion_models"
    vae_dir = "/root/comfy/ComfyUI/models/vae"
    text_enc_dir = "/root/comfy/ComfyUI/models/text_encoders"
    lora_dir = "/root/comfy/ComfyUI/models/loras"
    os.makedirs(diffusion_dir, exist_ok=True)
    os.makedirs(vae_dir, exist_ok=True)
    os.makedirs(text_enc_dir, exist_ok=True)
    os.makedirs(lora_dir, exist_ok=True)

    # Diffusion model: Qwen Image Edit (FP8 e4m3fn)
    qwen_edit = hf_hub_download(
        repo_id="Comfy-Org/Qwen-Image-Edit_ComfyUI",
        filename="split_files/diffusion_models/qwen_image_edit_fp8_e4m3fn.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -sf {qwen_edit} {os.path.join(diffusion_dir, 'qwen_image_edit_fp8_e4m3fn.safetensors')}",
        shell=True,
        check=True,
    )

    # Text encoder: Qwen VL 7B FP8 scaled
    qwen_text = hf_hub_download(
        repo_id="Comfy-Org/Qwen-Image_ComfyUI",
        filename="split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -sf {qwen_text} {os.path.join(text_enc_dir, 'qwen_2.5_vl_7b_fp8_scaled.safetensors')}",
        shell=True,
        check=True,
    )

    # VAE: Qwen Image VAE
    qwen_vae = hf_hub_download(
        repo_id="Comfy-Org/Qwen-Image_ComfyUI",
        filename="split_files/vae/qwen_image_vae.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -sf {qwen_vae} {os.path.join(vae_dir, 'qwen_image_vae.safetensors')}",
        shell=True,
        check=True,
    )

    # LORAs: Qwen Image Lightning (4 steps) + Lenovo LoRA
    qwen_lightning = hf_hub_download(
        repo_id="lightx2v/Qwen-Image-Lightning",
        filename="Qwen-Image-Lightning-4steps-V1.0.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -sf {qwen_lightning} {os.path.join(lora_dir, 'Qwen-Image-Lightning-4steps-V1.0.safetensors')}",
        shell=True,
        check=True,
    )

    lenovo_lora = hf_hub_download(
        repo_id="Danrisi/Lenovo_Qwen",
        filename="lenovo.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -sf {lenovo_lora} {os.path.join(lora_dir, 'lenovo.safetensors')}",
        shell=True,
        check=True,
    )


# Persist HF cache between runs
vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

# Add HF client (with hf_transfer) and pre-populate cache inside the image build
image = (
    image
    .pip_install("huggingface_hub[hf_transfer]>=0.34.0,<1.0")
    .run_function(
        hf_download,
        volumes={"/cache": vol},
    )
)

# ------------------------------
# Modal App & Web Server
# ------------------------------
app = modal.App(name="comfyui-qwen-image-edit", image=image)


@app.function(
    max_containers=1,
    gpu="L40S",
    volumes={"/cache": vol},
)
@modal.concurrent(max_inputs=10)
@modal.web_server(8000, startup_timeout=60)
def ui():
    # Launch ComfyUI listening on 0.0.0.0:8000
    subprocess.Popen(
        "comfy launch -- --listen 0.0.0.0 --port 8000",
        shell=True,
    )
