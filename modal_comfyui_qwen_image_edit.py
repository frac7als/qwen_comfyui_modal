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
        # install comfy
        "comfy --skip-prompt install --fast-deps --nvidia --version 0.3.59",
        # add KJNodes here
        "git clone https://github.com/kijai/ComfyUI-KJNodes.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-KJNodes",
    )
)

# ------------------------------
# Custom Nodes (git-cloned; idempotent) with dependency installation
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
    
    # Install dependencies for custom nodes
    "bash -lc 'cd /root/comfy/ComfyUI/custom_nodes && for dir in */; do if [ -f \"$dir/requirements.txt\" ]; then echo \"Installing requirements for $dir\"; pip install -r \"$dir/requirements.txt\" || echo \"Failed to install requirements for $dir\"; fi; done'",
    
    # Install common dependencies that custom nodes often need
    "pip install opencv-python-headless pillow numpy scipy torch torchvision torchaudio transformers diffusers accelerate",
    
    # Specific dependencies for some of your custom nodes
    "pip install segment-anything timm ultralytics",
    
    # For efficiency nodes
    "pip install simpleeval numexpr",
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
    upscale_dir = "/root/comfy/ComfyUI/models/upscale_models"
    os.makedirs(diffusion_dir, exist_ok=True)
    os.makedirs(vae_dir, exist_ok=True)
    os.makedirs(text_enc_dir, exist_ok=True)
    os.makedirs(lora_dir, exist_ok=True)
    os.makedirs(upscale_dir, exist_ok=True)

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

    # ESRGAN Upscaler: ITF SkinDiff Detail Lite v1
    esrgan_model = hf_hub_download(
        repo_id="uwg/upscaler",
        filename="ESRGAN/1x-ITF-SkinDiffDetail-Lite-v1.pth",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -sf {esrgan_model} {os.path.join(upscale_dir, '1x-ITF-SkinDiffDetail-Lite-v1.pth')}",
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
    timeout=300,  # Increased timeout for startup
)
@modal.concurrent(max_inputs=10)
@modal.web_server(8000, startup_timeout=120)  # Increased startup timeout
def ui():
    # Launch ComfyUI listening on 0.0.0.0:8000
    # Add verbose flag to see what's happening during startup
    subprocess.Popen(
        "comfy launch -- --listen 0.0.0.0 --port 8000 --verbose",
        shell=True,
    )

# ------------------------------------------------------------
# Debug helpers
# ------------------------------------------------------------
@app.function(volumes={"/cache": vol})
def list_custom_nodes():
    import os, json
    listing = os.listdir("/root/comfy/ComfyUI/custom_nodes")
    print("custom_nodes:", json.dumps(listing))
    return listing

@app.function(volumes={"/cache": vol})
def debug_node_loading():
    """Debug function to check if custom nodes are loading properly"""
    import os
    import sys
    
    # Check if custom nodes directory exists and list contents
    custom_nodes_path = "/root/comfy/ComfyUI/custom_nodes"
    if os.path.exists(custom_nodes_path):
        print(f"Custom nodes directory exists at: {custom_nodes_path}")
        for node_dir in os.listdir(custom_nodes_path):
            node_path = os.path.join(custom_nodes_path, node_dir)
            if os.path.isdir(node_path):
                print(f"Node directory: {node_dir}")
                # Check for __init__.py or main Python files
                python_files = [f for f in os.listdir(node_path) if f.endswith('.py')]
                print(f"  Python files: {python_files}")
                
                # Check for requirements.txt
                req_file = os.path.join(node_path, 'requirements.txt')
                if os.path.exists(req_file):
                    print(f"  Has requirements.txt")
                else:
                    print(f"  No requirements.txt found")
    
    # Try to import ComfyUI and see what happens
    try:
        sys.path.insert(0, "/root/comfy/ComfyUI")
        import execution
        print("ComfyUI execution module imported successfully")
    except Exception as e:
        print(f"Error importing ComfyUI: {e}")
    
    return "Debug complete"

# Alternative minimal startup function for testing
@app.function(
    max_containers=1,
    gpu="L40S", 
    volumes={"/cache": vol},
)
def test_comfy_startup():
    """Test ComfyUI startup without web server to see error messages"""
    import subprocess
    
    print("Testing ComfyUI startup...")
    
    # Try to start ComfyUI and capture output
    result = subprocess.run(
        "cd /root/comfy && python ComfyUI/main.py --cpu --listen 127.0.0.1 --port 8188",
        shell=True,
        capture_output=True,
        text=True,
        timeout=60
    )
    
    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")  
    print(result.stderr)
    print(f"Return code: {result.returncode}")
    
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode
    }
