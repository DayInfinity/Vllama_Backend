import os, time
import json
import subprocess
import tempfile
import shutil


def run_video_kaggle(model_name, prompt, output_dir):
    credentials_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(credentials_path) and not (
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    ):
        print("Error: Kaggle API credentials not found. Please set up your kaggle API token in ~/.kaggle/kaggle.json.")
        return 
    
    try:
        subprocess.run(["kaggle", "--version"], stdout=subprocess.DEVNULL, check=True)
    except FileNotFoundError:
        print("Error: Kaggle CLI is not installed. Please install it with 'pip install kaggle'.")
        return

    username = None
    if os.path.exists(credentials_path):
        try:
            with open(credentials_path, "r") as f:
                creds = json.load(f)
                username = creds.get("username")
        except Exception:
            username = None

    if not username:
        username = os.environ.get("KAGGLE_USERNAME")
    if not username:
        print("Error: Could not determine Kaggle username from credentials.")
        return

    # 2. Prepare a temporary directory with the kernel script and metadata
    kernel_dir = tempfile.mkdtemp(prefix="vllama_kaggle_")
    try:
        # Write the Kaggle kernel script that installs dependencies and runs the model
        script_path = os.path.join(kernel_dir, "vllama_kernel.py")
        model_str = json.dumps(model_name)    # safely quote the model string
        prompt_str = json.dumps(prompt)     # safely quote the prompt string
        script_code = f"""
import subprocess
# Install required packages inside Kaggle (quietly, without cache to speed up start)
subprocess.run(
    ["pip", "uninstall", "-y", "jax", "jaxlib", "flax"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
subprocess.run(['pip', 'install', '--no-cache-dir', 'diffusers[torch]==0.20.2',
                'transformers==4.33.0', 'accelerate==0.22.0', 'xformers==0.0.20','protobuf==3.20.3', 'huggingface-hub==0.25.2' , '--quiet'])
from diffusers import StableDiffusionPipeline
import torch
import numpy as np
import imageio
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
pipe = DiffusionPipeline.from_pretrained({model_str}, torch_dtype = torch.float16, variant = "fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.to('cuda')
prompt = {prompt_str}
result = pipe(prompt, num_inference_steps = 50)
frames = result.frames  # this should be a list/array of individual frames

print("Number of frames:", len(frames))
print("Single frame shape:", np.array(frames[0]).shape)

def export_to_video(frames, output_path="output.mp4", fps=8):
    out = []
    for frame in frames:
        f = np.array(frame)
        # f should now be (H, W, 3)
        if f.dtype != np.uint8:
            f = (255 * np.clip(f, 0, 1)).astype(np.uint8)
        out.append(f)

    imageio.mimsave(
        output_path,
        out,
        fps=fps,
        quality=8,
        macro_block_size=1,  # avoid the resizing warning
    )
    return output_path


video_path = export_to_video(frames, "result.mp4")
print("Video saved at:", video_path)
"""
        with open(script_path, 'w') as f:
            f.write(script_code.strip() + "\n")

        # Write kernel-metadata.json for Kaggle
        kernel_slug = "vllama-" + model_name.replace('/', '-')
        # Sanitize slug to meet Kaggle requirements (alphanumeric and hyphens)
        kernel_slug = "".join(ch if ch.isalnum() or ch == '-' else '-' for ch in kernel_slug.lower())
        if len(kernel_slug) > 50:  # slug length limit (if any)
            kernel_slug = kernel_slug[:50]
        title = "vllama " + model_name.replace('/', ' ')
        metadata = {
            "id": f"{username}/{kernel_slug}",
            "title": title,
            "code_file": os.path.basename(script_path),
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": []
        }
        meta_path = os.path.join(kernel_dir, "kernel-metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # 3. Push the kernel to Kaggle and trigger execution
        print(f"Pushing Kaggle kernel (model: {model_name})...")
        result = subprocess.run(["kaggle", "kernels", "push", "-p", kernel_dir], capture_output=True, text=True)
        if result.returncode != 0:
            print("Failed to push Kaggle kernel. Error output:")
            print(result.stderr or result.stdout)
            return
        print("Kernel pushed successfully. Kaggle is running the kernel...")

        # 4. Poll Kaggle for kernel status until it finishes
        kernel_ref = f"{username}/{kernel_slug}"
        start_time = time.time()
        timestamp = int(time.time())
        while True:
            time.sleep(5)  # wait 5 seconds between status checks
            status_res = subprocess.run(["kaggle", "kernels", "status", kernel_ref], capture_output=True, text=True)
            status_text = (status_res.stdout or "") + (status_res.stderr or "")
            status_lower = status_text.lower()
            if "complete" in status_lower:
                print("Kaggle kernel execution completed.")
                break
            if "error" in status_lower or "failed" in status_lower:
                print("Kaggle kernel execution failed. Please check the Kaggle notebook for errors.")
                return
            if time.time() - start_time > 900:  # timeout after 15 minutes
                print("Timed out waiting for Kaggle kernel to complete.")
                return

        # 5. Download the generated video from Kaggle
        os.makedirs(output_dir, exist_ok=True)
        print(f"Downloading output to {output_dir}...")
        out_res = subprocess.run(["kaggle", "kernels", "output", kernel_ref, "-p", output_dir],
        capture_output=True, text=True, errors='ignore')
        
        # Check if the file was actually downloaded (regardless of return code)
        source_path = os.path.join(output_dir, "result.mp4")
        final_video_path = os.path.join(output_dir, f"vllama_video_output_{timestamp}.mp4")
        
        if os.path.exists(source_path):
            os.rename(source_path, final_video_path)
            print(f"Video successfully downloaded and saved to {final_video_path}")
        else:
            print("Error: Video file not found after download.")
            if out_res.returncode != 0:
                print("Kaggle output error:")
                print(out_res.stderr or out_res.stdout)
    finally:
        # Clean up the temporary kernel files
        shutil.rmtree(kernel_dir, ignore_errors=True)
    
    print("Returning Final video path:", final_video_path)
    return final_video_path