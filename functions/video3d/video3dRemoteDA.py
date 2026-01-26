import os
import json
import time
import shutil
import tempfile
import subprocess
from pathlib import Path
from .convert_mov_mp4 import ensure_mp4


def run_kaggle_video_to_3d(
    video_input,
    output_dir: str,
    username: str,
    api_key: str,
    frame_interval: int = 10,
):
    """
    Runs Pi3 Video-to-3D pipeline on Kaggle GPU using OFFICIAL Pi3 inference.
    
    Args:
        video_input: Can be either:
            - str: Path to video file
            - bytes: Video file bytes
        output_dir: Output directory for the 3D model
        username: Kaggle username
        api_key: Kaggle API key
        frame_interval: Frame sampling interval (default: 10)
    
    Returns:
        str: Path to the generated PLY file
    """
    
    # ---------------------------------------------------------
    # Handle different input types
    # ---------------------------------------------------------
    temp_video_path = None
    try:
        if isinstance(video_input, str):
            # Already a path
            video_path = Path(video_input).resolve()
        elif isinstance(video_input, bytes):
            # Save bytes to temp file
            temp_video_path = Path(tempfile.mktemp(suffix='.mp4'))
            temp_video_path.write_bytes(video_input)
            video_path = temp_video_path
        else:
            raise ValueError(f"Unsupported video_input type: {type(video_input)}")
        
        # ---------------------------------------------------------
        # Resolve and validate paths
        # ---------------------------------------------------------
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Ensure video is MP4 format
        video_path = Path(ensure_mp4(str(video_path))).resolve()
        
        print("=" * 70)
        print("PI3 VIDEO → 3D (KAGGLE)")
        print("=" * 70)
        print(f"Input video     : {video_path}")
        print(f"Output dir      : {output_dir}")
        print(f"Frame interval  : {frame_interval}")
        
        # ---------------------------------------------------------
        # Kaggle setup
        # ---------------------------------------------------------
        subprocess.run(["kaggle", "--version"], check=True)
        
        print("Using Kaggle credentials from Frontend App...")
        
        # Set up Kaggle credentials
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        kaggle_json = kaggle_dir / "kaggle.json"
        kaggle_json.write_text(
            json.dumps({"username": username, "key": api_key}),
            encoding="utf-8"
        )
        os.chmod(kaggle_json, 0o600)
        
        # ---------------------------------------------------------
        # Create Kaggle dataset
        # ---------------------------------------------------------
        dataset_slug = f"pi3-video-{int(time.time())}"
        dataset_dir = Path(tempfile.mkdtemp(prefix="kaggle_dataset_"))
        
        try:
            # Copy video to dataset directory
            shutil.copy2(video_path, dataset_dir / video_path.name)
            
            # Create dataset metadata
            with open(dataset_dir / "dataset-metadata.json", "w") as f:
                json.dump(
                    {
                        "title": dataset_slug,
                        "id": f"{username}/{dataset_slug}",
                        "licenses": [{"name": "CC0-1.0"}],
                    },
                    f,
                    indent=2,
                )
            
            print("Uploading dataset to Kaggle...")
            result = subprocess.run(
                ["kaggle", "datasets", "create", "-p", str(dataset_dir)],
                capture_output=True,
                text=True,
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Dataset upload failed: {result.stderr or result.stdout}")
            
            print(f"Dataset uploaded: {username}/{dataset_slug}")
            
            # ---------------------------------------------------------
            # Wait for dataset to be ready
            # ---------------------------------------------------------
            print("Waiting for dataset to become available...")
            max_wait_time = 300  # 5 minutes timeout
            start_time = time.time()
            
            while True:
                if time.time() - start_time > max_wait_time:
                    raise RuntimeError("Dataset did not become ready within timeout period")
                
                time.sleep(5)
                check = subprocess.run(
                    ["kaggle", "datasets", "status", f"{username}/{dataset_slug}"],
                    capture_output=True,
                    text=True,
                )
                
                if "ready" in check.stdout.lower():
                    print("Dataset is ready ✔")
                    break
            
            # ---------------------------------------------------------
            # Create kernel
            # ---------------------------------------------------------
            kernel_dir = Path(tempfile.mkdtemp(prefix="kaggle_pi3_kernel_"))
            kernel_slug = f"pi3-video-to-3d-{int(time.time())}"
            unique_output_name = f"result_{int(time.time() * 1000)}.ply"
            
            print("Creating Kaggle kernel...")
            print("Kernel slug:", kernel_slug)
            print("Unique output name:", unique_output_name)
            
            try:
                # ---------------------------------------------------------
                # Kaggle kernel script
                # ---------------------------------------------------------
                script_code = f"""
import subprocess
from pathlib import Path
import torch
import warnings

warnings.filterwarnings("ignore")

print("=" * 70)
print("PI3 OFFICIAL INFERENCE (KAGGLE)")
print("=" * 70)

# ---------------------------------------------------------
# Install dependencies
# ---------------------------------------------------------
print("Installing PyTorch...")
subprocess.run(
    [
        "pip", "install", "--no-cache-dir",
        "torch", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/cu118"
    ],
    check=True
)

print("Installing Pi3...")
subprocess.run(
    ["pip", "install", "--no-cache-dir", "git+https://github.com/yyfz/Pi3.git"],
    check=True
)

print("Installing additional dependencies...")
subprocess.run(
    ["pip", "install", "--no-cache-dir", "plyfile", "safetensors"],
    check=True
)

# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------
from pi3.utils.basic import load_images_as_tensor, write_ply
from pi3.utils.geometry import depth_edge
from pi3.models.pi3 import Pi3

# ---------------------------------------------------------
# Input / Output
# ---------------------------------------------------------
VIDEO_PATH = next(Path("/kaggle/input").rglob("*.mp4"))
OUTPUT_DIR = Path("/kaggle/working/output")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_PLY = OUTPUT_DIR / "{unique_output_name}"

print("Video:", VIDEO_PATH)
print("Output:", OUTPUT_PLY)

# ---------------------------------------------------------
# Device
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
print("Loading Pi3 model...")
model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()

# ---------------------------------------------------------
# Load frames
# ---------------------------------------------------------
print("Loading video frames...")
imgs = load_images_as_tensor(
    str(VIDEO_PATH),
    interval={frame_interval}
).to(device)
print(f"Loaded {{imgs.shape[0]}} frames")

# ---------------------------------------------------------
# Inference
# ---------------------------------------------------------
print("Running Pi3 inference...")
dtype = (
    torch.bfloat16
    if torch.cuda.get_device_capability()[0] >= 8
    else torch.float16
)

with torch.no_grad():
    with torch.amp.autocast("cuda", dtype=dtype):
        res = model(imgs[None])

# ---------------------------------------------------------
# Mask filtering
# ---------------------------------------------------------
print("Applying masks and filtering...")
masks = torch.sigmoid(res["conf"][..., 0]) > 0.1
non_edge = ~depth_edge(res["local_points"][..., 2], rtol=0.03)
masks = torch.logical_and(masks, non_edge)[0]

# ---------------------------------------------------------
# Save PLY
# ---------------------------------------------------------
print("Saving PLY file...")
write_ply(
    res["points"][0][masks].cpu(),
    imgs.permute(0, 2, 3, 1)[masks],
    OUTPUT_PLY
)

print("✅ Saved:", OUTPUT_PLY)
"""
                
                # Write kernel files
                (kernel_dir / "kernel.py").write_text(script_code, encoding="utf-8")
                
                # Create kernel metadata
                with open(kernel_dir / "kernel-metadata.json", "w") as f:
                    json.dump(
                        {
                            "id": f"{username}/{kernel_slug}",
                            "title": kernel_slug,
                            "code_file": "kernel.py",
                            "language": "python",
                            "kernel_type": "script",
                            "is_private": "true",
                            "enable_gpu": "true",
                            "enable_internet": "true",
                            "dataset_sources": [f"{username}/{dataset_slug}"],
                        },
                        f,
                        indent=2,
                    )
                
                # ---------------------------------------------------------
                # Check and delete existing kernel if running
                # ---------------------------------------------------------
                try:
                    print("Checking for existing kernel...")
                    status = subprocess.run(
                        ["kaggle", "kernels", "status", f"{username}/{kernel_slug}"],
                        capture_output=True,
                        text=True
                    ).stdout.lower()
                    
                    if "running" in status or "queued" in status:
                        print("Deleting existing kernel...")
                        subprocess.run(
                            ["kaggle", "kernels", "delete", f"{username}/{kernel_slug}"],
                            check=True
                        )
                        print("Existing kernel deleted successfully.")
                except subprocess.CalledProcessError:
                    print("No existing kernel found.")
                
                # ---------------------------------------------------------
                # Push kernel
                # ---------------------------------------------------------
                print("Pushing Kaggle kernel...")
                subprocess.run(
                    ["kaggle", "kernels", "push", "-p", str(kernel_dir)],
                    check=True,
                )
                
                kernel_ref = f"{username}/{kernel_slug}"
                print("Kernel ref:", kernel_ref)
                print(f"Monitor at: https://www.kaggle.com/code/{kernel_ref}")
                
                # ---------------------------------------------------------
                # Wait for completion
                # ---------------------------------------------------------
                print("Waiting for kernel to complete...")
                max_kernel_wait = 1800  # 30 minutes timeout
                kernel_start_time = time.time()
                
                while True:
                    if time.time() - kernel_start_time > max_kernel_wait:
                        raise RuntimeError("Kernel did not complete within timeout period")
                    
                    time.sleep(15)
                    status = subprocess.run(
                        ["kaggle", "kernels", "status", kernel_ref],
                        capture_output=True,
                        text=True,
                    ).stdout.lower()
                    
                    if "complete" in status:
                        print("✅ Kernel completed successfully")
                        break
                    if "error" in status or "failed" in status:
                        raise RuntimeError(
                            f"Kernel failed. Check logs at: https://www.kaggle.com/code/{kernel_ref}"
                        )
                
                # ---------------------------------------------------------
                # Download output
                # ---------------------------------------------------------
                print("Downloading kernel output...")
                
                result = subprocess.run(
                    ["kaggle", "kernels", "output", kernel_ref, "-p", str(output_dir)],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="ignore",
                )
                
                if result.returncode != 0:
                    print("⚠ Kaggle CLI returned non-zero exit (may be harmless)")
                    print(result.stderr.strip() if result.stderr else "")
                
                # Find the PLY file
                expected_ply_file = None
                
                # Try multiple potential locations
                potential_paths = [
                    output_dir / "output" / unique_output_name,
                    output_dir / unique_output_name,
                ]
                
                for path in potential_paths:
                    if path.exists():
                        expected_ply_file = path
                        break
                
                # If not found by unique name, find most recent PLY
                if expected_ply_file is None:
                    ply_files = list(output_dir.rglob("*.ply"))
                    if not ply_files:
                        raise FileNotFoundError(
                            f"PLY output not found in {output_dir}. "
                            f"Check kernel logs at: https://www.kaggle.com/code/{kernel_ref}"
                        )
                    
                    expected_ply_file = max(ply_files, key=lambda p: p.stat().st_mtime)
                    print(f"⚠ Using most recent PLY: {expected_ply_file}")
                
                if not expected_ply_file.exists():
                    raise FileNotFoundError(f"Expected PLY file not found: {expected_ply_file}")
                
                print("=" * 70)
                print("✅ SUCCESS - Final 3D model:", expected_ply_file)
                print("=" * 70)
                
                return str(expected_ply_file)
            
            finally:
                # Clean up kernel directory
                shutil.rmtree(kernel_dir, ignore_errors=True)
        
        finally:
            # Clean up dataset directory
            shutil.rmtree(dataset_dir, ignore_errors=True)
            
            # Optional: Delete the dataset from Kaggle after processing
            # Uncomment if you want to clean up datasets automatically
            try:
                subprocess.run(
                    ["kaggle", "datasets", "delete", f"{username}/{dataset_slug}"],
                    check=True
                )
                print(f"Deleted dataset: {username}/{dataset_slug}")
            except:
                print(f"Could not delete dataset: {username}/{dataset_slug}")
    
    finally:
        # Clean up temporary video file if we created one
        if temp_video_path and temp_video_path.exists():
            temp_video_path.unlink()