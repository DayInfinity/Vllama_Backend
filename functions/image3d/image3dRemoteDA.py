import os
import json
import time
import shutil
import tempfile
import subprocess
import base64
from pathlib import Path
from io import BytesIO
from PIL import Image

def run_kaggle_image_to_3d(image_input, output_dir: str, username: str, api_key: str):
    """
    Runs Apple's SHARP Image-to-3D Gaussian pipeline on Kaggle GPU
    
    Args:
        image_input: Can be either:
            - str: Path to image file
            - bytes: Image file bytes
            - BytesIO: Image file-like object
        output_dir: Output directory for the 3D model
        username: Kaggle username
        api_key: Kaggle API key
    """
    # ---------------------------------------------------------
    # Handle different input types
    # ---------------------------------------------------------
    temp_image_path = None
    try:
        if isinstance(image_input, str):
            # Already a path
            image_path = Path(image_input).resolve()
        elif isinstance(image_input, bytes):
            # Save bytes to temp file
            temp_image_path = Path(tempfile.mktemp(suffix='.jpg'))
            temp_image_path.write_bytes(image_input)
            image_path = temp_image_path
        elif isinstance(image_input, BytesIO):
            # Save BytesIO to temp file
            temp_image_path = Path(tempfile.mktemp(suffix='.jpg'))
            temp_image_path.write_bytes(image_input.read())
            image_path = temp_image_path
            image_input.seek(0)  # Reset stream
        else:
            raise ValueError(f"Unsupported image_input type: {type(image_input)}")
        
        # ---------------------------------------------------------
        # Resolve and validate paths
        # ---------------------------------------------------------
        output_dir = Path(output_dir).resolve()
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        print("Input image:", image_path)
        print("Output directory:", output_dir)
        
        # ---------------------------------------------------------
        # Encode image as Base64
        # ---------------------------------------------------------
        print("Encoding image to Base64...")
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        
        # ---------------------------------------------------------
        # Kaggle setup
        # ---------------------------------------------------------
        subprocess.run(["kaggle", "--version"], check=True)

        print("Using the kaggle credentials sent from the Frontend App.... ")
        
        # Set up Kaggle credentials
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        kaggle_json = kaggle_dir / "kaggle.json"
        kaggle_json.write_text(json.dumps({"username": username, "key": api_key}), encoding="utf-8")
        os.chmod(kaggle_json, 0o600)
        
        # ---------------------------------------------------------
        # Create temporary kernel directory
        # ---------------------------------------------------------
        kernel_dir = Path(tempfile.mkdtemp(prefix="kaggle_3dgs_"))
        kernel_slug = f"image-to-3d-{int(time.time())}"
        # Generate unique output filename using kernel slug to ensure uniqueness
        unique_output_name = f"output_{int(time.time() * 1000)}.ply"
        print("Created temp kernel dir:", kernel_dir)
        print("Kernel slug:", kernel_slug)
        print("Unique output name:", unique_output_name)
        
        try:
            # -----------------------------------------------------
            # Kaggle kernel script (FLOAT-SAFE & STABLE)
            # -----------------------------------------------------
            script_code = f"""
import base64
import subprocess
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

print("Installing SHARP...")
subprocess.run(
    ["pip", "install", "--no-cache-dir", "git+https://github.com/apple/ml-sharp.git"],
    check=True
)

import torch
import torch.nn.functional as F
from sharp.models import PredictorParams, create_predictor
from sharp.utils import io
from sharp.utils.gaussians import save_ply, unproject_gaussians

# -------------------------------------------------
# Device
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------------------------
# Restore image
# -------------------------------------------------
IMAGE_PATH = Path("/kaggle/working/input_image.jpg")
IMAGE_PATH.write_bytes(base64.b64decode(\"\"\"{image_b64}\"\"\"))

# -------------------------------------------------
# Output
# -------------------------------------------------
OUTPUT_DIR = Path("/kaggle/working/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
output_ply = OUTPUT_DIR / "{unique_output_name}"

# -------------------------------------------------
# Load image
# -------------------------------------------------
image, _, f_px = io.load_rgb(IMAGE_PATH)
h, w = image.shape[:2]

# -------------------------------------------------
# Load model
# -------------------------------------------------
state_dict = torch.hub.load_state_dict_from_url(
    "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt",
    progress=True
)
predictor = create_predictor(PredictorParams())
predictor.load_state_dict(state_dict)
predictor.eval().to(device)

# -------------------------------------------------
# Preprocess (STRICT float32)
# -------------------------------------------------
image_pt = (
    torch.from_numpy(image)
    .permute(2, 0, 1)
    .contiguous()
    .to(device)
    .float()
    / 255.0
)
image_resized = F.interpolate(
    image_pt.unsqueeze(0),
    size=(1536, 1536),
    mode="bilinear",
    align_corners=True
).float()
disparity_factor = torch.tensor(
    [float(f_px / w)],
    device=device,
    dtype=torch.float32
)

# -------------------------------------------------
# Predict
# -------------------------------------------------
print("Running SHARP inference...")
gaussians_ndc = predictor(image_resized, disparity_factor)

# -------------------------------------------------
# Unproject (STRICT float32)
# -------------------------------------------------
intrinsics = torch.tensor(
    [[f_px, 0.0, w / 2.0, 0.0],
     [0.0, f_px, h / 2.0, 0.0],
     [0.0, 0.0, 1.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]],
    device=device,
    dtype=torch.float32
)
intrinsics_resized = intrinsics.clone()
intrinsics_resized[0] *= float(1536 / w)
intrinsics_resized[1] *= float(1536 / h)

gaussians = unproject_gaussians(
    gaussians_ndc,
    torch.eye(4, device=device, dtype=torch.float32),
    intrinsics_resized,
    (1536, 1536)
)

# -------------------------------------------------
# Save
# -------------------------------------------------
save_ply(gaussians, f_px, (h, w), output_ply)
print("Saved PLY:", output_ply)
"""
            # Write kernel files
            (kernel_dir / "kernel.py").write_text(script_code, encoding="utf-8")
            metadata = {
                "id": f"{username}/{kernel_slug}",
                "title": kernel_slug,
                "code_file": "kernel.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": "true",
                "enable_gpu": "true",
                "enable_internet": "true",
            }
            with open(kernel_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            
            # -----------------------------------------------------
            # Check if a kernel with the same slug is already running
            # -----------------------------------------------------
            try:
                print("Checking if a kernel with the same slug is already running...")
                status = subprocess.run(
                    ["kaggle", "kernels", "status", f"{username}/{kernel_slug}"],
                    capture_output=True,
                    text=True
                ).stdout.lower()

                if "running" in status or "queued" in status:
                    print("Kernel is already running. Deleting the existing kernel...")
                    subprocess.run(
                        ["kaggle", "kernels", "delete", f"{username}/{kernel_slug}"],
                        check=True
                    )
                    print("Existing kernel deleted successfully.")
            except subprocess.CalledProcessError as e:
                print("No existing kernel found or unable to check status. Proceeding with new kernel.")

            # -----------------------------------------------------
            # Push kernel
            # -----------------------------------------------------
            print("Pushing Kaggle kernel...")
            subprocess.run(
                ["kaggle", "kernels", "push", "-p", str(kernel_dir)],
                check=True
            )
            kernel_ref = f"{username}/{kernel_slug}"
            print("Kernel ref:", kernel_ref)

            # -----------------------------------------------------
            # Wait for completion
            # -----------------------------------------------------
            print("Waiting for kernel to finish...")
            while True:
                time.sleep(6)
                status = subprocess.run(
                    ["kaggle", "kernels", "status", kernel_ref],
                    capture_output=True,
                    text=True
                ).stdout.lower()
                if "complete" in status:
                    break
                if "failed" in status or "error" in status:
                    raise RuntimeError(
                        f"Kaggle kernel failed: https://www.kaggle.com/code/{kernel_ref}"
                    )
            
            # -----------------------------------------------------
            # Download output (SAFE MODE)
            # -----------------------------------------------------
            print("Downloading kernel outputs...")
            output_dir.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                ["kaggle", "kernels", "output", kernel_ref, "-p", str(output_dir)],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print("⚠ Kaggle CLI returned non-zero exit (harmless)")
                print(result.stderr.strip())
            
            # Find the PLY file - check both the unique name and any outputs subdirectory
            expected_ply_file = None
            
            # First, try to find the file with the unique name we specified
            potential_paths = [
                output_dir / "outputs" / unique_output_name,  # Kaggle might create outputs/ subdirectory
                output_dir / unique_output_name,  # Or directly in output_dir
            ]
            
            for path in potential_paths:
                if path.exists():
                    expected_ply_file = path
                    break
            
            # If not found by unique name, find the most recently created PLY file
            if expected_ply_file is None:
                ply_files = list(output_dir.rglob("*.ply"))
                if not ply_files:
                    raise FileNotFoundError("PLY output not found in downloaded files")
                
                # Get the most recently modified file (should be our new output)
                expected_ply_file = max(ply_files, key=lambda p: p.stat().st_mtime)
                print(f"⚠ Could not find file by unique name, using most recent PLY: {expected_ply_file}")

            if not expected_ply_file.exists():
                raise FileNotFoundError(f"Expected PLY file not found: {expected_ply_file}")

            print("✅ Final 3D model:", expected_ply_file)
            return str(expected_ply_file)
            
        finally:
            shutil.rmtree(kernel_dir, ignore_errors=True)
    
    finally:
        # Clean up temporary image file if we created one
        if temp_image_path and temp_image_path.exists():
            temp_image_path.unlink()