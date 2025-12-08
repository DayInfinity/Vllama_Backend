import torch
import os
import time
from diffusers import StableDiffusionPipeline

# Global pipeline (same as your existing code)
_pipeline = None

def _generate_image(prompt: str, model_name: str, output_dir: str = "."):
    """
    Generate image using your existing logic
    Returns the image object (PIL Image)
    """
    global _pipeline
    
    # Device detection (your existing code)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)
        print(f"CUDA device: {props.name}, VRAM: {vram_gb:.2f} GB")
        
        if vram_gb <= 3:
            device = "cuda"
            dtype = torch.float32
            low_vram = True
        else:
            device = "cuda"
            dtype = torch.float16
            low_vram = False
        print("CUDA device detected. Using GPU for inference.")
    
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
        low_vram = False
        print("MPS device detected. Using GPU for inference.")
    
    else:
        device = "cpu"
        dtype = torch.float32
        low_vram = True
        print("No CUDA device detected. Using CPU for inference.")

    # Load model if needed
    if _pipeline is None or getattr(_pipeline, 'model_name', None) != model_name:
        print(f"Loading model '{model_name}' on {device} with dtype = {dtype} ...")
        try:
            _pipeline = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=dtype,
                safety_checker=None,
            )
            _pipeline = _pipeline.to(device)
            _pipeline.low_vram = low_vram
            
            if device == "cuda":
                try:
                    _pipeline.enable_xformers_memory_efficient_attention()
                except Exception as e:
                    print(f"Failed to enable xformers: {e}")
            
            _pipeline.enable_attention_slicing()
            _pipeline.enable_vae_tiling()
            _pipeline.model_name = model_name
            print(f"Model loaded: {model_name}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise

    # Generation parameters
    steps = 50
    height = 512
    width = 512
    guidance = 7.5

    low_vram = getattr(_pipeline, 'low_vram', False)
    
    if low_vram:
        steps = min(steps, 3)
        height = width = 512
        print("Low VRAM mode: reducing steps")
    elif torch.backends.mps.is_available():
        steps = 200
        height = width = 512
        guidance = 7.5

    # Generate image
    print(f"Generating: '{prompt}'...")
    result = _pipeline(
        prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=height,
        width=width
    )
    
    image = result.images[0]
    
    # Save to file (optional, for backup)
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    out_path = os.path.join(output_dir, f"vllama_output_{timestamp}.png")
    image.save(out_path)
    print(f"Saved to {out_path}")
    
    return image