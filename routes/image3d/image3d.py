from flask import request, jsonify
from io import BytesIO
import base64
import tempfile
from pathlib import Path
from functions.image3d.image3dRemote import run_kaggle_image_to_3d  # Adjust import path as needed
import trimesh  # Add this import

import numpy as np
import trimesh
from plyfile import PlyData

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def convert_ply_to_glb(ply_file_path, max_points=250_000):
    """
    Convert SHARP gaussian PLY (with f_dc_0/1/2) into a GLB point cloud
    with real vertex colors (COLOR_0).

    max_points: downsample for performance (GLB with 1.1M points is huge).
    """
    ply = PlyData.read(str(ply_file_path))
    v = ply["vertex"].data

    # Positions
    xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float32)

    # Colors: prefer standard RGB if present, else derive from f_dc_*
    if all(k in v.dtype.names for k in ("red", "green", "blue")):
        rgb = np.column_stack([v["red"], v["green"], v["blue"]]).astype(np.float32)
        if rgb.max() <= 1.0:
            rgb = (rgb * 255.0)
    elif all(k in v.dtype.names for k in ("f_dc_0", "f_dc_1", "f_dc_2")):
        dc = np.column_stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]]).astype(np.float32)

        # Heuristic:
        # - If already in 0..1 -> use directly
        # - Else map through sigmoid -> 0..1
        if dc.min() >= 0.0 and dc.max() <= 1.0:
            rgb01 = dc
        else:
            rgb01 = _sigmoid(dc)

        rgb = (rgb01 * 255.0)
    else:
        # Fallback: white
        rgb = np.full((xyz.shape[0], 3), 255.0, dtype=np.float32)

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    rgba = np.concatenate([rgb, np.full((rgb.shape[0], 1), 255, dtype=np.uint8)], axis=1)

    # Downsample (IMPORTANT: 1.18M points will be slow/heavy in Flutter)
    n = xyz.shape[0]
    if max_points is not None and n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
        xyz = xyz[idx]
        rgba = rgba[idx]

    pc = trimesh.points.PointCloud(vertices=xyz, colors=rgba)
    glb_data = pc.export(file_type="glb")  # bytes
    return glb_data

def generate_3d():
    """
    Flask route handler for 3D model generation from image.
    Accepts multipart/form-data with 'image' field.
    Returns GLB format instead of PLY.
    """
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        
        # Validate file exists
        if image_file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Read image file data
        image_data = image_file.read()
        
        # Validate it's actually an image (optional but recommended)
        try:
            from PIL import Image
            img = Image.open(BytesIO(image_data))
            img.verify()  # Verify it's a valid image
        except Exception as e:
            return jsonify({'error': f'Invalid image file: {str(e)}'}), 400
        
        # Create output directory
        output_dir = Path("./outputs/3d_models")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate 3D model (pass image bytes directly)
        # This returns the path to the generated PLY file
        ply_file_path = run_kaggle_image_to_3d(image_data, str(output_dir))
        ply_file_path = Path(ply_file_path)
        
        # Convert PLY to GLB
        try:
            glb_data = convert_ply_to_glb(ply_file_path)
        except Exception as e:
            # If conversion fails, log error but you might want to return PLY as fallback
            print(f"Warning: PLY to GLB conversion failed: {e}")
            # Option 1: Return error
            return jsonify({
                'error': f'Failed to convert PLY to GLB: {str(e)}',
                'fallback': 'PLY file generated but conversion failed'
            }), 500
            # Option 2: Fallback to PLY (uncomment if preferred)
            # ply_data = ply_file_path.read_bytes()
            # ply_base64 = base64.b64encode(ply_data).decode('utf-8')
            # return jsonify({
            #     'status': 'success',
            #     'message': '3D model generated (PLY format, GLB conversion failed)',
            #     'plyData': ply_base64,
            #     'fileName': ply_file_path.name,
            # })
        
        # Encode GLB to base64
        glb_base64 = base64.b64encode(glb_data).decode('utf-8')
        
        # Generate filename with .glb extension
        glb_filename = ply_file_path.stem + '.glb'
        
        # Optionally, save the GLB file for future use
        glb_file_path = output_dir / glb_filename
        glb_file_path.write_bytes(glb_data)
        print(f"GLB file saved: {glb_file_path}")
        
        # Return success response with GLB file data
        return jsonify({
            'status': 'success',
            'message': '3D model generated successfully',
            'glbData': glb_base64,  # Changed from 'plyData' to 'glbData'
            'fileName': glb_filename,  # Changed to .glb extension
            'filePath': str(glb_file_path)  # Optional: return GLB path
        })
    
    except FileNotFoundError as e:
        return jsonify({'error': f'File not found: {str(e)}'}), 404
    except RuntimeError as e:
        return jsonify({'error': f'Generation failed: {str(e)}'}), 500
    except Exception as e:
        print(f"Error in generate_3d: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500