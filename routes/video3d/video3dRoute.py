from flask import request, jsonify
from io import BytesIO
import base64
import tempfile
from pathlib import Path
from functions.video3d.video3dRemoteDA import run_kaggle_video_to_3d
import numpy as np
import trimesh
from plyfile import PlyData


def _sigmoid(x):
    """Apply sigmoid activation function"""
    return 1.0 / (1.0 + np.exp(-x))


def convert_ply_to_glb(ply_file_path, max_points=250_000):
    """
    Convert SHARP/Pi3 gaussian PLY to GLB point cloud format.
    
    Args:
        ply_file_path: Path to input PLY file
        max_points: Maximum points to include (for performance)
    
    Returns:
        bytes: GLB file data
    """
    ply = PlyData.read(str(ply_file_path))
    v = ply["vertex"].data

    # Extract positions
    xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float32)

    # Extract colors - try multiple formats
    if all(k in v.dtype.names for k in ("red", "green", "blue")):
        # Standard RGB format
        rgb = np.column_stack([v["red"], v["green"], v["blue"]]).astype(np.float32)
        if rgb.max() <= 1.0:
            rgb = (rgb * 255.0)
    elif all(k in v.dtype.names for k in ("f_dc_0", "f_dc_1", "f_dc_2")):
        # Gaussian splatting format
        dc = np.column_stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]]).astype(np.float32)
        
        # Apply sigmoid if values are outside [0, 1]
        if dc.min() >= 0.0 and dc.max() <= 1.0:
            rgb01 = dc
        else:
            rgb01 = _sigmoid(dc)
        
        rgb = (rgb01 * 255.0)
    else:
        # Fallback to white
        rgb = np.full((xyz.shape[0], 3), 255.0, dtype=np.float32)

    # Clip and convert to uint8
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    
    # Add alpha channel
    rgba = np.concatenate([rgb, np.full((rgb.shape[0], 1), 255, dtype=np.uint8)], axis=1)

    # Downsample if needed (important for performance)
    n = xyz.shape[0]
    if max_points is not None and n > max_points:
        print(f"Downsampling from {n} to {max_points} points for GLB")
        idx = np.random.choice(n, max_points, replace=False)
        xyz = xyz[idx]
        rgba = rgba[idx]

    # Create point cloud and export to GLB
    pc = trimesh.points.PointCloud(vertices=xyz, colors=rgba)
    glb_data = pc.export(file_type="glb")
    
    return glb_data


def generate_3d_from_video():
    """
    Flask route handler for video-to-3D model generation.
    
    Accepts:
        - multipart/form-data with 'video' file
        - form fields: username, apiKey, frameInterval (optional)
    
    Returns:
        JSON response with GLB file data (base64 encoded)
    """
    try:
        # ---------------------------------------------------------
        # Validate video file presence
        # ---------------------------------------------------------
        if 'video' not in request.files:
            return jsonify({
                'error': 'No video file provided',
                'message': 'Please upload a video file in the "video" field'
            }), 400
        
        video_file = request.files['video']
        
        # Validate file exists and has a name
        if video_file.filename == '':
            return jsonify({
                'error': 'No video file selected',
                'message': 'The uploaded file has no filename'
            }), 400
        
        # ---------------------------------------------------------
        # Extract form parameters
        # ---------------------------------------------------------
        data = request.form
        
        username = data.get('username')
        api_key = data.get('apiKey')
        frame_interval = int(data.get('frameInterval', 10))  # Default: 10
        
        # Validate required credentials
        if not username or not api_key:
            return jsonify({
                'error': 'Missing Kaggle credentials',
                'message': 'Both username and apiKey are required'
            }), 400
        
        # Validate frame interval
        if frame_interval < 1 or frame_interval > 100:
            return jsonify({
                'error': 'Invalid frame interval',
                'message': 'Frame interval must be between 1 and 100'
            }), 400
        
        print("=" * 70)
        print("VIDEO TO 3D GENERATION REQUEST")
        print("=" * 70)
        print(f"Video filename: {video_file.filename}")
        print(f"Username: {username}")
        print(f"Frame interval: {frame_interval}")
        
        # ---------------------------------------------------------
        # Read video file data
        # ---------------------------------------------------------
        video_data = video_file.read()
        
        # Validate video file size (optional but recommended)
        video_size_mb = len(video_data) / (1024 * 1024)
        print(f"Video size: {video_size_mb:.2f} MB")
        
        if video_size_mb > 500:  # 500 MB limit
            return jsonify({
                'error': 'Video file too large',
                'message': f'Video size ({video_size_mb:.1f} MB) exceeds 500 MB limit'
            }), 400
        
        # Validate video format (basic check)
        allowed_extensions = {'.mp4', '.mov', '.avi', '.mkv'}
        file_ext = Path(video_file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            return jsonify({
                'error': 'Unsupported video format',
                'message': f'Supported formats: {", ".join(allowed_extensions)}'
            }), 400
        
        # ---------------------------------------------------------
        # Create output directory
        # ---------------------------------------------------------
        output_dir = Path("./outputs/3d_models_video")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # ---------------------------------------------------------
        # Generate 3D model (pass video bytes directly)
        # ---------------------------------------------------------
        print("\nStarting 3D generation process...")
        
        ply_file_path = run_kaggle_video_to_3d(
            video_input=video_data,
            output_dir=str(output_dir),
            username=username,
            api_key=api_key,
            frame_interval=frame_interval
        )
        
        ply_file_path = Path(ply_file_path)
        
        if not ply_file_path.exists():
            raise FileNotFoundError(f"Generated PLY file not found: {ply_file_path}")
        
        print(f"\n✅ PLY file generated: {ply_file_path}")
        
        # ---------------------------------------------------------
        # Convert PLY to GLB
        # ---------------------------------------------------------
        print("\nConverting PLY to GLB format...")
        
        try:
            glb_data = convert_ply_to_glb(ply_file_path, max_points=250_000)
            print("✅ PLY to GLB conversion successful")
        except Exception as e:
            print(f"❌ PLY to GLB conversion failed: {e}")
            
            # Return error with option to download PLY as fallback
            return jsonify({
                'error': 'GLB conversion failed',
                'message': f'3D model was generated but GLB conversion failed: {str(e)}',
                'fallback': 'PLY file available',
                'plyPath': str(ply_file_path)
            }), 500
        
        # ---------------------------------------------------------
        # Encode GLB to base64
        # ---------------------------------------------------------
        glb_base64 = base64.b64encode(glb_data).decode('utf-8')
        glb_filename = ply_file_path.stem + '.glb'
        
        # Save GLB file for future reference
        glb_file_path = output_dir / glb_filename
        glb_file_path.write_bytes(glb_data)
        
        glb_size_mb = len(glb_data) / (1024 * 1024)
        print(f"✅ GLB file saved: {glb_file_path} ({glb_size_mb:.2f} MB)")
        
        # ---------------------------------------------------------
        # Return success response
        # ---------------------------------------------------------
        print("=" * 70)
        print("✅ VIDEO TO 3D GENERATION COMPLETE")
        print("=" * 70)
        
        return jsonify({
            'status': 'success',
            'message': '3D model generated successfully from video',
            'glbData': glb_base64,
            'fileName': glb_filename,
            'filePath': str(glb_file_path),
            'fileSize': f"{glb_size_mb:.2f} MB",
            'frameInterval': frame_interval
        }), 200
    
    except FileNotFoundError as e:
        print(f"❌ File not found error: {e}")
        return jsonify({
            'error': 'File not found',
            'message': str(e)
        }), 404
    
    except RuntimeError as e:
        print(f"❌ Runtime error: {e}")
        return jsonify({
            'error': 'Generation failed',
            'message': str(e)
        }), 500
    
    except ValueError as e:
        print(f"❌ Value error: {e}")
        return jsonify({
            'error': 'Invalid input',
            'message': str(e)
        }), 400
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'error': 'Unexpected error',
            'message': str(e)
        }), 500