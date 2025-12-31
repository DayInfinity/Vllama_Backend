from flask import request, jsonify
from io import BytesIO
import base64
import tempfile
from pathlib import Path
from functions.object_detect.object_detection_image import object_detection_image # Adjust import path as needed
from functions.object_detect.object_detection_image import object_detection_image
import cv2  # Add this import

import numpy as np
import trimesh
from plyfile import PlyData

def generate_object_detection_image():

    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        data = request.form

        model = data.get('model', "yolov8l.pt")
        use_kaggle = data.get('use_kaggle', "false")
        
        image_file = request.files["image"]

        # Validate file exists
        if image_file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        if use_kaggle == "true":
            return jsonify({'error': 'Kaggle support not implemented yet'}), 501
        
        # Read image file data
        image_data = image_file.read()

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(image_file.filename).suffix) as temp_image:
            temp_image.write(image_data)
            temp_image_path = temp_image.name

        # Call the object detection function
        result = object_detection_image(path=temp_image_path, model_id=model, output_dir="outputs/object_detection")

        # Clean up temporary file
        # Path(temp_image_path).unlink(missing_ok=True)

        # Return the result as base64
        _, buffer = cv2.imencode('.jpg', result)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Return the result
        return jsonify({
            'imageData': img_base64,
            'model': model,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500