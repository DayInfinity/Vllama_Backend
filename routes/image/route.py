from flask import request, jsonify
from io import BytesIO
import base64   
from functions.image.image_local import _generate_image


# @app.route('/api/generate/image', methods=['POST'])
def generate_image():
    try:
        data = request.get_json()
        
        # Get parameters
        prompt = data.get('prompt')
        model = data.get('model', 'stabilityai/stable-diffusion-2-1')
        use_kaggle = data.get('use_kaggle', False)
        
        # Validate
        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400
        
        # TODO: Add Kaggle support later if needed
        if use_kaggle:
            return jsonify({'error': 'Kaggle support not implemented yet'}), 501
        
        # Map frontend model names to actual model IDs
        model_mapping = {
            'Flux Realism': 'stabilityai/sd-turbo',
            'Stable Diffusion 3': 'stabilityai/sd-turbo',
            'DALL-E Style': 'stabilityai/sd-turbo',
        }
        
        model_id = model_mapping.get(model, 'stabilityai/sd-turbo')
        
        # Generate image
        image = _generate_image(prompt, model_id, output_dir="./outputs")
        
        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Return base64 image data
        return jsonify({
            'imageData': img_base64,
            'prompt': prompt,
            'model': model,
            'status': 'success'
        })
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500