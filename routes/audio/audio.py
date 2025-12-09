from flask import request, jsonify
# from io import BytesIO
import base64   
from functions.audio.audio import text_to_speech


def generate_audio():
    try:
        data = request.get_json()
        
        # Get parameters
        prompt = data.get('prompt')
        model = data.get('model', 'stabilityai/stable-diffusion-2-1')
        use_kaggle = data.get('use_kaggle', False)
        
        print("Video generation request received:")
        print(f"Prompt: {prompt}")
        print(f"Model: {model}")
        print(f"Use Kaggle: {use_kaggle}")
        
        # Validate
        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400
        
        # TODO: Add Kaggle support later if needed
        if use_kaggle:
            return jsonify({'error': 'Kaggle support not implemented yet'}), 501
        
        # Map frontend model names to actual model IDs
        model_mapping = {
            'Flux Realism': 'microsoft/speecht5_tts',
            'Stable Diffusion 3': 'microsoft/speecht5_tts',
            'DALL-E Style': 'microsoft/speecht5_tts',
        }
        
        model_id = model_mapping.get(model, 'microsoft/speecht5_tts')
        
        # Generate image
        path = text_to_speech(prompt, model_id, output_dir="./outputs")
        
        print("Audio generated at:", path)

        # Convert image to base64
        with open(path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            audio_data = base64.b64encode(audio_bytes).decode('utf-8')
        
        print(f"Returning audio data as base64. {len(audio_data)} characters long.")
        
        # Return base64 image data
        return jsonify({
            'audioData': audio_data,
            'prompt': prompt,
            'model': model,
            'status': 'success'
        })
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500