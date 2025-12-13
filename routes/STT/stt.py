from flask import request, jsonify
# from io import BytesIO
import base64   
from functions.STT.stt import  transcribe_from_path


def generate_transcription():
    try:
        data = request.get_json()
        
        # Get parameters
        path = data.get('path')
        language = data.get('language', 'en')
        model = data.get('model', 'openai/whisper-small')
        use_kaggle = data.get('use_kaggle', False)
        
        print("Transcription generation request received:")
        print(f"Path: {path}")
        print(f"Language: {language}")
        print(f"Model: {model}")
        print(f"Use Kaggle: {use_kaggle}")
        
        # Validate
        if not path:
            return jsonify({'error': 'Path is required'}), 400
        
        # TODO: Add Kaggle support later if needed
        if use_kaggle:
            return jsonify({'error': 'Kaggle support not implemented yet'}), 501
        
        # Map frontend model names to actual model IDs
        model_mapping = {
            'Flux Realism': 'openai/whisper-small',
            'Stable Diffusion 3': 'openai/whisper-small',
            'DALL-E Style': 'openai/whisper-small',
        }
        
        model_id = model_mapping.get(model, 'openai/whisper-small')
        
        # Generate transcription
        transcription = transcribe_from_path(path = path, model_id = model_id, language = language)
        
        print(" Transcription:", transcription)

       
        return jsonify({
            'transcription': transcription,
            'path': path,
            'language': language,
            'model': model,
            'status': 'success'
        })
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500