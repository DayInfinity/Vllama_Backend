from flask import request, jsonify
# from io import BytesIO
import base64   
from functions.translate.translate import translate_fast


def generate_translation():
    try:
        data = request.get_json()
        
        # Get parameters
        text = data.get('text')
        source_language = data.get('source_language', 'en')
        target_language = data.get('target_language', 'es')
        model = data.get('model', 'facebook/nllb-200-distilled-600M')
        use_kaggle = data.get('use_kaggle', False)
        
        print("Transcription generation request received:")
        print(f"Text: {text}")
        print(f"Source Language: {source_language}")
        print(f"Target Language : {target_language}" )
        print(f"Model: {model}")
        print(f"Use Kaggle: {use_kaggle}")
        
        # Validate
        if not text:
            return jsonify({'error': 'Text is required to translate'}), 400
        
        # TODO: Add Kaggle support later if needed
        if use_kaggle:
            return jsonify({'error': 'Kaggle support not implemented yet'}), 501
        
        # Map frontend model names to actual model IDs
        model_mapping = {
            'Flux Realism': 'facebook/nllb-200-distilled-600M',
            'Stable Diffusion 3': 'facebook/nllb-200-distilled-600M',
            'DALL-E Style': 'facebook/nllb-200-distilled-600M',
        }
        
        model_id = model_mapping.get(model, 'facebook/nllb-200-distilled-600M')
        
        # Generate transcription
        translated_text = translate_fast(text = text, source_language = source_language, target_language = target_language, model_id = model_id)
        
        print("Translation:", translated_text)

       
        return jsonify({
            'translation': translated_text,
            'text': text,
            'source_language': source_language,
            'target_language': target_language,
            'model': model_id,
            'status': 'success'
        })
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500