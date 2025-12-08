from flask import request, jsonify
from io import BytesIO
import base64   
from functions.video.video_kaggle import run_video_kaggle

def generate_video():
    try:
        data = request.get_json()
        
        # Get parameters
        prompt = data.get('prompt')
        model = data.get('model', 'damo-vilab/text-to-video-ms-1.7b')
        use_kaggle = data.get('use_kaggle', True)

        print("Video generation request received:")
        print(f"Prompt: {prompt}")
        print(f"Model: {model}")
        print(f"Use Kaggle: {use_kaggle}")

        model_mapping = {
            'Flux Realism': 'damo-vilab/text-to-video-ms-1.7b',
            'Stable Diffusion 3': 'damo-vilab/text-to-video-ms-1.7b',
            'DALL-E Style': 'damo-vilab/text-to-video-ms-1.7b',
        }

        model_id = model_mapping.get(model, 'damo-vilab/text-to-video-ms-1.7b')
        
        # Validate
        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400

        if use_kaggle:
            print(f"Entered Kaggle video generation block. Building with model: {model_id}")
            # Run video generation on Kaggle
            output_dir = "./outputs"
            video_path = run_video_kaggle(model_id, prompt, output_dir)
            print("Video generated at:", video_path)
            # Read video and convert to base64
            with open(video_path, "rb") as video_file:
                video_data = video_file.read()
                video_base64 = base64.b64encode(video_data).decode('utf-8')
            
            print("Returning video data as base64.")
            return jsonify({
                'videoData': video_base64,
                'prompt': prompt,
                'model': model,
                'status': 'success'
            })
        else:
            print("Non-Kaggle video generation requested, but not implemented.")
            return jsonify({'error': 'Only Kaggle video generation is supported currently'}), 501
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500