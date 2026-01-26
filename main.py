# backend.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from routes.image.route import generate_image
from routes.video.video_kaggle import generate_video
from routes.audio.audio import generate_audio
from routes.health.health import health_check
from routes.STT.stt import generate_transcription
from routes.translate.translate import generate_translation
# from routes.image3d.image3d import generate_3d
from routes.image3d.image3d_da import generate_3d
from routes.object_detect.image_obj_det import generate_object_detection_image
from routes.video3d.video3dRoute import generate_3d_from_video

app = Flask(__name__)
CORS(app)  # Enable CORS for Next.js frontend

# Image URLs
app.add_url_rule('/api/generate/image', 'generate_image', generate_image, methods=['POST'])


# Video URLs
app.add_url_rule('/api/generate/video', 'generate_video', generate_video, methods=['POST'])


# Transcription URLs
app.add_url_rule('/api/generate/transcription', 'generate_transcription', generate_transcription, methods=['POST'])


# Translation URLs
app.add_url_rule('/api/generate/translation', 'generate_translation', generate_translation, methods=['POST'])


# Audio URLs
app.add_url_rule('/api/generate/audio', 'generate_audio', generate_audio, methods=['POST'])


# Health Check
app.add_url_rule('/health', 'health_check', health_check, methods=['GET'])


# Image to 3d URLs
app.add_url_rule('/api/generate/3d', 'generate_3d', generate_3d, methods=['POST'])


# Video to 3d URLs
app.add_url_rule('/api/generate/generate_video_3d', 'generate_3d_from_video', generate_3d_from_video, methods=['POST'])


# Object Detection URLs
app.add_url_rule('/api/generate/object_detection_image', 'generate_object_detection_image', generate_object_detection_image, methods=['POST'])


if __name__ == '__main__':
    # Run on port 5000 (or whatever you want)
    app.run(host='0.0.0.0', port=5000, debug=True)