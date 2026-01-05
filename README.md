---
title: Vllama Backend API
emoji: 🦙
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: apache-2.0
---

# 🦙 Vllama Backend API

A powerful Flask-based REST API for AI-powered content generation including:

- 🖼️ **Image Generation** - Stable Diffusion models
- 🎥 **Video Generation** - Text-to-video via Kaggle
- 🎵 **Audio Generation** - Text-to-speech
- 🗣️ **Speech-to-Text** - Whisper transcription
- 🌍 **Translation** - Multi-language translation
- 🎯 **Object Detection** - YOLOv8 detection
- 🧊 **3D Generation** - Image-to-3D conversion

## API Endpoints

### Health Check
```bash
GET /health
```

### Image Generation
```bash
POST /api/generate/image
Content-Type: application/json

{
  "prompt": "a beautiful sunset over mountains",
  "model": "Flux Realism",
  "use_kaggle": false
}
```

### Video Generation
```bash
POST /api/generate/video
Content-Type: application/json

{
  "prompt": "a cat playing with a ball",
  "model": "Flux Realism",
  "use_kaggle": true
}
```

### Audio Generation
```bash
POST /api/generate/audio
Content-Type: application/json

{
  "prompt": "Hello, welcome to Vllama!",
  "model": "Flux Realism"
}
```

### Speech-to-Text
```bash
POST /api/generate/transcription
Content-Type: application/json

{
  "path": "/path/to/audio.wav",
  "language": "en",
  "model": "openai/whisper-small"
}
```

### Translation
```bash
POST /api/generate/translation
Content-Type: application/json

{
  "text": "Hello world",
  "source_language": "en",
  "target_language": "es",
  "model": "facebook/nllb-200-distilled-600M"
}
```

### Object Detection
```bash
POST /api/generate/object_detection_image
Content-Type: multipart/form-data

image: <file>
model: yolov8l.pt
```

### 3D Generation
```bash
POST /api/generate/3d
Content-Type: multipart/form-data

image: <file>
model: <model_name>
```

## Tech Stack

- **Framework**: Flask + Gunicorn
- **ML Libraries**: 
  - PyTorch
  - Transformers (Hugging Face)
  - Diffusers
  - Ultralytics (YOLOv8)
- **Processing**: OpenCV, Pillow, Trimesh
- **Deployment**: Docker on Hugging Face Spaces

## Notes

- First request may take longer due to model downloading
- Large models are cached after first load
- GPU acceleration when available
- Some features require Kaggle API credentials

## License

Apache 2.0
