from transformers import pipeline
import torch

def transcribe_from_path(path : str = None, model_id: str = 'openai/whisper-small', language: str = 'en'):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Whisper model
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        device=device,
        return_timestamps=True,
        language= language,
    )

    try:
        result = asr_pipeline(path)
        print("\nTranscription:")
        print(result["text"])
        return result["text"]
    except Exception as e:
        print(f"\nError: {e}")
        return