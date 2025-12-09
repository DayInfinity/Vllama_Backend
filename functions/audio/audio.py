from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
import numpy as np
import soundfile as sf
import re

def split_text(text, max_chars=450):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current = ""

    for s in sentences:
        if len(current) + len(s) <= max_chars:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = s
    if current:
        chunks.append(current.strip())

    return chunks

def text_to_speech(text: str = "Hello world!", model_id: str = "microsoft/speecht5_tts", output_dir="./outputs"):
    # Load components
    processor = SpeechT5Processor.from_pretrained(model_id)
    model = SpeechT5ForTextToSpeech.from_pretrained(model_id)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    speaker_embedding = torch.tensor(np.random.randn(1, 512), dtype=torch.float32)

    # text = """YI have built a small but highly useful framework VLLAMA, which can help anyone to take
    # leverage of the open models and build something great, even without great architecture
    # available with them.
    # This is a library using which users can download vision models to generate images, videos with
    # just a single terminal command in their own system or using the free gpu providing services like
    # Kaggle's 30 hr free gpu usage for week, and also inspired from Ollama, it also has llm feature as
    # well, like the user can download the llm, and use it anywhere. One good feature is that I have
    # built an extension in vscode, named vllama, using which the developers can directly interact
    # with the local llm from vscode's chat with ai section. In future, I am planning to expand this
    # feature such that any company or individuals or community can deploy the llm on a gpu service
    # and can directly use this in the vscode to chat, fix in line code and add agentic features using
    # the local llm. It has many other features as well.
    # """

    # Split into safe chunks
    chunks = split_text(text)
    print("Chunks:", len(chunks))

    all_audio = []

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...")

        inputs = processor(text=chunk, return_tensors="pt")

        with torch.no_grad():
            spectrogram = model.generate_speech(
                input_ids=inputs["input_ids"],
                speaker_embeddings=speaker_embedding,
            )

        with torch.no_grad():
            audio = vocoder(spectrogram)

        audio = audio.squeeze().cpu().numpy()
        all_audio.append(audio)

    # Combine audio parts
    final_audio = np.concatenate(all_audio)

    output_path = f"{output_dir}/output.wav"

    sf.write(output_path, final_audio, 16000)
    print("Saved output.wav")
    return output_path

if __name__ == "__main__":
    # text_to_speech(text = "Hello world! My name is VLLAMA. I am a useful framework to build applications using open models. I can help you build image generation, video generation and local llm applications with just a single command.")
    text_to_speech()