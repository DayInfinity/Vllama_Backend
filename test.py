from functions.translate.translate import translate_fast
from functions.object_detect.object_detection_image import object_detection_image
from functions.object_detect.object_detection_video import object_detection_video
from functions.image3d.image3dRemote import run_kaggle_image_to_3d

from pathlib import Path
import base64
import requests


if __name__ == "__main__":
    # text = """
    # hello world this is your boy from bangalore india. I am building vllama which helps in translation"""

    # translated = translate_fast(
    #     text=text,
    #     input_lang="en",
    #     output_lang="de",
    # )

    # print(translated)

    # object_detection_image(path = "outputs/vishvesh_1.jpeg")
    # object_detection_video(video_path = "outputs/test_video.mp4")

    # path = "outputs/test.png"

    url = "http://127.0.0.1:5000/api/generate/object_detection_image"

    image_path = Path("outputs/test.png")
    with image_path.open("rb") as f:
        files = {
            "image": (image_path.name, f, "image/png"),
        }
        data = {
            "model": "yolov8l.pt",
            "use_kaggle": "false",
        }

        resp = requests.post(url, files=files, data=data, timeout=600)

    print(resp.status_code)
    print(resp.json().keys())
    img_b64 = resp.json()["imageData"]
    base64.b64decode(img_b64)  # sanity check that it's valid base64

    print(resp.json()["status"])

    print("test")
    # pass