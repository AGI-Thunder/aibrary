import asyncio
import base64

from aibrary import AiBrary


def main():
    aibrary = AiBrary()

    # Function to encode the image
    def encode_image(path):
        with open(path, "rb") as file:
            return base64.b64encode(file.read()).decode("utf-8")

    # Getting the base64 string
    english_audio_base64 = encode_image("tests/assets/file.mp3")
    return aibrary.chat.completions.create(
        model="gpt-4o-audio-preview",
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "wav"},
        messages=[
            {"role": "system", "content": "translate to persian"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": english_audio_base64, "format": "mp3"},
                    }
                ],
            },
        ],
    )


main()
