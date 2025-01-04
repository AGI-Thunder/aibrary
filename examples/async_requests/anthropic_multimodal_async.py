import asyncio
import base64

from aibrary import AsyncAiBrary


async def main():
    def encode_image(path):
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # Getting the base64 string
    base64_image = encode_image("tests/assets/test-image.jpg")

    aibrary = AsyncAiBrary()
    message_list = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image,
                    },
                },
                {"type": "text", "text": "what is in picture"},
            ],
        }
    ]

    return await aibrary.chat.completions.create(
        model="claude-3-haiku-20240307", max_tokens=2048, messages=message_list
    )


asyncio.run(main())
