import asyncio
import base64

from aibrary import AsyncAiBrary


async def main():
    aibrary = AsyncAiBrary()

    # Function to encode the image
    def encode_image(path):
        with open(path, "rb") as file:
            return base64.b64encode(file.read()).decode("utf-8")

    # Getting the base64 string
    base64_image = encode_image("tests/assets/test-image.jpg")

    return await aibrary.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is in this image?",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
    )


asyncio.run(main())
