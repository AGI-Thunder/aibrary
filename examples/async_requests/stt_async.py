import asyncio

from aibrary import AsyncAiBrary


async def main():
    aibrary = AsyncAiBrary()
    response = await aibrary.audio.speech.create(
        input="Hey Cena", model="tts-1", response_format="mp3", voice="alloy"
    )
    open("file.mp3", "wb").write(response.content)


asyncio.run(main())
