from aibrary import AsyncAiBrary
import asyncio

async def main():
    aibrary = AsyncAiBrary(api_key=None)
    response = await aibrary.audio.speech.create(
        input="Hey Cena", model="tts-1", response_format="mp3", voice="alloy"
    )
    open("file.mp3", "wb").write(response.content)

asyncio.run(main())
