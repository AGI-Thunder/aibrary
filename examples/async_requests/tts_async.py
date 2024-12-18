from aibrary import AsyncAiBrary
import asyncio

async def main():
    aibrary = AsyncAiBrary(api_key=None)
    await aibrary.audio.transcriptions.create(
        model="whisper-large-v3", file=open("path/to/audio", "rb")
    )

asyncio.run(main())
