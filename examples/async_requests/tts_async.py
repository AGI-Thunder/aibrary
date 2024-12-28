import asyncio

from aibrary import AsyncAiBrary


async def main():
    aibrary = AsyncAiBrary()
    await aibrary.audio.transcriptions.create(
        model="whisper-large-v3", file=open("path/to/audio", "rb")
    )


asyncio.run(main())
