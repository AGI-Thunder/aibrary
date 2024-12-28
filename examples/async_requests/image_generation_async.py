import asyncio

from aibrary import AsyncAiBrary


async def main():
    aibrary = AsyncAiBrary()
    await aibrary.images.generate(model="dall-e-2", size="256x256", prompt="Draw cat")


asyncio.run(main())
