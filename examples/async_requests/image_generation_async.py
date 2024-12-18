from aibrary import AsyncAiBrary
import asyncio

async def main():
    aibrary = AsyncAiBrary(api_key=None)
    await aibrary.images.generate(model="dall-e-2", size="256x256", prompt="Draw cat")

asyncio.run(main())
