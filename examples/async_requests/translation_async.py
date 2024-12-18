from aibrary import AsyncAiBrary
import asyncio

async def main():
    aibrary = AsyncAiBrary(api_key=None)
    await aibrary.translation.automatic_translation("HI", "phedone", "en", "fa")

asyncio.run(main())
