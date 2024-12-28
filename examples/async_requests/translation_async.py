import asyncio

from aibrary import AsyncAiBrary


async def main():
    aibrary = AsyncAiBrary()
    await aibrary.translation.automatic_translation("HI", "phedone", "en", "fa")


asyncio.run(main())
