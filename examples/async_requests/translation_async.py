import asyncio

from aibrary import AsyncAiBrary


async def main():
    aibrary = AsyncAiBrary()
    return await aibrary.translation("HI", "phedone", "en", "fa")


asyncio.run(main())
