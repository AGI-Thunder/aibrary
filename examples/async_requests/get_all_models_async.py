import asyncio

from aibrary import AsyncAiBrary


async def main():
    aibrary = AsyncAiBrary()
    await aibrary.get_all_models(filter_category="TTS", return_as_objects=False)


asyncio.run(main())
