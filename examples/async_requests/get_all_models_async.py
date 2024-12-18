from aibrary import AsyncAiBrary
import asyncio

async def main():
    aibrary = AsyncAiBrary(api_key=None)
    await aibrary.get_all_models(filter_category="TTS", return_as_objects=False)

asyncio.run(main())
