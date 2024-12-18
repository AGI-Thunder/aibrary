from aibrary import AsyncAiBrary
import asyncio

async def main():
    aibrary = AsyncAiBrary(api_key=None)
    await aibrary.chat.completions.create(
        model="claude-3-5-haiku-20241022",
        messages=[
            {"role": "user", "content": "How are you today?"},
            {"role": "assistant", "content": "what is computer"},
        ],
        temperature=0.7,
        system="you are a teacher of cmputer",
    )

asyncio.run(main())
