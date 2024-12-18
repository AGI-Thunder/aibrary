from aibrary import AsyncAiBrary
import asyncio

async def main():
    aibrary = AsyncAiBrary(api_key=None)
    await aibrary.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "How are you today?"},
            {"role": "assistant", "content": "I'm doing great, thank you!"},
        ],
        temperature=0.7,
    )

asyncio.run(main())
