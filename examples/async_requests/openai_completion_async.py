import asyncio

from aibrary import AsyncAiBrary


async def main():
    aibrary = AsyncAiBrary()
    await aibrary.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "How are you today?"},
            {"role": "assistant", "content": "I'm doing great, thank you!"},
        ],
        temperature=0.7,
    )


asyncio.run(main())
