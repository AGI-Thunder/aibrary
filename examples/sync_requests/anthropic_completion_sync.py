from aibrary import AiBrary
aibrary = AiBrary(api_key=None)
aibrary.chat.completions.create(
    model="claude-3-5-haiku-20241022",
    messages=[
        {"role": "user", "content": "How are you today?"},
        {"role": "assistant", "content": "what is computer"},
    ],
    temperature=0.7,
    system="you are a teacher of cmputer",
)
