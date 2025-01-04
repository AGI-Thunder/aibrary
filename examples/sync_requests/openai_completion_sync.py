from aibrary import AiBrary

aibrary = AiBrary()
aibrary.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "How are you today?"},
        {"role": "assistant", "content": "I'm doing great, thank you!"},
    ],
    temperature=0.7,
)
