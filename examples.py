# %%
from aibrary.resources.aibrary_wrapper import AiBrary

aibrary = AiBrary(api_key=None)
aibrary.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "How are you today?"},
        {"role": "assistant", "content": "I'm doing great, thank you!"},
    ],
    temperature=0.7,
)

# %%
from aibrary.resources.aibrary_wrapper import AiBrary

aibrary = AiBrary(api_key=None)
aibrary.chat.completions.create(
    model="claude-3-5-haiku-20241022",
    messages=[
        {"role": "user", "content": "How are you today?"},
        {"role": "assistant", "content": "I'm doing great, thank you!"},
    ],
    temperature=0.7,
    extra_body={"system": "you are a math teacher"},
)

# %%
from aibrary.resources.aibrary_wrapper import AiBrary

aibrary = AiBrary(api_key=None)
aibrary.audio.transcriptions.create(
    model="whisper-large-v3", file=open("path/to/audio", "rb")
)

# %%
from aibrary.resources.aibrary_wrapper import AiBrary

aibrary = AiBrary(api_key=None)
aibrary.translation.automatic_translation("HI", "phedone", "en", "fa")
