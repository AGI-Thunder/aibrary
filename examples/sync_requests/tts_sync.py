from aibrary import AiBrary
aibrary = AiBrary(api_key=None)
aibrary.audio.transcriptions.create(
    model="whisper-large-v3", file=open("path/to/audio", "rb")
)
