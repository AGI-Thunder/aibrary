from aibrary import AiBrary

aibrary = AiBrary()
aibrary.audio.transcriptions.create(
    model="whisper-large-v3", file=open("path/to/audio", "rb")
)
