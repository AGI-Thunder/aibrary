from aibrary import AiBrary
aibrary = AiBrary(api_key=None)
response = aibrary.audio.speech.create(
    input="Hey Cena", model="tts-1", response_format="mp3", voice="alloy"
)
open("file.mp3", "wb").write(response.content)
