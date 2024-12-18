# Install
With these command you can build and install it in your enviourment
```bash
python -m pip venv .venv
source ./venv/bin/activate
pip install -q build
python -m build

pip install ./dist/aibrary-0.1.0-py3-none-any.whl
pip install ./dist/aibrary-0.1.0.tar.gz
```
# Snippet Code
## Sync Requests
```python
from aibrary import AiBrary
aibrary = AiBrary(api_key=None) # either passing api_key to the client or setting the AIBRARY_API_KEY in environment variable
```
### Get All Models
```python
aibrary.get_all_models(filter_category="TTS",return_as_objects=False)
```
### OpenAI Completion Models
```python
aibrary.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "How are you today?"},
        {"role": "assistant", "content": "I'm doing great, thank you!"},
    ],
    temperature=0.7,
)
```
### Anthropic Completion Model
```python
aibrary.chat.completions.create(
    model="claude-3-5-haiku-20241022",
    messages=[
        {"role": "user", "content": "How are you today?"},
        {"role": "assistant", "content": "what is computer"},
    ],
    temperature=0.7,
    system="you are a teacher of cmputer",
)
```

### Text to Speech (TTS)
```python
aibrary.audio.transcriptions.create(
    model="whisper-large-v3", file=open("path/to/audio", "rb")
)
```
### Speech to Text (STT)
```python
response = aibrary.audio.speech.create(
    input="Hey Cena", model="tts-1", response_format="mp3", voice="alloy"
)
open("file.mp3", "wb").write(response.content)
```
### Image Generation
```python
aibrary.images.generate(model="dall-e-2", size="256x256", prompt="Draw cat")
```
### Translation
```python
aibrary.translation.automatic_translation("HI", "phedone", "en", "fa")
```
### OCR
```python
aibrary.ocr(providers='amazon',file=open('tests/assets/ocr-test.jpg','rb').read(),file_name="test.jpg")
# OR only send path
aibrary.ocr(providers='amazon',file='tests/assets/ocr-test.jpg')
# OR send url
aibrary.ocr(providers='amazon',file_url="https://builtin.com/sites/www.builtin.com/files/styles/ckeditor_optimize/public/inline-images/5_python-ocr.jpg")

```
## Async Client
For async version of client you can import `from aibrary import AsyncAiBrary` and only put await behind above exmpale like this example:

```python
from aibrary import AsyncAiBrary
aibrary = AsyncAiBrary(api_key=None)
await aibrary.chat.completions.create(
    model="claude-3-5-haiku-20241022",
    messages=[
        {"role": "user", "content": "who are you what is your role??"},
    ],
    temperature=0.7,
    system="you are a math teacher"
    )
```