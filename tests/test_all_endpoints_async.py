import asyncio

import pytest

from aibrary import AsyncAiBrary


@pytest.fixture
def aibrary():
    return AsyncAiBrary()


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop()
    yield loop


@pytest.mark.asyncio
async def test_chat_completions(aibrary: AsyncAiBrary):
    models = await aibrary.get_all_models(filter_category="Chat")
    tasks = [
        aibrary.chat.completions.create(
            model=model.model_name,
            messages=[
                {"role": "user", "content": "what is 2+2? return only a number."},
            ],
            temperature=0.7,
        )
        for model in models
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for response in results:
        if isinstance(response, Exception):
            print(f"An error occurred: {response}")
            continue
        assert response, "Response should not be empty"
        assert "4" in response.choices[0].message.content, "Value not exist!"


@pytest.mark.asyncio
async def test_get_all_models(aibrary: AsyncAiBrary):
    response = await aibrary.get_all_models(return_as_objects=False)
    assert len(response) > 0, "There is no model!!!"
    assert isinstance(response, list), "Response should be a list"


@pytest.mark.asyncio
async def test_chat_completions_with_system(aibrary: AsyncAiBrary):
    response = await aibrary.chat.completions.create(
        model="claude-3-5-haiku-20241022",
        messages=[
            {"role": "user", "content": "How are you today?"},
            {"role": "assistant", "content": "what is computer"},
        ],
        temperature=0.7,
        system="you are a teacher of cmputer",
    )
    assert response, "Response should not be empty"


@pytest.mark.asyncio
async def test_audio_transcriptions(aibrary: AsyncAiBrary):
    with open("var/file.mp3", "rb") as audio_file:
        response = await aibrary.audio.transcriptions.create(
            model="whisper-large-v3", file=audio_file
        )
    assert response, "Response should not be empty"


@pytest.mark.asyncio
async def test_automatic_translation(aibrary: AsyncAiBrary):
    response = await aibrary.translation(
        text="HI", model="phedone", source_language="en", target_language="fa"
    )
    assert response["text"], "Response should not be empty"


@pytest.mark.asyncio
async def test_audio_speech_creation(aibrary: AsyncAiBrary):
    response = await aibrary.audio.speech.create(
        input="Hey Cena", model="tts-1", response_format="mp3", voice="alloy"
    )
    with open("file.mp3", "wb") as output_file:
        output_file.write(response.content)
    assert response.content, "Audio content should not be empty"


@pytest.mark.asyncio
async def test_image_generation(aibrary: AsyncAiBrary):
    response = await aibrary.images.generate(
        model="dall-e-2", size="256x256", prompt="Draw cat"
    )
    assert response, "Response should not be empty"
