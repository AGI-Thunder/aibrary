import asyncio

import pytest

from aibrary.resources.aibrary_async import AsyncAiBrary
from aibrary.resources.models import Model
from tests.conftest import get_min_model_by_size


@pytest.fixture
def aibrary():
    return AsyncAiBrary()


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop()
    yield loop


@pytest.mark.asyncio
async def test_chat_completions(aibrary: AsyncAiBrary):
    models = await aibrary.get_all_models(filter_category="chat")
    assert len(models) > 0, "There is no model!!!"
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
        assert response.choices[0].message.content, "Value not exist!"


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
    assert response.choices[0].message.content, "Response should not be empty"


@pytest.mark.asyncio
async def test_audio_transcriptions(aibrary: AsyncAiBrary):
    async def _inner_fun(model: Model):
        try:
            with open("tests/assets/file.mp3", "rb") as audio_file:
                return (
                    await aibrary.audio.transcriptions.create(
                        model=model.model_name, file=audio_file
                    ),
                    model,
                )
        except Exception as e:
            return (e, model)

    models = await aibrary.get_all_models(filter_category="stt")
    assert len(models) > 0, "There is no model!!!"
    tasks = [_inner_fun(model) for model in models]

    results = await asyncio.gather(*tasks, return_exceptions=True)
    error = []

    for response_model in results:
        response, model = response_model
        if isinstance(response, Exception):
            print(f"An error occurred: {response}")
            error.append(f"No audio content generated for model: {model.model_name}")
            continue
        assert response, "Response should not be empty"
    if len(error):
        raise AssertionError("\n".join(error))


@pytest.mark.asyncio
async def test_automatic_translation(aibrary: AsyncAiBrary):
    async def _inner_fun(model: Model):
        try:
            return (
                await aibrary.translation(
                    text="HI",
                    model=model.model_name,
                    source_language="en",
                    target_language="ar",
                ),
                model,
            )
        except Exception as e:
            return (e, model)

    models = await aibrary.get_all_models(filter_category="translation")
    assert len(models) > 0, "There is no model!!!"
    tasks = [await _inner_fun(model) for model in models]
    error = []

    for response_model in tasks:
        response, model = response_model

        if isinstance(response, Exception):
            print(f"An error occurred: {response}")
            continue
        assert response["text"], "Response should not be empty"
        if response:
            assert response.content, "Audio content should not be empty"
        else:
            error.append(f"No audio content generated for model: {model.model_name}")
    if len(error):
        raise AssertionError("\n".join(error))


@pytest.mark.asyncio
async def test_audio_speech_creation(aibrary: AsyncAiBrary):
    async def _inner_fun(model: Model):
        await asyncio.sleep(1.5)
        try:
            return (
                await aibrary.audio.speech.create(
                    input="Hey Cena",
                    model=model.model_name,
                    response_format="mp3",
                    voice="FEMALE" if model.provider != "openai" else "alloy",
                )
            ), model
        except Exception as e:
            return (e, model)

    models = await aibrary.get_all_models(filter_category="tts")
    assert len(models) > 0, "There is no model!!!"
    tasks = [await _inner_fun(model) for model in models]
    error = []
    for response_model in tasks:
        response, model = response_model
        if isinstance(response, Exception):
            error.append(f"An error occurred: {model} - {response}")
            continue
        if response:
            assert response.content, "Audio content should not be empty"
            with open(f"var/file-{model.model_name}-async.mp3", "wb") as output_file:
                output_file.write(response.content)
        else:
            error.append(f"No audio content generated for model: {model.model_name}")
    if len(error):
        raise AssertionError("\n".join(error))


@pytest.mark.asyncio
async def test_image_generation_with_multiple_models(aibrary: AsyncAiBrary):
    async def _inner_fun(model: Model):
        try:
            return (
                await aibrary.images.generate(
                    model=model.model_name,
                    size=model.size,
                    prompt="Draw a futuristic cityscape",
                )
            ), model
        except Exception as e:
            return (e, model)

    models = await aibrary.get_all_models(filter_category="image")
    models = get_min_model_by_size(models)

    assert len(models) > 0, "There is no model!!!"

    tasks = [await _inner_fun(model) for model in models]
    error = []
    for response_model in tasks:
        response, model = response_model
        if isinstance(response, Exception):
            error.append(f"An error occurred: {model} - {response}")
            continue
        if response:
            assert response, "Image content should not be empty"
        else:
            error.append(
                f"No image content generated for model: {model.model_name}, response: {response}"
            )

    if len(error):
        raise AssertionError("\n".join(error))
