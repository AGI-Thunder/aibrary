import asyncio
import os

import pytest

from aibrary import AsyncAiBrary
from aibrary.resources.models import Model
from tests.conftest import get_min_model_by_size

# Const
MAX_RETRIES = 5


@pytest.fixture
def aibrary():
    return AsyncAiBrary()


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop()
    yield loop


@pytest.mark.asyncio
async def test_chat_completions(aibrary: AsyncAiBrary):
    async def new_func(aibrary: AsyncAiBrary, model: Model, index: int):
        last_exception = None
        for attempt in range(MAX_RETRIES):
            try:
                return await aibrary.chat.completions.create(
                    model=f"{model.model_name}@{model.provider}",
                    max_completion_tokens=50,
                    messages=[
                        {
                            "role": "user",
                            "content": "Hi!",
                        },
                    ],
                )
            except Exception as e:
                last_exception = e
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(attempt * 3)
                    print("Failed:", attempt)
                    continue  # Retry
                else:
                    raise last_exception

    models = await aibrary.get_all_models(filter_category="chat")
    assert len(models) > 0, "There is no model!!!"
    atasks = [new_func(aibrary, model, index) for index, model in enumerate(models)]

    tasks = await asyncio.gather(*atasks, return_exceptions=True)
    error = []
    for response_model in zip(tasks, models):
        response = response_model[0]
        model: Model = response_model[1]
        if isinstance(response, Exception):
            message = f"No chat generated for Provider/Model:{model.provider}/{model.model_name} - {type(response)} - {response}"
            error.append(message)
            continue

    if len(error):
        raise AssertionError(
            f"Passed {len(tasks) - len(error)}/{len(tasks)}\n" + "\n".join(error)
        )


@pytest.mark.asyncio
async def test_get_all_models(aibrary: AsyncAiBrary):
    response = await aibrary.get_all_models(return_as_objects=False)
    assert isinstance(response, list), "Response should be a list"


@pytest.mark.asyncio
async def test_chat_completions_with_system(aibrary: AsyncAiBrary):
    response = await aibrary.chat.completions.create(
        model="claude-3-5-haiku-20241022",
        messages=[
            {"role": "system", "content": "you are math teacher"},
            {"role": "user", "content": "what is subtraction?"},
        ],
        temperature=0.7,
    )
    assert response.choices[0].message.content, "Response should not be empty"


@pytest.mark.asyncio
async def test_audio_transcriptions(aibrary: AsyncAiBrary):
    async def _inner_fun(model: Model):
        last_exception = None
        for attempt in range(MAX_RETRIES):
            try:
                with open("tests/assets/file.mp3", "rb") as audio_file:
                    return (
                        await aibrary.audio.transcriptions.create(
                            model=model.model_name, file=audio_file
                        ),
                        model,
                    )
            except Exception as e:
                last_exception = e
                if attempt < MAX_RETRIES - 1:
                    continue  # Retry
                else:
                    return (last_exception, model)

    models = await aibrary.get_all_models(filter_category="stt")
    assert len(models) > 0, "There is no model!!!"
    tasks = [await _inner_fun(model) for model in models]
    error = []

    for response_model in tasks:
        response, model = response_model
        if isinstance(response, Exception):
            print(f"An error occurred: {response}")
            error.append(f"No audio content generated for model: {model.model_name}")
            continue

    if len(error):
        raise AssertionError(
            f"Passed {len(tasks) - len(error)}/{len(tasks)}\n" + "\n".join(error)
        )


@pytest.mark.asyncio
async def test_automatic_translation(aibrary: AsyncAiBrary):
    async def _inner_fun(model: Model):
        last_exception = None
        for attempt in range(MAX_RETRIES):
            try:
                await asyncio.sleep(2)
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
                last_exception = e
                if attempt < MAX_RETRIES - 1:
                    continue  # Retry
                else:
                    return (last_exception, model)

    models = await aibrary.get_all_models(filter_category="translation")
    assert len(models) > 0, "There is no model!!!"
    tasks = [await _inner_fun(model) for model in models]
    error = []

    for response_model in tasks:
        response, model = response_model

        if isinstance(response, Exception):
            error.append(f"An error occurred: {response}")
    if len(error):
        raise AssertionError(
            f"Passed {len(tasks) - len(error)}/{len(tasks)}\n" + "\n".join(error)
        )


@pytest.mark.asyncio
async def test_audio_speech_creation(aibrary: AsyncAiBrary):
    async def _inner_fun(model: Model):
        last_exception = None
        for attempt in range(MAX_RETRIES):
            try:
                await asyncio.sleep(3)
                return (
                    await aibrary.audio.speech.create(
                        input="Hey Cena",
                        model=model.model_name,
                        response_format="mp3",
                        voice="FEMALE" if model.provider != "openai" else "alloy",
                    )
                ), model
            except Exception as e:
                last_exception = e
                if attempt < MAX_RETRIES - 1:
                    continue  # Retry
                else:
                    return (last_exception, model)

    models = await aibrary.get_all_models(filter_category="tts")
    assert len(models) > 0, "There is no model!!!"
    tasks = [await _inner_fun(model) for model in models]
    error = []
    for response_model in tasks:
        response, model = response_model
        if isinstance(response, Exception):
            error.append(f"An error occurred: {model} - {response}")
            continue

    if len(error):
        raise AssertionError(
            f"Passed {len(tasks) - len(error)}/{len(tasks)}\n" + "\n".join(error)
        )


@pytest.mark.asyncio
async def test_image_generation_with_multiple_models(aibrary: AsyncAiBrary):
    async def _inner_fun(model: Model):
        last_exception = None
        for attempt in range(MAX_RETRIES):
            try:
                return (
                    await aibrary.images.generate(
                        model=model.model_name,
                        size=model.size,
                        prompt="Draw a futuristic cityscape",
                        response_format="b64_json",
                    )
                ), model
            except Exception as e:
                last_exception = e
                if attempt < MAX_RETRIES - 1:
                    continue  # Retry
                else:
                    return (last_exception, model)

    models = await aibrary.get_all_models(filter_category="image")
    models = get_min_model_by_size(models)

    assert len(models) > 0, "There is no model!!!"

    tasks = [await _inner_fun(model) for model in models]
    error = []
    for response_model in tasks:
        response, model = response_model
        if isinstance(response, Exception):
            error.append(f"An error occurred: {model.model_name} - {response}")
            continue

    if len(error):
        raise AssertionError(f"{len(error)}/{len(models)}" + "\n".join(error))


async def generic_with_multiple_modes(
    aibrary: AsyncAiBrary,
    method: str,
    filter_category: str,
    include_language: bool = True,
):
    async def _inner_fun(model: Model, mode: str, input_data: str):
        last_exception = None
        for attempt in range(MAX_RETRIES):
            try:
                await asyncio.sleep(1)
                kwargs = {
                    "providers": model.model_name,
                    "file": input_data if mode == "file" else None,
                    "file_url": input_data if mode == "url" else None,
                }
                if include_language:
                    kwargs["language"] = "en"

                response = await getattr(aibrary, method)(
                    **{k: v for k, v in kwargs.items() if v is not None}
                )
                return response, mode, input_data, model

            except Exception as e:
                last_exception = e
                if attempt < MAX_RETRIES - 1:
                    continue  # Retry
                else:
                    return last_exception, mode, input_data, model

    # Test data
    file_path = "tests/assets/test-image.jpg"  # Replace with an actual test file path
    file_url = "https://builtin.com/sites/www.builtin.com/files/styles/ckeditor_optimize/public/inline-images/5_python-ocr.jpg"  # Replace with an actual test URL

    # Ensure test inputs are valid
    assert os.path.isfile(file_path), f"Test file does not exist: {file_path}"

    # Test modes
    inputs = [
        ("file", file_path),
        # ("url", file_url),
    ]
    models = await aibrary.get_all_models(filter_category=filter_category)

    assert len(models) > 0, f"There is no model for category '{filter_category}'!!!"
    for mode, input_data in inputs:
        tasks = [await _inner_fun(model, mode, input_data) for model in models]
    errors = []

    for response_data in tasks:
        response, mode, input_data, model = response_data
        if isinstance(response, Exception):
            errors.append(
                f"An error occurred in mode '{mode}' with input '{input_data}': {response} model:{model}"
            )
            continue

    if len(errors):
        raise AssertionError("\n".join(errors))


@pytest.mark.asyncio
async def test_ocr_with_multiple_modes(aibrary: AsyncAiBrary):
    await generic_with_multiple_modes(aibrary, method="ocr", filter_category="ocr")


@pytest.mark.asyncio
async def test_object_detection_with_multiple_modes(aibrary: AsyncAiBrary):
    await generic_with_multiple_modes(
        aibrary,
        method="object_detection",
        filter_category="object detection",
        include_language=False,
    )
