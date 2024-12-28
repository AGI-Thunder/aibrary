import asyncio
import os

import pytest

from aibrary import AiBrary
from aibrary.resources.models import Model
from tests.conftest import get_min_model_by_size


@pytest.fixture
def aibrary():
    return AiBrary()


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop()
    yield loop


# @pytest.mark.asyncio
def test_chat_completions(aibrary: AiBrary):
    models = aibrary.get_all_models(filter_category="chat")
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
    error = []
    for response_model in zip(tasks, models):
        response = response_model[0]
        model: Model = response_model[1]
        if isinstance(response, Exception):
            message = f"No chat generated for Provider/Model:{model.provider}/{model.model_name} - {type(response)} - {response}"
            error.append(message)
            continue
        assert response, "Response should not be empty"
        assert response.choices[0].message.content, "Value not exist!"

    if len(error):
        raise AssertionError(f"Errors {len(error)}/{len(tasks)}" + "\n".join(error))


def test_get_all_models(aibrary: AiBrary):
    response = aibrary.get_all_models(filter_category="TTS", return_as_objects=False)
    assert isinstance(response, list), "Response should be a list"


def test_chat_completions_with_system(aibrary: AiBrary):
    response = aibrary.chat.completions.create(
        model="claude-3-5-haiku-20241022",
        messages=[
            {"role": "user", "content": "you are math teacher"},
            {"role": "assistant", "content": "what is subtraction?"},
        ],
        temperature=0.7,
        system="you are math teacher",
    )
    assert response.choices[0].message.content, "Response should not be empty"


def test_audio_transcriptions(aibrary: AiBrary):
    def _inner_fun(model: Model):
        try:
            with open("tests/assets/file.mp3", "rb") as audio_file:
                return (
                    aibrary.audio.transcriptions.create(
                        model=model.model_name, file=audio_file
                    ),
                    model,
                )
        except Exception as e:
            return (e, model)

    models = aibrary.get_all_models(filter_category="stt")
    assert len(models) > 0, "There is no model!!!"
    tasks = [_inner_fun(model) for model in models]
    error = []

    for response_model in tasks:
        response, model = response_model
        if isinstance(response, Exception):
            print(f"An error occurred: {response}")
            error.append(f"No audio content generated for model: {model.model_name}")
            continue
        assert response, "Response should not be empty"
    if len(error):
        raise AssertionError("\n".join(error))


def test_automatic_translation(aibrary: AiBrary):
    def _inner_fun(model: Model):
        asyncio.run(asyncio.sleep(1.5))
        try:
            return (
                aibrary.translation(
                    text="HI",
                    model=model.model_name,
                    source_language="en",
                    target_language="ar",
                ),
                model,
            )
        except Exception as e:
            return (e, model)

    models = aibrary.get_all_models(filter_category="translation")
    assert len(models) > 0, "There is no model!!!"
    tasks = [_inner_fun(model) for model in models]
    error = []

    for response_model in tasks:
        response, model = response_model

        if isinstance(response, Exception):
            print(f"An error occurred: {response}")
            continue
        if response:
            assert "text" in response, "translation content should not be empty"
        else:
            error.append(
                f"No translation content generated for model: {model.model_name}"
            )
    if len(error):
        raise AssertionError("\n".join(error))


def test_audio_speech_creation(aibrary: AiBrary):
    def _inner_fun(model: Model):
        asyncio.run(asyncio.sleep(1.5))
        try:
            return (
                aibrary.audio.speech.create(
                    input="Hey Cena",
                    model=model.model_name,
                    response_format="mp3",
                    voice="FEMALE" if model.provider != "openai" else "alloy",
                )
            ), model
        except Exception as e:
            return (e, model)

    models = aibrary.get_all_models(filter_category="tts")
    assert len(models) > 0, "There is no model!!!"
    tasks = [_inner_fun(model) for model in models]
    error = []
    for response_model in tasks:
        response, model = response_model
        if isinstance(response, Exception):
            error.append(f"An error occurred: {model} - {response}")
            continue
        if response:
            assert response.content, "Audio content should not be empty"
            with open(f"var/file-{model.model_name}-sync.mp3", "wb") as output_file:
                output_file.write(response.content)
        else:
            error.append(f"No audio content generated for model: {model.model_name}")
    if len(error):
        raise AssertionError("\n".join(error))


def test_image_generation_with_multiple_models(aibrary: AiBrary):
    def _inner_fun(model: Model):
        try:
            return (
                aibrary.images.generate(
                    model=model.model_name,
                    size=model.size,
                    prompt="Draw a futuristic cityscape",
                )
            ), model
        except Exception as e:
            return (e, model)

    models = aibrary.get_all_models(filter_category="image")
    models = get_min_model_by_size(models)

    assert len(models) > 0, "There is no model!!!"

    tasks = [_inner_fun(model) for model in models]
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


def test_ocr_with_multiple_modes(aibrary: AiBrary):
    def _inner_fun(model: Model, mode: str, input_data: str):
        asyncio.run(asyncio.sleep(1))
        try:
            if mode == "file":
                response = aibrary.ocr(
                    providers=model.model_name,
                    language="en",
                    file=input_data,
                )
            elif mode == "url":
                response = aibrary.ocr(
                    providers=model.model_name,
                    language="en",
                    file_url=input_data,
                )
            else:
                raise ValueError(f"Invalid mode: {mode}")

            return response, mode, input_data
        except Exception as e:
            return e, mode, input_data

    # Test data
    file_path = "tests/assets/ocr-test.jpg"  # Replace with an actual test file path
    file_url = "https://builtin.com/sites/www.builtin.com/files/styles/ckeditor_optimize/public/inline-images/5_python-ocr.jpg"  # Replace with an actual test URL

    # Ensure test inputs are valid
    assert os.path.isfile(file_path), f"Test file does not exist: {file_path}"

    # Test modes
    inputs = [
        ("file", file_path),
        # ("url", file_url),
    ]
    models = aibrary.get_all_models(filter_category="ocr")

    assert len(models) > 0, "There is no model!!!"
    for mode, input_data in inputs:
        tasks = [_inner_fun(model, mode, input_data) for model in models]
    errors = []

    for response_data in tasks:
        response, mode, input_data = response_data
        if isinstance(response, Exception):
            errors.append(
                f"An error occurred in mode '{mode}' with input '{input_data}': {response}"
            )
            continue
        if response:
            assert (
                response
            ), f"OCR content should not be empty for mode '{mode}' and input '{input_data}'"
        else:
            errors.append(
                f"No OCR content generated for mode '{mode}', input: {input_data}, response: {response}"
            )

    if len(errors):
        raise AssertionError("\n".join(errors))


def generic_with_multiple_modes(
    aibrary: AiBrary,
    method: str,
    filter_category: str,
    include_language: bool = True,
):
    def _inner_fun(model: Model, mode: str, input_data: str):
        asyncio.run(asyncio.sleep(1))
        try:
            kwargs = {
                "providers": model.model_name,
                "file": input_data if mode == "file" else None,
                "file_url": input_data if mode == "url" else None,
            }
            if include_language:
                kwargs["language"] = "en"

            response = getattr(aibrary, method)(
                **{k: v for k, v in kwargs.items() if v is not None}
            )
            return response, mode, input_data, model
        except Exception as e:
            return e, mode, input_data, model

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
    models = aibrary.get_all_models(filter_category=filter_category)

    assert len(models) > 0, f"There is no model for category '{filter_category}'!!!"
    for mode, input_data in inputs:
        tasks = [_inner_fun(model, mode, input_data) for model in models]
    errors = []

    for response_data in tasks:
        response, mode, input_data, model = response_data
        if isinstance(response, Exception):
            errors.append(
                f"An error occurred in mode '{mode}' with input '{input_data}': {response} model:{model}"
            )
            continue
        if response:
            assert (
                response
            ), f"Content should not be empty for mode '{mode}' and input '{input_data}'"
        else:
            errors.append(
                f"No content generated for mode '{mode}', input: {input_data}, response: {response}"
            )

    if len(errors):
        raise AssertionError("\n".join(errors))


@pytest.mark.asyncio
async def test_ocr_with_multiple_modes(aibrary: AiBrary):
    await generic_with_multiple_modes(aibrary, method="ocr", filter_category="ocr")


@pytest.mark.asyncio
async def test_object_detection_with_multiple_modes(aibrary: AiBrary):
    await generic_with_multiple_modes(
        aibrary,
        method="object_detection",
        filter_category="object detection",
        include_language=False,
    )
