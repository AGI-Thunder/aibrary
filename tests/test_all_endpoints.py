import pytest

from aibrary import AiBrary
from aibrary.resources.models import Model


@pytest.fixture
def aibrary():
    return AiBrary()


def test_chat_completions(aibrary: AiBrary):
    models = aibrary.get_all_models(filter_category="chat")
    assert len(models) > 0, "There is no model!!!"
    results = []
    for model in models:
        try:
            response = aibrary.chat.completions.create(
                model=model.model_name,
                messages=[
                    {"role": "user", "content": "what is 2+2? return only a number."},
                ],
                temperature=0.7,
            )
            results.append(response)
        except Exception as e:
            results.append(e)

    for response in results:
        if isinstance(response, Exception):
            print(f"An error occurred: {response}")
            continue
        assert response, "Response should not be empty"
        assert "4" in response.choices[0].message.content, "Value not exist!"


def test_get_all_models(aibrary: AiBrary):
    response = aibrary.get_all_models(return_as_objects=False)
    assert len(response) > 0, "There is no model!!!"
    assert isinstance(response, list), "Response should be a list"


def test_chat_completions_with_system(aibrary: AiBrary):
    response = aibrary.chat.completions.create(
        model="claude-3-5-haiku-20241022",
        messages=[
            {"role": "user", "content": "How are you today?"},
            {"role": "assistant", "content": "what is computer"},
        ],
        temperature=0.7,
        system="you are a teacher of computer",
    )
    assert response, "Response should not be empty"


def test_audio_transcriptions(aibrary: AiBrary):
    def _inner_fun(model: Model):
        with open("../var/file.mp3", "rb") as audio_file:
            return aibrary.audio.transcriptions.create(
                model=model.model_name, file=audio_file
            )

    models = aibrary.get_all_models(filter_category="stt")
    assert len(models) == 3, "There is no model!!!"
    results = []
    for model in models:
        try:
            results.append(_inner_fun(model))
        except Exception as e:
            results.append(e)

    for response in results:
        if isinstance(response, Exception):
            print(f"An error occurred: {response}")
            continue
        assert response, "Response should not be empty"


def test_automatic_translation(aibrary: AiBrary):
    def _inner_fun(model: Model):
        return aibrary.translation(
            text="HI",
            model=model.model_name,
            source_language="en",
            target_language="fa",
        )

    models = aibrary.get_all_models(filter_category="translation")
    assert len(models) > 0, "There is no model!!!"
    results = []
    for model in models:
        try:
            results.append(_inner_fun(model))
        except Exception as e:
            results.append(e)

    for response in results:
        if isinstance(response, Exception):
            print(f"An error occurred: {response}")
            continue
        assert response["text"], "Response should not be empty"


def test_audio_speech_creation(aibrary: AiBrary):
    def _inner_fun(model: Model):
        try:
            return (
                aibrary.audio.speech.create(
                    input="Hey Cena",
                    model=model.model_name,
                    response_format="mp3",
                    voice="alloy",
                ),
                model,
            )
        except Exception as e:
            return (e, model)

    models = aibrary.get_all_models(filter_category="tts")
    assert len(models) > 0, "There is no model!!!"
    results = []
    for model in models:
        results.append(_inner_fun(model))

    error = []
    for response_model in results:
        response, model = response_model
        if isinstance(response, Exception):
            error.append(f"An error occurred: {model} - {response}")
            continue
        if response:
            assert response.content, "Audio content should not be empty"
            with open(f"file-{model.model_name}.mp3", "wb") as output_file:
                output_file.write(response.content)
        else:
            error.append(f"No audio content generated for model: {model.model_name}")
    if len(error):
        raise AssertionError("\n".join(error))


@pytest.mark.asyncio
async def test_image_generation(aibrary: AiBrary):
    def _inner_fun(model_name: str):
        try:
            return aibrary.images.generate(
                model=model_name, size="256x256", prompt="Draw cat"
            )
        except Exception as e:
            return e

    models = aibrary.get_all_models(filter_category="image")
    models = list(set([model.model_name for model in models]))
    assert len(models) > 0, "There is no model!!!"
    results = []
    for model in models:
        results.append(_inner_fun(model))

    errors = []
    for response in results:
        if isinstance(response, Exception):
            errors.append(f"An error occurred: {response}")
            continue
        assert response, "Response should not be empty"

    if errors:
        raise AssertionError("\n".join(errors))
