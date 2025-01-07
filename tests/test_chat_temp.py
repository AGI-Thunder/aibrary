import asyncio

import pytest

from aibrary import AsyncAiBrary
from aibrary.resources.models import Model

models = [
    "together/llama-3-8b-instruct-lite",
    "together/Llama-3.1-Nemotron-70B-Instruct-HF",
    "together/mixtral-8x22b-instruct-v0.1",
    "together/mixtral-8x7b-instruct-v0.1",
    "together/llama-3.1-405b-instruct-turbo",
    "together/llama-3-8b-instruct-turbo",
    "together/gemma-2b-it",
    "together/mistral-7b-instruct-v0.1",
    "together/mistral-7b-instruct-v0.2",
    "fake/fake-model-chat",
    "together/dbrx-instruct",
    "together/llama-3-70b-instruct-turbo",
    "together/SOLAR-10.7B-Instruct-v1.0",
    "together/Nous-Hermes-2-Mixtral-8x7B-DPO",
    "together/Qwen2-72B-Instruct",
    "together/llama-3-70b-instruct-lite",
    "together/Qwen2.5-72B-Instruct-Turbo",
    "deepinfra/gemma-2-27b-it",
    "together/deepseek-llm-67b-chat",
    "deepinfra/mixtral-8x7b-instruct-v0.1",
    "deepinfra/QwQ-32B-Preview",
    "together/llama-3.2-3b-instruct-turbo",
    "deepinfra/llama-3.3-70b-instruct-turbo",
    "replicate/llama-3-8b-chat",
    "mistral-ai/mixtral-8x7b-instruct-v0.1",
    "groq/llama-3-8b-chat",
    "aws-bedrock/llama-3-70b-chat",
    "aws-bedrock/llama-3-8b-chat",
    "lepton-ai/llama-3-70b-chat",
    "lepton-ai/llama-3-8b-chat",
    "aws-bedrock/mistral-7b-instruct-v0.2",
    "fireworks-ai/mixtral-8x22b-instruct-v0.1",
    "replicate/llama-3-70b-chat",
    "mistral-ai/mixtral-8x22b-instruct-v0.1",
    "groq/llama-3-70b-chat",
]


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
        await asyncio.sleep(index * 1.5)
        return await aibrary.chat.completions.create(
            model=f"{model.model_name}@{model.provider}",
            messages=[
                {"role": "user", "content": "what is 2+2? return only a number."},
            ],
        )

    models = await aibrary.get_all_models(filter_category="chat")
    models = [model for model in models if model.provider == "together"]
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
