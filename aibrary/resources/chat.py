from typing import override

import openai
import openai.resources


class AibraryChatCompletionSync(openai.resources.chat.completions.Completions):
    def __init__(self, client):
        super().__init__(client)

    @override
    def create(self, system: str = None, **kwargs):
        # For Anthropic we need to pass system as a seperate argument, not as a role in message argument.
        if system is not None:
            kwargs["extra_body"] = {**kwargs.get("extra_body", {}), "system": system}
        return super().create(**kwargs)


class AibraryChatCompletionAsync(openai.resources.chat.completions.AsyncCompletions):
    def __init__(self, client):
        super().__init__(client)

    @override
    async def create(self, system: str = None, **kwargs):
        # For Anthropic we need to pass system as a seperate argument, not as a role in message argument.
        if system is not None:
            kwargs["extra_body"] = {**kwargs.get("extra_body", {}), "system": system}
        return await super().create(**kwargs)
