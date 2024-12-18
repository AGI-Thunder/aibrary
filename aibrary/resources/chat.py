from typing import override

import openai
import openai.resources


class AibraryChatCompletion(openai.resources.chat.completions.Completions):
    @override
    def create(self, system: str = None, **kwargs):
        # For Anthropic we need to pass system as a seperate argument, not as a role in message argument.
        if system is not None:
            kwargs["extra_body"] = {**kwargs.get("extra_body", {}), "system": system}
        return super().create(**kwargs)
