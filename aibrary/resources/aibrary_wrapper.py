import os

import openai

from aibrary.resources.translation import TranslationClient


class AiBrary(openai.OpenAI):
    """Base class for OpenAI modules."""

    def __init__(self, api_key: str = None):
        """Initialize with API key and base URL."""
        if api_key is None:
            api_key = os.environ.get("AIBRARY_API_KEY")
        if api_key is None:
            raise ValueError(
                "The api_key client option must be set either by passing api_key to the client or by setting the AIBRARY_API_KEY environment variable"
            )
        self.api_key = api_key
        self.base_url = f"https://api.aibrary.dev/v0"
        openai.api_key = api_key
        openai.base_url = self.base_url

        """Initialize all modules."""
        self.translation = TranslationClient(api_key=api_key, base_url=self.base_url)
        super().__init__(api_key=self.api_key, base_url=self.base_url)
