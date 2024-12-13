# from openai.resources import *


import httpx


class TranslationClient:
    def __init__(self, base_url: str, api_key: str):
        """
        Initialize the TranslationClient.

        :param base_url: The base URL of the translation API
        :param api_key: The Bearer token for authorization.
        """
        self.base_url = base_url
        self.headers = {"authorization": f"Bearer {api_key}"}

    async def automatic_translation_async(
        self, text, model, source_language, target_language
    ):
        """
        Perform automatic translation.

        :param text: The text to be translated.
        :param model: The model to use for translation (e.g., 'phedone').
        :param source_language: The source language code (e.g., 'en').
        :param target_language: The target language code (e.g., 'fa').
        :return: The translated text or an error message.
        """
        payload = {
            "text": text,
            "model": model,
            "source_language": source_language,
            "target_language": target_language,
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}translation/automatic_translation",
                json=payload,
                headers=self.headers,
            )

        # Raise an error if the response status is not successful
        response.raise_for_status()

        # Return the JSON response
        return response.json()

    def automatic_translation(self, text, model, source_language, target_language):
        """
        Perform automatic translation.

        :param text: The text to be translated.
        :param model: The model to use for translation (e.g., 'phedone').
        :param source_language: The source language code (e.g., 'en').
        :param target_language: The target language code (e.g., 'fa').
        :return: The translated text or an error message.
        """
        payload = {
            "text": text,
            "model": model,
            "source_language": source_language,
            "target_language": target_language,
        }
        with httpx.Client() as client:
            response = client.post(
                f"{self.base_url}translation/automatic_translation",
                json=payload,
                headers=self.headers,
            )

        # Raise an error if the response status is not successful
        response.raise_for_status()

        # Return the JSON response
        return response.json()
