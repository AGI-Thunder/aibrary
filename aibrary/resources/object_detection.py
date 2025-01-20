import mimetypes
import os

import httpx

from aibrary.schemas.object_detection import ObjectDetectionResponse


class ObjectDetectionClient:
    """Client for interacting with the Object Detection API."""

    def __init__(self, *, base_url: str, api_key: str):
        """
        Initialize the ObjectDetectionClient.

        :param base_url: The base URL of the Object Detection API
        :param api_key: The Bearer token for authorization.
        """
        self.base_url = base_url
        self.headers = {"authorization": f"Bearer {api_key}"}

    async def process_object_detection_async(
        self,
        providers: str,
        file: str | bytes = None,
        file_name: str = None,
        file_url: str = None,
        settings: str = None,
        response_as_dict: bool = True,
        attributes_as_list: bool = False,
        show_base_64: bool = True,
        show_original_response: bool = False,
    ) -> ObjectDetectionResponse:
        """Internal method to process Object Detection requests."""
        if not (file or file_url):
            raise ValueError("Either 'file' or 'file_url' must be provided.")

        if file and file_url:
            raise ValueError("Provide only one: either 'file' or 'file_url'.")

        url = f"{self.base_url}images/object_detection"
        data = {
            "providers": providers,
            "settings": settings,
            "response_as_dict": response_as_dict,
            "attributes_as_list": attributes_as_list,
            "show_base_64": show_base_64,
            "show_original_response": show_original_response,
        }

        files = None
        if file:
            if isinstance(file, str):  # Treat as a file path
                if not os.path.isfile(file):
                    raise ValueError(f"File path does not exist: {file}")
                file_name = os.path.basename(file)
                mime_type, _ = mimetypes.guess_type(file_name)
                if mime_type is None:
                    mime_type = "application/octet-stream"
                with open(file, "rb") as f:
                    files = {"file": (file_name, f.read(), mime_type)}
            elif isinstance(file, bytes):  # Treat as file content
                if not file_name:
                    raise ValueError(
                        "When passing file as bytes, 'file_name' must be provided."
                    )
                mime_type, _ = mimetypes.guess_type(file_name)
                if mime_type is None:
                    mime_type = "application/octet-stream"
                files = {"file": (file_name, file, mime_type)}
            else:
                raise TypeError(
                    "Invalid type for 'file'. Expected str (path) or bytes."
                )

        async with httpx.AsyncClient() as client:
            if files:
                response = await client.post(
                    url,
                    data=data,
                    files=files,
                    headers=self.headers,
                )
            else:
                data["file_url"] = file_url
                response = await client.post(
                    url,
                    data=data,
                    headers=self.headers,
                )
        return ObjectDetectionResponse(**response.json())

    def process_object_detection(
        self,
        providers: str,
        file: str | bytes = None,
        file_name: str = None,
        file_url: str = None,
        settings: str = None,
        response_as_dict: bool = True,
        attributes_as_list: bool = False,
        show_base_64: bool = True,
        show_original_response: bool = False,
    ) -> ObjectDetectionResponse:
        """Internal method to process Object Detection requests."""
        if not (file or file_url):
            raise ValueError("Either 'file' or 'file_url' must be provided.")

        if file and file_url:
            raise ValueError("Provide only one: either 'file' or 'file_url'.")

        url = f"{self.base_url}images/object_detection"
        data = {
            "providers": providers,
            "settings": settings,
            "response_as_dict": response_as_dict,
            "attributes_as_list": attributes_as_list,
            "show_base_64": show_base_64,
            "show_original_response": show_original_response,
        }

        files = None
        if file:
            if isinstance(file, str):  # Treat as a file path
                if not os.path.isfile(file):
                    raise ValueError(f"File path does not exist: {file}")
                file_name = os.path.basename(file)
                mime_type, _ = mimetypes.guess_type(file_name)
                if mime_type is None:
                    mime_type = "application/octet-stream"
                with open(file, "rb") as f:
                    files = {"file": (file_name, f.read(), mime_type)}
            elif isinstance(file, bytes):  # Treat as file content
                if not file_name:
                    raise ValueError(
                        "When passing file as bytes, 'file_name' must be provided."
                    )
                mime_type, _ = mimetypes.guess_type(file_name)
                if mime_type is None:
                    mime_type = "application/octet-stream"
                files = {"file": (file_name, file, mime_type)}
            else:
                raise TypeError(
                    "Invalid type for 'file'. Expected str (path) or bytes."
                )

        with httpx.Client() as client:
            if files:
                response = client.post(
                    url,
                    data=data,
                    files=files,
                    headers=self.headers,
                )
            else:
                data["file_url"] = file_url
                response = client.post(
                    url,
                    data=data,
                    headers=self.headers,
                )
        return ObjectDetectionResponse(**response.json())
