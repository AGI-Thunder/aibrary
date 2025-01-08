import base64


def encode_file(file: bytes):
    return base64.b64encode(file).decode("utf-8")


def decode_file(file: str) -> bytes:
    return base64.b64decode(file)


def prepare_image(file_type: str, file: bytes):

    base64_file = encode_file(file)

    return {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:{file_type};base64,{base64_file}"},
            },
        ],
    }


def prepare_audio(file_type: str, file: bytes):

    base64_file = encode_file(file)

    return {
        "role": "user",
        "content": [
            {
                "type": "input_audio",
                "input_audio": {"data": base64_file, "format": file_type.split("/")[1]},
            }
        ],
    }
