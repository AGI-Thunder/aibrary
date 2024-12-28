import asyncio

from aibrary import AiBrary


def main():
    aibrary = AiBrary()
    aibrary.object_detection(
        providers="amazon",
        file=open("tests/assets/ocr-test.jpg", "rb").read(),
        file_name="test.jpg",
    )
    aibrary.object_detection(
        providers="amazon",
        file_url="https://builtin.com/sites/www.builtin.com/files/styles/ckeditor_optimize/public/inline-images/5_python-ocr.jpg",
    )
    aibrary.object_detection(
        providers="fake-model-object-detection", file="tests/assets/ocr-test.jpg"
    )


asyncio.run(main())
