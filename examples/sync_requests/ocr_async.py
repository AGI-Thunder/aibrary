import asyncio

from aibrary import AiBrary


def main():
    aibrary = AiBrary()
    aibrary.ocr(
        providers="amazon",
        file=open("tests/assets/ocr-test.jpg", "rb").read(),
        file_name="test.jpg",
    )
    aibrary.ocr(providers="amazon", file="tests/assets/ocr-test.jpg")
    aibrary.ocr(
        providers="amazon",
        file_url="https://builtin.com/sites/www.builtin.com/files/styles/ckeditor_optimize/public/inline-images/5_python-ocr.jpg",
    )


asyncio.run(main())
