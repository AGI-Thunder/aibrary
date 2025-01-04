import asyncio

from aibrary import AiBrary


def main():
    aibrary = AiBrary()
    return aibrary.translation("HI", "phedone", "en", "fa")


asyncio.run(main())
