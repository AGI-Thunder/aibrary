from aibrary import AiBrary
aibrary = AiBrary(api_key=None)
aibrary.images.generate(model="dall-e-2", size="256x256", prompt="Draw cat")
