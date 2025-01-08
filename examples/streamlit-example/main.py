from typing import Tuple

import streamlit as st
from PIL import Image
from utils.file_convertor import encode_file

from aibrary import AiBrary, Model


def intro():
    st.write("# Welcome to AiBrary! 👋")
    st.write("## Your gateway to the world of AI at your fingertips. 🌟")
    st.markdown(
        """
        # Snippet Code
        ## Sync Requests
        ```python
        from aibrary import AiBrary
        aibrary = AiBrary(api_key=None) # either passing api_key to the client or setting the AIBRARY_API_KEY in environment variable
        ```
        ### Get All Models
        ```python
        aibrary.get_all_models(filter_category="TTS",return_as_objects=False)
        ```
        ### OpenAI Completion Models
        ```python
        aibrary.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "How are you today?"},
                {"role": "assistant", "content": "I'm doing great, thank you!"},
            ],
            temperature=0.7,
        )
        ```
    """
    )


def sidebar() -> Tuple[Model, AiBrary]:
    with st.sidebar:
        aibrary_api_key = st.text_input(
            "AiBrary API Key",
            key="aibrary_api_key",
            type="password",
        )
        aibrary = AiBrary(api_key=aibrary_api_key) if aibrary_api_key else AiBrary()
        categories = sorted(
            {item.category for item in aibrary.get_all_models()} - {"chat"}
        )

        categories.insert(0, "chat")
        category_name = st.selectbox("Choose a category", categories)
        if category_name == "intro":
            return None, None
        models = {
            f"{item.model_name}@{item.provider}"
            + (f"-{item.size}" if item.size is not None else "")
            + (f",{item.quality}" if item.quality is not None else ""): item
            for item in aibrary.get_all_models(filter_category=category_name)
        }
        model_name = st.selectbox("Choose a model", models.keys())

    return models[model_name], aibrary


def chat_category(model: Model, aibrary: AiBrary):
    st.title("🧠 Chat")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("messages_data", [])

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter your message:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages_data.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                response = aibrary.chat.completions.create(
                    model=f"{model.model_name}@{model.provider}",
                    messages=st.session_state.messages_data,
                    stream=True,
                )
                response = st.write_stream(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
                st.session_state.messages_data.append(
                    {"role": "assistant", "content": response}
                )
            except Exception as e:
                st.error(f"Error: {e}")


def multimodal_category(model: Model, aibrary: AiBrary):
    from utils.file_convertor import decode_file, prepare_audio, prepare_image

    isAudio = "audio" in model.model_name
    st.title("🖼️🎤📝 Multimodal")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("messages_data", [])

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    uploaded_file = st.file_uploader(
        "Upload an image or audio file", type=["jpg", "png", "jpeg", "mp3", "wav"]
    )
    if uploaded_file:
        if uploaded_file.type.startswith("image"):
            image_file = uploaded_file.read()
            image = Image.open(uploaded_file)
            st.image(
                image,
                caption="Uploaded Image",
            )
            st.session_state.messages_data.append(
                prepare_image(uploaded_file.type, image_file)
            )
        elif uploaded_file.type.startswith("audio"):
            st.audio(
                uploaded_file,
            )
            st.session_state.messages_data.append(
                prepare_audio(uploaded_file.type, uploaded_file.read())
            )

    if prompt := st.chat_input("Describe the file or ask a question:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages_data.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                response = aibrary.chat.completions.create(
                    model=f"{model.model_name}@{model.provider}",
                    messages=st.session_state.messages_data,
                    modalities=["text", "audio"] if isAudio else None,
                    audio={"voice": "alloy", "format": "mp3"} if isAudio else None,
                )
                if isAudio:
                    audio_response = decode_file(response.choices[0].message.audio.data)
                    st.audio(audio_response)
                    response = response.choices[0].message.audio.transcript

                else:
                    response = response.choices[0].message.content
                st.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

            except Exception as e:
                st.error(f"Error: {e}")


def image_category(model: Model, aibrary: AiBrary):
    from openai.types.images_response import ImagesResponse
    from utils.file_convertor import decode_file

    st.title("🖼️ Image Generation")
    st.session_state.setdefault("image_data", [])

    for message in st.session_state.image_data:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.image(decode_file(message["content"]))
            else:
                st.markdown(message["content"])

    if prompt := st.chat_input("What you want to draw?"):
        st.session_state.image_data.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                response: ImagesResponse = aibrary.images.generate(
                    model=f"{model.model_name}",
                    prompt=prompt,
                    size=model.size,
                    quality=model.quality,
                    response_format="b64_json",
                    n=1,
                )
                if len(response.data) == 0:
                    st.error("🔴 No image generated")
                    return
                st.session_state.image_data.append(
                    {"role": "assistant", "content": response.data[0].b64_json}
                )
                st.image(decode_file(response.data[0].b64_json))

            except Exception as e:
                st.error(f"Error: {e}")


def ocr_category(model: Model, aibrary: AiBrary):
    from utils.draw_box_ocr import draw_box_ocr
    from utils.file_convertor import decode_file, encode_file

    st.title("🖼️ OCR")
    st.subheader("Optical Character Recognition")
    st.session_state.setdefault("ocr_data", [])

    for message in st.session_state.ocr_data:
        with st.chat_message(message["role"]):
            if message["type"] == "image":
                st.image(decode_file(message["content"]))
            else:
                st.code(message["content"], language="md", wrap_lines=True)

    uploaded_file = st.file_uploader(
        "Upload an image or audio file", type=["jpg", "png", "jpeg"]
    )
    if uploaded_file:
        if uploaded_file.type.startswith("image"):
            image_file = uploaded_file.read()
            image = Image.open(uploaded_file)
            st.image(
                image,
                caption="Uploaded Image",
            )
            st.session_state.ocr_data.append(
                {"role": "user", "type": "image", "content": encode_file(image_file)}
            )
            response = aibrary.ocr(
                providers=model.model_name,
                file=image_file,
                file_name=uploaded_file.name,
            )
            response_file = draw_box_ocr(image_file, response)
            st.image(response_file)
            st.code(response.text, language="md", wrap_lines=True)
            st.session_state.ocr_data.extend(
                [
                    {
                        "role": "user",
                        "type": "image",
                        "content": encode_file(response_file),
                    },
                    {"role": "user", "type": "text", "content": response.text},
                ]
            )


def object_detection_category(model: Model, aibrary: AiBrary):
    from utils.draw_box_object_detection import draw_box_object_detection
    from utils.file_convertor import decode_file, encode_file

    st.title("🖼️ Object Detection")
    st.session_state.setdefault("object_detection_data", [])

    for message in st.session_state.object_detection_data:
        with st.chat_message(message["role"]):
            if message["type"] == "image":
                st.image(decode_file(message["content"]))
            else:
                st.chat(message["content"])

    uploaded_file = st.file_uploader(
        "Upload an image or audio file", type=["jpg", "png", "jpeg"]
    )
    if uploaded_file:
        if uploaded_file.type.startswith("image"):
            image_file = uploaded_file.read()
            image = Image.open(uploaded_file)
            st.image(
                image,
                caption="Uploaded Image",
            )
            st.session_state.object_detection_data.append(
                {"role": "user", "type": "image", "content": encode_file(image_file)}
            )
            response = aibrary.object_detection(
                providers=model.model_name,
                file=image_file,
                file_name=uploaded_file.name,
            )
            response_file = draw_box_object_detection(image_file, response)
            st.image(response_file)
            st.session_state.object_detection_data.extend(
                {
                    "role": "user",
                    "type": "image",
                    "content": encode_file(response_file),
                },
            )


def stt_category(model: Model, aibrary: AiBrary):
    from utils.file_convertor import decode_file, encode_file

    st.title("🎤📝 Speech to Text")
    st.session_state.setdefault("stt_data", [])

    for message in st.session_state.stt_data:
        with st.chat_message(message["role"]):
            if message["type"] == "audio":
                st.audio(decode_file(message["content"]))
            else:
                st.code(message["content"], language="md", wrap_lines=True)
    voice_input = st.audio_input(
        "Record your voice",
    )
    uploaded_file = st.file_uploader("OR Upload an audio file", type=["mp3", "wav"])
    if uploaded_file or voice_input:
        if uploaded_file and uploaded_file.type.startswith("audio"):
            voice_input = uploaded_file

        # if voice_input.type.startswith("audio"):
        audio_file = voice_input.read()
        with st.chat_message("user"):
            st.audio(
                voice_input,
            )

        response = aibrary.audio.transcriptions.create(
            model=model.model_name, file=voice_input
        )
        with st.chat_message("assistant"):
            st.code(response.text, language="md", wrap_lines=True)

        st.session_state.stt_data.extend(
            [
                {
                    "role": "user",
                    "type": "audio",
                    "content": encode_file(audio_file),
                },
                {
                    "role": "assistant",
                    "type": "text",
                    "content": response.text,
                },
            ]
        )


def tts_category(model: Model, aibrary: AiBrary):
    from utils.file_convertor import decode_file

    st.title("📝🎤 Text to Speech")
    st.session_state.setdefault("tts_data", [])

    for message in st.session_state.tts_data:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.audio(decode_file(message["content"]))
            else:
                st.markdown(message["content"])

    if prompt := st.chat_input("Write something you'd like me to say!"):
        st.session_state.tts_data.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                response = aibrary.audio.speech.create(
                    input=prompt,
                    model=model.model_name,
                    response_format="mp3",
                    voice="FEMALE" if model.provider != "openai" else "alloy",
                )
                response = response.read()
                st.audio(response)

                st.session_state.tts_data.append(
                    {"role": "assistant", "content": encode_file(response)}
                )

            except Exception as e:
                st.error(f"Error: {e}")


def translation_category(model: Model, aibrary: AiBrary):

    st.title("🌎 Translation")
    st.session_state.setdefault("translation_data", [])
    languages = {
        "en": "English",
        "fa": "Persian",
        "af": "Afrikaans",
        "ar": "Arabic",
        "bn": "Bengali",
        "bg": "Bulgarian",
        "ca": "Catalan",
        "zh": "Chinese",
        "hr": "Croatian",
        "cs": "Czech",
        "da": "Danish",
        "nl": "Dutch",
        "et": "Estonian",
        "fi": "Finnish",
        "fr": "French",
        "de": "German",
        "el": "Greek",
        "he": "Hebrew",
        "hi": "Hindi",
        "hu": "Hungarian",
        "id": "Indonesian",
        "it": "Italian",
        "ja": "Japanese",
        "ko": "Korean",
        "lv": "Latvian",
        "lt": "Lithuanian",
        "ms": "Malay",
        "no": "Norwegian",
        "pl": "Polish",
        "pt": "Portuguese",
        "ro": "Romanian",
        "ru": "Russian",
        "sk": "Slovak",
        "sl": "Slovenian",
        "es": "Spanish",
        "sv": "Swedish",
        "th": "Thai",
        "tr": "Turkish",
        "uk": "Ukrainian",
        "vi": "Vietnamese",
    }

    # Dropdowns for source and destination languages
    col1, col2 = st.columns(2)

    with col1:
        source_language = st.selectbox(
            "Source Language",
            options=list(languages.keys()),
            format_func=lambda x: languages[x],
        )

    with col2:
        destination_language = st.selectbox(
            "Destination Language",
            options=list(languages.keys()),
            format_func=lambda x: languages[x],
        )

    # Display selected languages
    st.write(f"Source Language: {languages[source_language]} ({source_language})")
    st.write(
        f"Destination Language: {languages[destination_language]} ({destination_language})"
    )

    for message in st.session_state.translation_data:
        with st.chat_message(message["role"]):
            st.code(message["content"], language="md", wrap_lines=True)

    if prompt := st.chat_input("Write something you'd like me to say!"):
        st.session_state.translation_data.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                response = aibrary.translation(
                    source_language=source_language,
                    target_language=destination_language,
                    text=prompt,
                    model=model.model_name,
                )
                st.code(response.text, language="md", wrap_lines=True)

                st.session_state.translation_data.append(
                    {"role": "assistant", "content": response.text}
                )

            except Exception as e:
                st.error(f"Error: {e}")


def page_router(demo_name: str, model: Model, aibrary: AiBrary):
    if demo_name == "intro":
        intro()
    elif demo_name == "chat":
        chat_category(model, aibrary)
    elif demo_name == "multimodal":
        multimodal_category(model, aibrary)
    elif demo_name == "image":
        image_category(model, aibrary)
    elif demo_name == "ocr":
        ocr_category(model, aibrary)
    elif demo_name == "object detection":
        object_detection_category(model, aibrary)
    elif demo_name == "stt":
        stt_category(model, aibrary)
    elif demo_name == "tts":
        tts_category(model, aibrary)
    elif demo_name == "translation":
        translation_category(model, aibrary)
    else:
        st.markdown("🚧 This category is under development.")
        st.markdown("we are working on it...")


try:
    model, aibrary = sidebar()
except:
    model, aibrary = None, None
demo_name = model.category if model else "intro"
if st.button("Clear All"):
    st.session_state.clear()
    st.cache_data.clear()
    st.rerun()

page_router(demo_name, model, aibrary)
