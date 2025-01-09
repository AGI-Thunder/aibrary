from typing import Tuple

from aibrary import AiBrary, Model


def intro():
    import streamlit as st

    st.markdown(
        """
    <h1 style="display: flex; align-items: center;">
        Welcome to
        <img src="https://www.aibrary.dev/_next/static/media/logo.26501b30.svg" alt="Logo" style="margin-left: 10px; width: 200px;">
        👋
    </h1>
    """,
        unsafe_allow_html=True,
    )

    if "api_key" not in st.session_state:
        st.session_state["api_key"] = ""

    def update_api_key(new_value):
        st.session_state["api_key"] = new_value

    st.title("🔑 API Key Manager")
    st.subheader("Securely manage your API key")

    main_api_key = st.text_input(
        "Enter your API Key",
        value=st.session_state["api_key"],
        on_change=lambda: update_api_key(st.session_state["api_key"]),
    )

    if main_api_key != st.session_state["api_key"]:
        update_api_key(main_api_key)
        st.rerun()


def generate_markdown_for_models(model) -> str:
    import streamlit as st

    markdown = []
    markdown.append(f"### Model: {model.model_name}")
    markdown.append(f"- **Provider**: {model.provider}")
    markdown.append(f"- **Category**: {model.category}")
    if model.quality:
        markdown.append(f"- **Quality**: {model.quality}")
    if model.size:
        markdown.append(f"- **Size**: {model.size}")

    markdown.append("\n#### Pricing Information")
    markdown.append("| Unit Type | Price Per Input Unit | Price Per Output Unit |")
    markdown.append("|-----------|-----------------------|------------------------|")
    for pricing in model.ai_models_pricing:
        markdown.append(
            f"| {pricing.unit_type} | ${pricing.price_per_input_unit:.6f} | ${pricing.price_per_output_unit:.6f} |"
        )
    with st.expander(f"See {model.model_name} info"):
        st.markdown(f"{"\n".join(markdown)}")


def sidebar() -> Tuple["Model", "AiBrary"]:
    from collections import defaultdict

    import streamlit as st

    from aibrary import AiBrary

    with st.sidebar:
        if aibrary_api_key := st.text_input(
            "AiBrary API Key",
            key="aibrary_api_key",
            value=st.session_state["api_key"],
            type="password",
        ):
            st.session_state["api_key"] = aibrary_api_key

            aibrary = AiBrary(api_key=aibrary_api_key) if aibrary_api_key else AiBrary()
            categories = sorted(
                {item.category for item in aibrary.get_all_models()} - {"chat"}
            )

            categories.insert(0, "chat")
            category_name = st.selectbox("Choose a category", categories)
            if category_name == "intro":
                return None, None
            models = {
                f"{item.model_name}"
                + (f"-{item.size}" if item.size is not None else "")
                + (f",{item.quality}" if item.quality is not None else ""): item
                for item in aibrary.get_all_models(filter_category=category_name)
            }
            if category_name == "chat":
                models.update(
                    {
                        f"{item.model_name}"
                        + (f"-{item.size}" if item.size is not None else "")
                        + (f",{item.quality}" if item.quality is not None else ""): item
                        for item in aibrary.get_all_models(filter_category="multimodal")
                    }
                )
            grouped = defaultdict(list)

            # Group data by the specified field
            for name, obj in models.items():
                grouped[obj.provider].append(name)

            def create_options_with_separator(grouped_dict, separator="🤖"):
                options = []
                for group, items in grouped_dict.items():
                    # Add separator for the group with its name
                    options.append(f"{separator}{group}")
                    # Add items from the group
                    options.extend(items)
                if options and options[-1] == separator:
                    options.pop()  # Remove the last separator if it exists
                return options

            # Create options with separator
            options = create_options_with_separator(grouped)

            # Display the dropdown
            model_name = st.selectbox("Select an option", options, index=1)

            # Handle selection
            if model_name.startswith("🤖"):
                st.warning("This is just a provider, please select a model.")

            return models[model_name], aibrary
    return None, None


def chat_category(model: "Model", aibrary: "AiBrary"):
    import streamlit as st

    st.session_state.setdefault("messages_data", [])
    title_with_clearBtn("🧠 Chat", ["messages_data"])

    for message in st.session_state.messages_data:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter your message:"):
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
                st.session_state.messages_data.append(
                    {"role": "assistant", "content": response}
                )
            except Exception as e:
                st.error(f"Error: {e}")


def multimodal_category(model: "Model", aibrary: "AiBrary"):

    import streamlit as st
    from PIL import Image
    from utils.file_convertor import (
        decode_file,
        encode_file,
        prepare_audio,
        prepare_image,
    )

    isAudio = "audio" in model.model_name
    st.session_state.setdefault("multimodal_data", [])
    st.session_state.setdefault("next_prompt_data", [])
    st.session_state.setdefault("multimodal_file_uploader_key", 0)
    st.session_state.setdefault("multimodal_audio_input_key", 1)
    title_with_clearBtn("🖼️🎤📝 Multimodal", ["multimodal_data", "next_prompt_data"])
    for message in st.session_state.multimodal_data:
        if "type" in message:
            with st.chat_message(message["role"]):
                if (tmessage := message.get("type")) is not None and tmessage == "text":
                    st.markdown(message["content"])
                elif tmessage == "image":
                    st.image(decode_file(message["content"]))
                elif tmessage == "audio":
                    st.audio(decode_file(message["content"]))
    if isAudio:
        voice_input = st.audio_input(
            "Record your voice",
            key=st.session_state["multimodal_audio_input_key"],
        )
    else:
        voice_input = None
    uploaded_file = st.file_uploader(
        "Upload an image or audio file",
        type=["jpg", "png", "jpeg", "mp3", "wav"],
        key=st.session_state["multimodal_file_uploader_key"],
    )
    if uploaded_file or voice_input:
        with st.chat_message("user"):
            if uploaded_file:
                if uploaded_file.type.startswith("image"):
                    image_file = uploaded_file.read()
                    image = Image.open(uploaded_file)
                    st.image(
                        image,
                        caption="Uploaded Image",
                    )
                    st.session_state.multimodal_data.append(
                        {
                            "role": "user",
                            "type": "image",
                            "content": encode_file(image_file),
                        }
                    )
                    st.session_state.next_prompt_data.append(
                        prepare_image(uploaded_file.type, image_file)
                    )
                if uploaded_file.type.startswith("audio"):
                    voice_input = uploaded_file
            if voice_input:
                audio_file = voice_input.read()
                st.audio(
                    voice_input,
                )
                st.session_state.multimodal_data.append(
                    {
                        "role": "user",
                        "type": "audio",
                        "content": encode_file(audio_file),
                    }
                )
                st.session_state.next_prompt_data.append(
                    prepare_audio(voice_input.type, audio_file)
                )
            st.session_state["multimodal_file_uploader_key"] += 1
            st.session_state["multimodal_audio_input_key"] += 1

    if prompt := st.chat_input("Describe the file or ask a question:") or voice_input:
        if not voice_input and prompt:

            st.session_state.multimodal_data.append(
                {"role": "user", "type": "text", "content": prompt}
            )
            st.session_state.next_prompt_data.append(
                {"role": "user", "content": prompt}
            )
            with st.chat_message("user"):
                st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                response = aibrary.chat.completions.create(
                    model=f"{model.model_name}@{model.provider}",
                    messages=st.session_state.next_prompt_data,
                    modalities=["text", "audio"] if isAudio else None,
                    audio={"voice": "alloy", "format": "mp3"} if isAudio else None,
                )
                if isAudio and response.choices[0].message.audio:
                    response_audio = response.choices[0].message.audio.data
                    response_transcript = response.choices[0].message.audio.transcript
                    audio_response = decode_file(response_audio)
                    st.audio(audio_response)
                    st.markdown(response_transcript)
                    st.session_state.multimodal_data.append(
                        {
                            "role": "assistant",
                            "type": "audio",
                            "content": response_audio,
                        }
                    )
                    st.session_state.multimodal_data.append(
                        {
                            "role": "assistant",
                            "type": "text",
                            "content": response_transcript,
                        }
                    )
                    st.session_state.next_prompt_data.append(
                        {
                            "role": "assistant",
                            "type": "text",
                            "content": response_transcript,
                        }
                    )

                else:
                    response = response.choices[0].message.content

                    st.markdown(response)
                    st.session_state.multimodal_data.append(
                        {"role": "assistant", "type": "text", "content": response}
                    )
            except Exception as e:
                st.error(f"Error: {e}")

        st.rerun()


def image_category(model: "Model", aibrary: "AiBrary"):

    import streamlit as st
    from openai.types.images_response import ImagesResponse
    from utils.file_convertor import decode_file

    st.title()
    st.session_state.setdefault("image_data", [])
    title_with_clearBtn("🖼️ Image Generation", ["image_data"])
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


def ocr_category(model: "Model", aibrary: "AiBrary"):
    import streamlit as st
    from PIL import Image
    from utils.draw_box_ocr import draw_box_ocr
    from utils.file_convertor import decode_file, encode_file

    st.subheader("Optical Character Recognition")
    st.session_state.setdefault("ocr_data", [])
    title_with_clearBtn("🖼️ OCR", ["ocr_data"])

    for message in st.session_state.ocr_data:
        with st.chat_message(message["role"]):
            if message["type"] == "json":
                st.code(message["content"], language="json", wrap_lines=True)
            elif message["type"] == "image":
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
            with st.chat_message("user"):
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
            with st.chat_message("assistant"):

                try:
                    response_file = draw_box_ocr(image_file, response)
                    st.image(response_file)
                    st.session_state.ocr_data.extend(
                        [
                            {
                                "role": "assistant",
                                "type": "image",
                                "content": encode_file(response_file),
                            },
                        ]
                    )

                except:
                    st.warning("Draw box failed for this image")
                finally:
                    st.code(response.model_dump(), language="json", wrap_lines=True)
                    st.code(response.text, language="md", wrap_lines=True)
                    st.session_state.ocr_data.extend(
                        [
                            {
                                "role": "assistant",
                                "type": "json",
                                "content": response.model_dump(),
                            },
                            {
                                "role": "assistant",
                                "type": "text",
                                "content": response.text,
                            },
                        ]
                    )


def object_detection_category(model: "Model", aibrary: "AiBrary"):

    import streamlit as st
    from PIL import Image
    from utils.draw_box_object_detection import draw_box_object_detection
    from utils.file_convertor import decode_file, encode_file

    st.session_state.setdefault("object_detection_data", [])
    title_with_clearBtn("🖼️ Object Detection", ["object_detection_data"])

    for message in st.session_state.object_detection_data:
        with st.chat_message(message["role"]):
            if message["type"] == "image":
                st.image(decode_file(message["content"]))
            elif message["type"] == "json":
                st.code(message["content"], language="json", wrap_lines=True)
            else:
                st.markdown(message["content"])

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
            try:
                response_file = draw_box_object_detection(image_file, response)
                st.image(response_file)
                st.session_state.object_detection_data.extend(
                    {
                        "role": "assistant",
                        "type": "image",
                        "content": encode_file(response_file),
                    },
                )
            except:
                pass
            finally:
                st.session_state.object_detection_data.extend(
                    {
                        "role": "assistant",
                        "type": "json",
                        "content": response.model_dump(),
                    },
                )


def stt_category(model: "Model", aibrary: "AiBrary"):

    import streamlit as st
    from utils.file_convertor import decode_file, encode_file

    st.session_state.setdefault("stt_data", [])
    title_with_clearBtn("🎤📝 Speech to Text", ["stt_data"])

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


def tts_category(model: "Model", aibrary: "AiBrary"):

    import streamlit as st
    from utils.file_convertor import decode_file, encode_file

    st.session_state.setdefault("tts_data", [])
    title_with_clearBtn("📝🎤 Text to Speech", ["tts_data"])

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


def translation_category(model: "Model", aibrary: "AiBrary"):

    import streamlit as st

    st.session_state.setdefault("translation_data", [])
    title_with_clearBtn("🌎 Translation", ["translation_data"])

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
            index=1,
        )

    # Display selected languages
    st.write(f"Source Language: {languages[source_language]} ({source_language})")
    st.write(
        f"Destination Language: {languages[destination_language]} ({destination_language})"
    )

    for message in st.session_state.translation_data:
        with st.chat_message(message["role"]):
            st.code(message["content"], language="md", wrap_lines=True)

    if prompt := st.chat_input("Write something you'd like to translate!"):
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


def page_router(demo_name: str, model: "Model", aibrary: "AiBrary"):
    import streamlit as st

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
        st.markdown("## 🚧 This category is under development.")
        st.markdown("## 🚧 We are working on it...")


def title_with_clearBtn(title: str, key: list | str):

    import streamlit as st

    col1, col2 = st.columns(
        spec=[0.7, 0.2],
        vertical_alignment="center",
        gap="large",
    )

    with col1:
        st.title(title)

    with col2:
        if isinstance(key, str):
            key = [key]

        if st.button("Clear All", icon="🗑"):
            for item in key:
                st.session_state.get(item, {}).clear()
            st.rerun()


try:
    model, aibrary = sidebar()
    generate_markdown_for_models(model)
except:
    model, aibrary = None, None

demo_name = model.category if model else "intro"
page_router(demo_name, model, aibrary)
