from aibrary import AiBrary, Model


def chat_category(model: "Model", aibrary: "AiBrary"):
    import streamlit as st
    from utils.title_with_btn import title_with_clearBtn

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
