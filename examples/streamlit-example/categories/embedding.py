from aibrary import AiBrary, Model


def embedding_category(embedding_model: "Model", aibrary: "AiBrary"):

    import streamlit as st
    from utils.rag.old_rag import SimpleRAGSystem
    from utils.render_model_option import render_model_option
    from utils.title_with_btn import title_with_clearBtn

    st.session_state.setdefault("rag_data", {})
    st.session_state.setdefault("rag_message_data", [])
    st.session_state.setdefault("rag_file_uploader_key", 0)
    title_with_clearBtn(
        "🌎 RAG",
        [
            "rag_data",
            "rag_message_data",
        ],
    )
    for message in st.session_state.rag_message_data:
        with st.chat_message(message["role"]):
            st.code(message["content"], language="md", wrap_lines=True)

    # Dropdowns for source and destination languages
    uploaded_file = st.file_uploader(
        "Upload a PDF file", type=["pdf"], key=st.session_state["rag_file_uploader_key"]
    )
    question = st.chat_input("Ask a Question")
    with st.sidebar:
        models, model_name = render_model_option(
            aibrary, "chat", selectbox_title="Select an embedding model"
        )
        st.success(model_name)
        st.success(embedding_model.model_name)
    chat_model = models[model_name]
    rag = SimpleRAGSystem(
        aibrary=aibrary,
        model_name=f"{chat_model.model_name}@{chat_model.provider}",
        embedding_model=f"{embedding_model.model_name}@{embedding_model.provider}",
        embeddings=st.session_state.rag_data,
    )

    if not uploaded_file and not rag.embeddings:
        st.warning("Please provide both a question and upload a PDF file.")
    else:
        if uploaded_file:
            st.session_state.rag_data.clear()
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())

            pages = rag.load_pdf(uploaded_file.name)
            rag.create_embeddings(pages)
            st.success("PDF processed and embeddings created!")
            st.info("Processing PDF...")
            st.session_state["rag_file_uploader_key"] += 1
        if question and rag.embeddings:
            try:
                with st.chat_message("user"):
                    st.code(question, language="md", wrap_lines=True)

                with st.spinner("Finding the answer..."):
                    answer = rag.ask_question(question)

                with st.chat_message("assistant"):
                    st.code(answer, language="md", wrap_lines=True)

                st.session_state.rag_message_data.append(
                    {"role": "user", "content": question}
                )
                st.session_state.rag_message_data.append(
                    {"role": "assistant", "content": answer}
                )
            except Exception as e:
                st.error(f"An error occurred while processing the PDF: {e}")
