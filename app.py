# app.py
import os
import tempfile
import time
import streamlit as st
from rag import ChatPDF

st.set_page_config(page_title="RAG with Local DeepSeek R1")

def clear_chat():
    st.session_state.chat_history = []
    st.rerun()
    
def read_and_save_file():
    """Handle file upload and ingestion."""

    st.session_state.chat_history = []
    st.session_state.assistant.clear()

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}..."):
            st.session_state.assistant.ingest(file_path)

        os.remove(file_path)

def chat():
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(key="chat", placeholder="What is up?"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):

            response_placeholder = st.empty()
            streamed_content = ""

            try:
                response_stream = st.session_state.assistant.ask(
                    prompt.strip(),
                    k=st.session_state["retrieval_k"],
                    score_threshold=st.session_state["retrieval_threshold"],
                )

                for chunk in response_stream:
                    streamed_content += chunk
                    response_placeholder.markdown(streamed_content)

                st.session_state.chat_history.append({"role": "assistant", "content": streamed_content})

            except ValueError as e:
                response_placeholder.markdown(f"**Error:** {str(e)}")

def page():
    """Main app page layout."""       

    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False

    if "assistant" not in st.session_state:
        st.session_state.assistant = None
    
    if "file_uploader" not in st.session_state:
        st.session_state.file_uploader = []

    if "model" not in st.session_state:
        st.session_state.model = "deepseek-r1:latest"

    if "embeddings" not in st.session_state:
        st.session_state.embeddings = "mxbai-embed-large"

    if "deepseek" not in st.session_state:
        st.session_state.deepseek = True
        
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""

    if "ingestion_spinner" not in st.session_state:
        st.session_state.ingestion_spinner = st.empty()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


    st.header("RAG with Local DeepSeek R1 / OpenAI API")
    chat()

    with st.sidebar:

        option = st.selectbox(
            "Model",
            ("deepseek-r1:latest", "gpt-4o-mini"),
            index=None,
            label_visibility=st.session_state.visibility,
            disabled=st.session_state.disabled,
        )

        if "selected_model" not in st.session_state or st.session_state.selected_model != option:
            st.session_state.selected_model = option  # Store the new selection
            st.session_state.assistant = None  # Reset assistant to force reloading

        if option == "deepseek-r1:latest":
            st.write(f"model: {option}")
            st.write("embeddings: mxbai-embed-large")
            st.session_state.deepseek = True
            st.session_state.model = option
            st.session_state.embeddings = "mxbai-embed-large"

        elif option == "gpt-4o-mini":
            st.write(f"models: {option}")
            st.write("embedding: text-embedding-3-large")
            st.session_state.model = option
            st.session_state.embeddings = "text-embedding-3-large"
            st.session_state.deepseek = False

            if not st.session_state.openai_api_key:
                api_key = st.text_input("Enter your OpenAI API Key", type="password", key="api_key")
                if st.button("Save API Key"):
                    st.session_state.deepseek = False
                    st.session_state.openai_api_key = api_key
                    clear_chat()
                else:
                    return
            else:
                if st.button("Show API Key"):
                    st.write(f'"{st.session_state.openai_api_key}"')
        else:
            st.warning("Please select a model.")
            return
        
        if st.session_state.assistant is None:
            st.session_state.assistant = ChatPDF(
                is_local=st.session_state.deepseek,
                openai_api_key=st.session_state.openai_api_key,
                llm_model=st.session_state.model,
                embedding_model=st.session_state.embeddings
            )
            read_and_save_file()
        
        # Retrieval settings
        st.subheader("Settings")
        st.session_state["retrieval_k"] = st.slider(
            "Number of Retrieved Results (k)", min_value=1, max_value=10, value=5
        )
        st.session_state["retrieval_threshold"] = st.slider(
            "Similarity Score Threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.05
        )
        if st.button("Clear conversation"):
            clear_chat()

        st.subheader("Upload Documents")
        st.file_uploader(
            "Upload a PDF document",
            type=["pdf"],
            key="file_uploader",
            on_change=read_and_save_file,
            label_visibility="collapsed",
            accept_multiple_files=True,
        )


if __name__ == "__main__":
    page()
