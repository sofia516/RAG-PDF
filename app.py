import os
import tempfile

import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------
# Page configuration
# --------------------------------

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)


# --------------------------------
# Header
# --------------------------------

st.title("📚 PDF RAG Assistant")

st.caption(
    "Chat with your documents using "
    "FAISS retrieval and a local LLM."
)


# --------------------------------
# Initialize session state
# --------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------
# Sidebar
# --------------------------------

with st.sidebar:

    st.header("📄 Documents")

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# --------------------------------
# Load embeddings
# --------------------------------

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# --------------------------------
# Load LLM
# --------------------------------

@st.cache_resource
def load_llm():

    return ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )


embeddings = load_embeddings()
llm = load_llm()


# --------------------------------
# Process PDFs
# --------------------------------

if uploaded_files:

    current_files = tuple(
        (file.name, file.size)
        for file in uploaded_files
    )

    if (
        "processed_files" not in st.session_state
        or st.session_state.processed_files != current_files
    ):

        with st.spinner(
            "Processing your documents..."
        ):

            all_documents = []

            total_pages = 0


            # --------------------------------
            # Load PDFs
            # --------------------------------

            for uploaded_file in uploaded_files:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getvalue()
                    )

                    temp_pdf_path = temp_file.name


                loader = PyPDFLoader(
                    temp_pdf_path
                )

                documents = loader.load()

                total_pages += len(documents)


                # Add PDF name to metadata
                for document in documents:

                    document.metadata["source"] = (
                        uploaded_file.name
                    )


                all_documents.extend(
                    documents
                )


                os.remove(temp_pdf_path)


            # --------------------------------
            # Split documents
            # --------------------------------

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=700,
                chunk_overlap=100
            )

            chunks = text_splitter.split_documents(
                all_documents
            )


            # --------------------------------
            # Create FAISS
            # --------------------------------

            vectorstore = FAISS.from_documents(
                chunks,
                embeddings
            )


            # --------------------------------
            # Save state
            # --------------------------------

            st.session_state.vectorstore = vectorstore

            st.session_state.processed_files = (
                current_files
            )

            st.session_state.total_pages = (
                total_pages
            )

            st.session_state.total_chunks = (
                len(chunks)
            )

            # New documents = new conversation
            st.session_state.messages = []


        st.success(
            f"Processed {len(uploaded_files)} PDF(s)"
        )


    # --------------------------------
    # Document information
    # --------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "📚 Documents",
            len(uploaded_files)
        )

    with col2:

        st.metric(
            "📄 Pages",
            st.session_state.total_pages
        )

    with col3:

        st.metric(
            "🧩 Chunks",
            st.session_state.total_chunks
        )


    st.divider()


    # --------------------------------
    # Vector database
    # --------------------------------

    vectorstore = st.session_state.vectorstore


    # --------------------------------
    # Retriever
    # --------------------------------

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 20
        }
    )


    # --------------------------------
    # Display chat history
    # --------------------------------

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )


    # --------------------------------
    # Chat input
    # --------------------------------

    question = st.chat_input(
        "Ask a question about your documents..."
    )


    if question:

        # --------------------------------
        # User message
        # --------------------------------

        with st.chat_message("user"):

            st.write(question)


        st.session_state.messages.append({
            "role": "user",
            "content": question
        })


        # --------------------------------
        # Retrieve + generate
        # --------------------------------

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching your documents..."
            ):

                documents = retriever.invoke(
                    question
                )


                # --------------------------------
                # Build context
                # --------------------------------

                context_parts = []

                for doc in documents:

                    page = doc.metadata.get(
                        "page",
                        "Unknown"
                    )

                    if isinstance(page, int):
                        page += 1

                    source = doc.metadata.get(
                        "source",
                        "Unknown PDF"
                    )

                    context_parts.append(
                        f"--- {source}, Page {page} ---\n"
                        f"{doc.page_content}"
                    )


                context = "\n\n".join(
                    context_parts
                )


                # --------------------------------
                # Conversation history
                # --------------------------------

                history_parts = []

                for message in st.session_state.messages[:-1]:

                    history_parts.append(
                        f"{message['role'].upper()}: "
                        f"{message['content']}"
                    )


                history = "\n".join(
                    history_parts
                )


                # --------------------------------
                # Prompt
                # --------------------------------

                prompt = ChatPromptTemplate.from_template("""
You are a helpful document assistant.

Answer the user's question using ONLY the
provided document context.

Use the conversation history only to understand
references in the current question.

If the answer is not present in the provided
documents, say:

"I couldn't find the answer in the provided documents."

Do not invent information.

Conversation history:
{history}

Document context:
{context}

Current question:
{question}

Answer:
""")


                messages = prompt.invoke({
                    "history": history,
                    "context": context,
                    "question": question
                })


                response = llm.invoke(
                    messages
                )


                answer = response.content


            st.write(answer)


        # --------------------------------
        # Save assistant message
        # --------------------------------

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })


        # --------------------------------
        # Sources
        # --------------------------------

        st.subheader("📄 Sources")

        unique_sources = {}

        for doc in documents:

            page = doc.metadata.get(
                "page",
                "Unknown"
            )

            if isinstance(page, int):
                page += 1

            source = doc.metadata.get(
                "source",
                "Unknown PDF"
            )

            key = (source, page)

            if key not in unique_sources:

                unique_sources[key] = (
                    doc.page_content
                )


        for (source, page), content in unique_sources.items():

            with st.expander(
                f"📄 {source} — Page {page}"
            ):

                st.caption(
                    f"Source: {source} | Page {page}"
                )

                st.write(content)


        # --------------------------------
        # Debugging
        # --------------------------------

        with st.expander(
            "🔍 Retrieved Context"
        ):

            for doc in documents:

                page = doc.metadata.get(
                    "page",
                    "Unknown"
                )

                if isinstance(page, int):
                    page += 1

                source = doc.metadata.get(
                    "source",
                    "Unknown PDF"
                )

                st.markdown(
                    f"### {source} — Page {page}"
                )

                st.write(
                    doc.page_content
                )

                st.divider()


else:

    # --------------------------------
    # Empty state
    # --------------------------------

    st.info(
        "👈 Upload one or more PDFs from the sidebar "
        "to start chatting."
    )