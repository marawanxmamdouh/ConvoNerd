 # Importing the necessary libraries
import time
from http.client import InvalidURL
from typing import Any, Callable

import streamlit as st
import torch
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.vectorstores.faiss import FAISS
from loguru import logger as log
from streamlit.runtime.uploaded_file_manager import UploadedFile
from youtube_transcript_api import TranscriptsDisabled

from conversation.conversation_chain import create_conversation_chain
from embedding.text_processing import get_text_chunks
from embedding.vector_store import get_vectorstore
from language_models.language_models import get_language_model, LanguageModel
from text_extraction.pdf_extractor import PDFTextExtractor
from text_extraction.text_file_extractor import TextFileExtractor
from text_extraction.url_extractor import URLTextExtractor
from text_extraction.youtube_extractor import YouTubeTextExtractor
from utils.helpers import has_internet_connection, save_uploaded_files

# %%: Set the device to GPU if available, else use CPU
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
log.info(f"Using device: {DEVICE}")

# %%: Configuration for the app's session state defaults
# These are the initial values stored in the session state
session_state_defaults: dict[str, Any] = {
    "uploaded_files": None,
    "conversation": None,
    "my_chat_history": [],
    "n_urls": 1,
    "urls": [],
    "text_area_input": "",
    "youtube_url": ""
}

# List of available data source options
data_source_options: list[str] = ["Upload Documents", "Web Links", "Manually Enter Text", "YouTube Videos"]

# List of available language models for the user to choose from
model_options: list[str] = ['Zephyr-7B (CPU)', 'Llama-2-7B GPTQ (GPU)',
                            'Llama-2-13B GGUF (CPU)', 'HuggingFace API (Online)', 'OpenAI API (Online)']


# %%: Functions to get raw text from different sources
def get_raw_text_from_youtube_video() -> str | None:
    """
    Retrieves the raw text (transcript) from a YouTube Video Link.

    Returns
    -------
    raw_text: str | None
        The extracted transcript from the given YouTube video.

    Raises
    ------
    TranscriptsDisabled
        If the transcript is not available for the video.
    Exception
        If any other error occurs during the extraction process.
    """
    # Check if the internet connection is active
    if not has_internet_connection():
        st.warning("Please check your internet connection and try again.")
        return None

    # Check if a YouTube URL has been provided
    if not st.session_state.youtube_url:
        st.warning("Please enter a YouTube video URL or ID first")
        return None

    # Initialize the YouTube extractor
    youtube_extractor: YouTubeTextExtractor = YouTubeTextExtractor()
    try:
        # Try to extract the transcript from the video
        return youtube_extractor.extract_text(st.session_state.youtube_url)
    except TranscriptsDisabled:
        log.warning("Transcript is not available for this video or this video is not available.")
        st.warning(f"Transcript is not available for this video or this video is not available.")
    except Exception as e:
        log.error(f"Something went wrong.\nError: {e}")
        st.warning(f"Something went wrong. Please try again later.\nError: {e}")


def get_raw_text_from_urls() -> list[Document] | None:
    """
    Retrieves raw text from a list of URLs.

    Returns
    -------
    extracted_text: list[Document] | None
        The extracted text from the provided URLs.

    Raises
    ------
    InvalidURL
        If the provided URLs are invalid.
    Exception
        For any unexpected errors during the extraction process.
    """
    urls: list[str] = st.session_state.urls

    # Check if the internet connection is active
    if not has_internet_connection():
        st.warning("Please check your internet connection and try again.")
        return None

    # Check if any URLs have been provided
    if not urls or all(url == "" for url in urls):
        st.warning("Please enter at least one URL")
        return None

    try:
        # Initialize URL extractor and extract text from provided URLs
        extractor: URLTextExtractor = URLTextExtractor(urls)
        extracted_text: list[Document] = extractor.extract_text_from_urls()
    except InvalidURL:
        log.warning("URLs are invalid, please check your input and try again.")
        st.warning("URLs are invalid, please check your input and try again.")
    except Exception as e:
        log.error(f"Something went wrong.\nError: {e}")
        st.warning(f"Something went wrong. Please try again later.\nError: {e}")
    else:
        return extracted_text


def get_raw_text_from_docs() -> list[str] | None:
    """
    Retrieve raw text from uploaded documents.

    Returns
    -------
    raw_text: list[str]
        The extracted text from the uploaded documents.

    Warnings
    --------
    - Displays a warning if no files are uploaded or if unsupported file types are uploaded.
    """
    uploaded_files: UploadedFile | list[UploadedFile] = st.session_state.uploaded_files

    # If no files are uploaded, warn the user
    if not uploaded_files:
        st.warning("Please upload your files first and click on 'Process'")
        return None

    # Save the uploaded files locally
    save_uploaded_files(uploaded_files)
    raw_text: list[str | Document] = []

    # Extract text based on file type
    for doc in uploaded_files:
        if doc.type == 'application/pdf':
            pdf_extractor: PDFTextExtractor = PDFTextExtractor()
            raw_text.extend(pdf_extractor.extract_text())
        elif doc.type == 'text/plain' or doc.type == 'application/octet-stream':
            txt_extractor: TextFileExtractor = TextFileExtractor()
            raw_text.extend(txt_extractor.extract_text(doc.name))
        else:
            st.warning(f"File type not supported yet {doc.name}")

    return raw_text


def get_raw_text_from_text_area() -> str | None:
    """
    Retrieve raw text from the manually entered text area.

    Returns
    -------
    raw_text: str | None
        The text entered by the user in the text area.

    Warnings
    --------
    - Displays a warning if no text is entered.
    """
    raw_text = st.session_state.text_area_input
    if not raw_text:
        st.warning("Please enter some text first")
        return None

    return raw_text


def get_raw_text(selected_data_source: str) -> str | list[str | Document] | None:
    """
    Retrieve raw text from different sources based on the selected input option.

    Parameters
    ----------
    selected_data_source: str
        The selected input option from the sidebar.

    Returns
    -------
    raw_text: str | list[str | Document] | None
        The extracted text based on the input option selected.
    """
    # Map the input options to their respective functions
    input_mapper: dict[str, Callable] = {
        "Upload Documents": get_raw_text_from_docs,
        "Web Links": get_raw_text_from_urls,
        "Manually Enter Text": get_raw_text_from_text_area,
        "YouTube Videos": get_raw_text_from_youtube_video
    }

    # Retrieve raw text using the corresponding function
    raw_text: str | list[str | Document] | None = input_mapper[selected_data_source]()

    if not raw_text:
        log.warning("No text found in the input to process.")
        return None

    return raw_text


# %%: Process the input text and create a conversation chain
def process_text(text: str | list[str | Document], model_options_spinner: str) -> None:
    """
    Process the input text and create a conversation chain using the selected language model.

    Parameters
    ----------
    text: str | list[str | Document]
        The input text to be processed.
    model_options_spinner: str
        The selected model option for language models.

    Returns
    -------
    None (Updates the conversation chain in the session state, displays a success message)
    """
    # Starting spinner widget to show the processing status
    with st.spinner("Processing"):
        # Mark the start time to calculate processing time
        start_time: float = time.time()

        # Split the text into smaller chunks
        text_chunks: list[str] | list[Document] = get_text_chunks(text)

        # Create a FAISS vector store from the text chunks
        vectorstore: FAISS = get_vectorstore(text_chunks)

        # Load the selected language model
        language_model: LanguageModel = get_language_model(model_options_spinner)

        # Create a conversation chain and store it in session state
        st.session_state.conversation = create_conversation_chain(
            vectorstore=vectorstore,
            language_model=language_model
        )

        # Display a success message with the time taken for processing
        message: str = f"Processing done!\n Time taken: {round(time.time() - start_time, 2)} Seconds"
        show_temp_success_message(message, 5)
        log.success(message)


# %%: Handle the user's question and update the chat UI
def handle_userinput(user_question: str, container) -> None:
    """
    Handle the user's input question, get a response from the conversation chain, and update the UI.

    Parameters
    ----------
    user_question : str
        User's question or input.
    container : st.container
        The Streamlit container where the conversation is displayed.

    Returns
    -------
    None (Updates chat history and the conversation display)
    """
    # Initialize chat history if not present in session state
    if "my_chat_history" not in st.session_state:
        st.session_state.my_chat_history: list[tuple[str, str]] = []

    # Get the conversation chain from session state
    conversation = st.session_state.conversation

    # If no conversation chain exists, warn the user to process the text first
    if not conversation:
        st.warning("Please upload/process your documents first")
        return

    # Get the response for the user's question from the conversation chain
    response = conversation.predict(input=user_question)

    # Update the chat history in session state
    st.session_state.my_chat_history.append((user_question, response))

    # Display the conversation in the container
    with container:
        for i, (question, answer) in enumerate(st.session_state.my_chat_history):
            message(question, is_user=True, key=str(i) + "_user")
            message(answer, key=str(i))
