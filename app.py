# Importing the necessary libraries
import time
from http.client import InvalidURL

import streamlit as st
import torch
from dotenv import load_dotenv
from loguru import logger as log
from youtube_transcript_api import TranscriptsDisabled

from conversation.conversation_chain import create_conversation_chain
from embedding.text_processing import get_text_chunks
from embedding.vector_store import get_vectorstore
from language_models.language_models import get_language_model
from text_extraction.pdf_extractor import PDFTextExtractor
from text_extraction.text_file_extractor import TextFileExtractor
from text_extraction.url_extractor import URLTextExtractor
from text_extraction.youtube_extractor import YouTubeTextExtractor
from utils.helpers import has_internet_connection, save_uploaded_files

# %%: Set the device to GPU if available
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
log.info(f"Using device: {DEVICE}")

# %%: Configuration for the app
session_state_defaults = {
    "uploaded_files": None,
    "conversation": None,
    "my_chat_history": [],
    "n_urls": 1,
    "urls": [],
    "text_area_input": "",
    "youtube_url": ""
}

data_source_options = ["Upload Documents", "Web Links", "Manually Enter Text", "YouTube Videos"]

model_options = ['Mistral-7B (CPU)', 'Llama-2-13B GPTQ (GPU)',
                 'Llama-2-13B GGUF (CPU)', 'HuggingFace API (Online)', 'OpenAI API (Online)']


# %%: Functions to get raw text from different sources
def get_raw_text_from_youtube_video():
    """
    Retrieves the raw text from a YouTube Video Link.

    Returns
    -------
    raw_text: str
        The extracted text from the given YouTube Video Link.

    Raises
    ------
    TranscriptsDisabled
        If the transcript is not available for the given video.
    Exception
        If an unexpected error occurs during the extraction process.

    Warnings
    --------
    - If there is no internet connection.
    - If the given YouTube video URL or ID is invalid.
    - If the transcript is not available for the given video.
    - If an unexpected error occurs during the extraction process.
    """
    if not has_internet_connection():
        st.warning("Please check your internet connection and try again.")
        return

    if not st.session_state.youtube_url:
        st.warning("Please enter a YouTube video URL or ID first")
        return

    youtube_extractor = YouTubeTextExtractor()
    try:
        return youtube_extractor.extract_text(st.session_state.youtube_url)
    except TranscriptsDisabled:
        log.warning("Transcript is not available for this video or this video is not available.")
        st.warning(f"Transcript is not available for this video or this video is not available.")
    except Exception as e:
        log.error(f"Something went wrong.\nError: {e}")
        st.warning(f"Something went wrong. Please try again later.\nError: {e}")


def get_raw_text_from_urls():
    """
    Retrieve raw text from a list of URLs.

    Returns
    -------
    extracted_text: str
        The extracted text from the given URLs.

    Raises
    ------
    InvalidURL
        If the given URLs are invalid.
    Exception
        If an unexpected error occurs during the extraction process.

    Warnings
    --------
    - If there is no internet connection.
    - If the given URLs are empty.
    - If the given URLs are invalid.
    - If an unexpected error occurs during the extraction process.
    """
    urls = st.session_state.urls

    if not has_internet_connection():
        st.warning("Please check your internet connection and try again.")
        return

    if not urls or all(url == "" for url in urls):
        st.warning("Please enter at least one URL")
        return

    try:
        extractor = URLTextExtractor(urls)
        extracted_text = extractor.extract_text_from_urls()
    except InvalidURL:
        log.warning("URLs are invalid, please check your input and try again.")
        st.warning("URLs are invalid, please check your input and try again.")
    except Exception as e:
        log.error(f"Something went wrong.\nError: {e}")
        st.warning(f"Something went wrong. Please try again later.\nError: {e}")
    else:
        return extracted_text


def get_raw_text_from_pdfs():
    """
    Retrieve raw text from a list of PDFs.

    Returns
    -------
    raw_text: str
        The extracted text from the given PDFs.

    Raises
    ------
    Exception
        If an unexpected error occurs during the extraction process.

    Warnings
    --------
    - If no files are uploaded.
    - If the file type is not supported.
    """
    uploaded_files = st.session_state.uploaded_files

    if not uploaded_files:
        st.warning("Please upload your files first and click on 'Process'")
        return

    save_uploaded_files(uploaded_files)
    raw_text = []
    for doc in uploaded_files:
        if doc.type == 'application/pdf':
            pdf_extractor = PDFTextExtractor()
            raw_text.extend(pdf_extractor.extract_text())
        elif doc.type == 'text/plain' or doc.type == 'application/octet-stream':
            txt_extractor = TextFileExtractor()
            raw_text.extend(txt_extractor.extract_text(doc.name))
        else:
            st.warning(f"File type not supported yet {doc.name}")

    return raw_text


def get_raw_text_from_text_area():
    """
    Retrieve raw text from the text area.

    Returns
    -------
    raw_text: str
        The text from the text area.

    Warnings
    --------
    - If the text area is empty.
    """
    raw_text = st.session_state.text_area_input
    if not raw_text:
        st.warning("Please enter some text first")

    return raw_text


def get_raw_text(selected_data_source):
    """
    Retrieve raw text from different sources based on the selected input option.

    Parameters
    ----------
    selected_data_source: str
        The selected input option from the radio group in the sidebar.

    Returns
    -------
    raw_text: str
        The extracted text from the given input.

    Warnings
    --------
    - If no text found in the input.
    """
    input_mapper = {
        "Upload Documents": get_raw_text_from_pdfs,
        "Web Links": get_raw_text_from_urls,
        "Manually Enter Text": get_raw_text_from_text_area,
        "YouTube Videos": get_raw_text_from_youtube_video
    }

    raw_text = input_mapper[selected_data_source]()

    if not raw_text:
        log.warning("No text found in the input to process.")

    return raw_text


# %%: Process the input text and create a conversation chain
def process_text(text, model_options_spinner) -> None:
    """
    Process the input text and create a conversation chain.
    Also, save the conversation chain in the session state,
    and display a success message with the processing time.

    Parameters
    ----------
    text: str | list of documents
        The input text to be processed.
    model_options_spinner: str
        The selected model option for language models.

    Returns
    -------
    None (Updates the conversation chain in the session state, displays a success message)
    """
    # Starting spinner widget to show the processing status
    with st.spinner("Processing"):
        # Marking the start time
        start_time = time.time()

        # Splitting text into chunks
        text_chunks = get_text_chunks(text)

        # Creating a vectorstore from text chunks
        vectorstore = get_vectorstore(text_chunks)

        # Getting the language model
        language_model = get_language_model(model_options_spinner)

        # Creating a conversation chain and saving it in the session state
        st.session_state.conversation = create_conversation_chain(
            vectorstore=vectorstore,
            language_model=language_model
        )

        # Displaying a temporary success message with processing time, disappears after 5 seconds
        message = f"Processing done!\n Time taken: {round(time.time() - start_time, 2)} Seconds"
        show_temp_success_message(message, 5)


# %%: Functions to handle the user input (questions)
def handle_userinput(user_question, container):
    """
    Handles the user's input question and updates relevant components accordingly.

    Parameters
    ----------
    user_question : str
        User's input question to be processed.
    container : UI_Container
        User interface container that will display the response.
    """
    response = get_response(user_question)
    log.debug(f'Response: {response}')

    helpful_answer = get_helpful_answer(response)
    update_chat_history(question=user_question, helpful_answer=helpful_answer)

    update_memory(helpful_answer)
    render_response_to_ui(container)


def get_response(user_question):
    """Send the user question to the conversation chain and get the response."""
    return st.session_state.conversation({'question': user_question})


def update_chat_history(question, helpful_answer):
    """Update the chat history."""
    st.session_state.my_chat_history.append(question)
    st.session_state.my_chat_history.append(helpful_answer)


def get_helpful_answer(response):
    """Extract the helpful answer from the response."""
    helpful_answer = response['answer'].split('Helpful Answer:')[-1]
    log.success(f'{helpful_answer = }')
    return helpful_answer


def update_memory(helpful_answer):
    """Update the chat_history in the memory with the new helpful answer."""
    st.session_state.conversation.memory.chat_memory.messages[-1].content = helpful_answer


def render_response_to_ui(container):
    """Display the response in the chat."""
    for i, message in enumerate(st.session_state.my_chat_history):
        sender = "human" if i % 2 == 0 else "assistant"
        container.chat_message(sender).write(message)


# %%: Handle the session state variables
def initialize_session_state_defaults():
    """Initializes the default values for the session state variables."""
    for variable, default_value in session_state_defaults.items():
        if variable not in st.session_state:
            st.session_state[variable] = default_value


def clear_cache():
    """Clears the session state variables."""
    keys = list(st.session_state.keys())
    for key in keys:
        st.session_state.pop(key)


# %%: Functions to render the input UI
def show_temp_success_message(message: str, delay: int):
    """
    Create an empty container with a success message and empty it after a delay.

    Parameters
    -----------
    message: str
        The success message to be displayed.
    delay: int
        The delay in seconds after which the container will be emptied.

    Returns
    -------
    None (Displays a success message and then empties the container after a delay)
    """
    container = st.empty()
    container.success(message)
    time.sleep(delay)
    container.empty()


def render_upload_input():
    """Renders a file uploader widget allowing the user to upload multiple files."""
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

    # Store uploaded files in Streamlit session state.
    st.session_state.uploaded_files = uploaded_files


def manage_url_count():
    """Manages the count of URL input fields based on the user's interactions with 'add' and 'remove' buttons."""
    # Increase the count of URL fields by one when 'add' button is clicked.
    if st.button(label="add"):
        st.session_state.n_urls += 1
        st.rerun()

    # Decrease the count of URL fields by one when the 'remove' button is clicked.
    if st.button(label="remove"):
        if st.session_state.n_urls > 1:
            st.session_state.n_urls -= 1
            st.rerun()


def render_urls_input():
    """Renders URL input fields according to the URL count managed by `manage_url_count` function."""
    st.subheader("Enter Web Link or more")

    # Generate a list of URLs
    urls_list = [st.text_input("URL", placeholder=f"URL {i + 1}", label_visibility="collapsed")
                 for i in range(st.session_state.n_urls)]

    manage_url_count()

    # Store URL list in Streamlit session state.
    st.session_state.urls = urls_list


def render_text_input():
    """Renders a text area input field where the user can Manually Enter Text."""
    st.subheader("Manually Enter Text")

    # Store text area input in Streamlit session state.
    st.session_state.text_area_input = st.text_area("Enter your text here", height=200)


def render_youtube_input():
    """Renders a text input field where the user can enter a YouTube video URL."""
    st.subheader("Enter YouTube Video Link")

    # Store YouTube URL in Streamlit session state.
    st.session_state.youtube_url = st.text_input("Enter a YouTube video Link or ID:")


def render_input_ui(input_option):
    """
    Render the input UI based on the selected input option.
    The input options are:
        1. Upload Documents
        2. Web Links
        3. Manually Enter Text
        4. YouTube Videos

    Parameters
    ----------
    input_option : str
        The selected input option from the radio group in the sidebar.

    Returns
    -------
    None (Updates the session state variables)
    """

    option_mapper = {
        "Upload Documents": render_upload_input,
        "Web Links": render_urls_input,
        "Manually Enter Text": render_text_input,
        "YouTube Videos": render_youtube_input
    }

    option_mapper[input_option]()


# %%: Main function
def main():
    load_dotenv()
    st.set_page_config(page_title="AI Chat Assistant",
                       page_icon=":speech_balloon:")

    # Define the state of the app
    initialize_session_state_defaults()

    st.header("Start a conversation with Your Data 🦜")
    chat_body = st.container()
    chat_body.markdown("---")

    # Create a toggle button to show/hide the chat history object in session state
    if st.session_state.conversation:
        view_messages = st.expander("Show chat history object", expanded=False)
        with view_messages:
            st.write(st.session_state.my_chat_history)

        # Add a button to clear the chat history and the conversation chain
        st.button("Clear Chat History", on_click=clear_cache)

    with st.sidebar:
        # create a radio group for the different input options
        st.subheader("Select Your Data Source")
        selected_data_source = st.selectbox("Data Source", data_source_options, on_change=clear_cache)

        # create divider to separate the input options from the rest
        st.markdown("---")

        # render the input UI based on the selected input option
        render_input_ui(selected_data_source)

        # create divider to separate the input options from the rest
        st.markdown("---")

        # Create a radio group for the different models
        st.subheader("Choose a Language Model")
        model_options_spinner = st.selectbox("Language Model", model_options)

        # create divider to separate the input options from the rest
        st.markdown("---")

        if st.button("Process", use_container_width=True):
            raw_text = get_raw_text(selected_data_source=selected_data_source)
            process_text(text=raw_text, model_options_spinner=model_options_spinner)

    # Create a form to get the user question
    with st.form(key='my_form', clear_on_submit=True):
        user_question = st.text_input(label="Ask a Question About Your Data:")
        submit_button = st.form_submit_button(label='Ask')

    if (user_question or submit_button) and st.session_state.conversation:
        handle_userinput(user_question, chat_body)

    elif user_question and not st.session_state.conversation:
        st.error("Please upload your PDFs first and click on 'Process'")


if __name__ == '__main__':
    main()
