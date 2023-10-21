import os
import shutil
import sys
import time
from abc import ABC, abstractmethod
from http.client import InvalidURL

import streamlit as st
import torch
from auto_gptq import AutoGPTQForCausalLM
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms import CTransformers
from langchain.llms import HuggingFacePipeline, HuggingFaceHub
from langchain.memory import ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from loguru import logger as log
from transformers import AutoTokenizer, TextStreamer, pipeline
from youtube_transcript_api import TranscriptsDisabled

from text_extraction.pdf_extractor import PDFTextExtractor
from text_extraction.text_file_extractor import TextFileExtractor
from text_extraction.url_extractor import URLTextExtractor
from text_extraction.youtube_extractor import YouTubeTextExtractor
from utils.helpers import has_internet_connection

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(DEVICE)

# %%: Configuration for the app
model_config = {
    'max_new_tokens': 4096, 'temperature': 0.1, 'top_p': 0.95,
    'repetition_penalty': 1.15, 'context_length': 4096
}

session_state_defaults = {
    "uploaded_files": None,
    "conversation": None,
    "my_chat_history": [],
    "n_urls": 1,
    "urls": [],
    "text_area_input": "",
    "youtube_url": ""
}

input_options_rg = ["Upload PDFs", "Enter URLs", "Enter text", "YouTube Video"]

model_options = ['Mistral-7B (CPU only)', 'Llama-2-13B-chat-GPTQ (GPU required)',
                 'Llama-2-13B-chat-GGML (CPU only)', 'HuggingFace Hub (Online)', 'OpenAI API (Online)']


# %%: Abstract class for language models
class LanguageModel(ABC):
    @abstractmethod
    def get_llm(self):
        pass


class HuggingFaceModel(LanguageModel):
    def get_llm(self):
        return HuggingFaceHub(repo_id="google/flan-t5-xxl", config=model_config)


class OpenAIModel(LanguageModel):
    def get_llm(self):
        return ChatOpenAI(config=model_config)


class MistralModel(LanguageModel):
    def get_llm(self):
        model_path = 'models/mistral-7b-instruct-v0.1.Q4_K_M.gguf'
        return CTransformers(model=model_path, model_type='mistral', device=DEVICE, do_sample=True, config=model_config)


class GgmlModel(LanguageModel):
    def get_llm(self):
        model_path = 'models/llama-2-13b-chat.Q4_K_M.gguf'
        return CTransformers(model=model_path, model_type='llama', device=DEVICE, do_sample=True, config=model_config)


class GptqModel(LanguageModel):
    def get_llm(self):
        model_name = 'TheBloke/Llama-2-13B-chat-GPTQ'
        model_basename = "model"
        model = AutoGPTQForCausalLM.from_quantized(model_name, revision="main", model_basename=model_basename,
                                                   use_safetensors=True, trust_remote_code=True,
                                                   inject_fused_attention=False,
                                                   device=DEVICE, quantize_config=None)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        text_pipeline = pipeline("text2text-generation",
                                 model=model,
                                 tokenizer=tokenizer,
                                 max_new_tokens=4096,
                                 temperature=0.1,
                                 top_p=0.95,
                                 do_sample=True,
                                 repetition_penalty=1.15,
                                 streamer=streamer)
        return HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0.1})


class ConversationChainFactory:
    @staticmethod
    def get_conversation_chain(vectorstore, language_model: LanguageModel):
        llm = language_model.get_llm()
        memory = ConversationBufferWindowMemory(k=1, memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
            memory=memory,
            chain_type="stuff",
            verbose=True,
        )
        return conversation_chain


# %%
def show_temp_success_message(message: str, delay: int):
    """
    Create an empty container with a success message and empty it after a delay.

    Parameters:
    message (str): The success message to display in the container.
    Delay (int): The time in seconds to wait before clearing the container.

    Returns:
    None
    """
    container = st.empty()
    container.success(message)
    time.sleep(delay)
    container.empty()


# %%
def save_uploaded_files(uploaded_files):
    # Delete the uploaded_files folder if it exists
    shutil.rmtree('uploaded_files') if os.path.isdir('uploaded_files') else None

    # Create the uploaded_files folder and the subfolders for the different file types
    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1]
        target_folder = os.path.join('uploaded_files', file_extension.replace('.', ''))

        if not os.path.isdir('uploaded_files'):
            os.mkdir('uploaded_files')
        if not os.path.isdir(target_folder):
            os.mkdir(target_folder)

        target_file_path = os.path.join(target_folder, uploaded_file.name)

        # Save the uploaded file to the target folder
        with open(target_file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())

        # show a success message for 2 seconds and then hide it.
        show_temp_success_message(f"File uploaded successfully: {uploaded_file.name}", 2)


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    # check if the text is a string
    if isinstance(text, str):
        chunks = text_splitter.split_text(text)
    else:
        chunks = text_splitter.split_documents(text)

    return chunks


def get_vectorstore(text_chunks):
    vectorstore = None
    model_name = "BAAI/bge-small-en"
    model_kwargs = {'device': DEVICE}
    encode_kwargs = {'normalize_embeddings': True}  # set True to compute cosine similarity
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    st.write(f"Loaded embeddings model: {text_chunks}")

    if text_chunks:
        # check if the text is a list of strings or a list of lists of strings
        if isinstance(text_chunks, list) and isinstance(text_chunks[0], str):
            vectorstore = FAISS.from_texts(text_chunks, embeddings)
        else:
            vectorstore = FAISS.from_documents(text_chunks, embeddings)
    else:
        st.error(f'Your input is empty: {len(text_chunks)}. Please use a different input and try again.')

    return vectorstore


def handle_userinput(user_question, container):
    # Send the user question to the conversation chain and get the response
    response = st.session_state.conversation({'question': user_question})
    log.debug(f'Response: {response}')

    # Get the question from the response
    st.session_state.my_chat_history.append(response['question'])
    log.debug(f'{response["question"] = }')

    # Get the helpful answer from the response
    helpful_answer = response['answer'].split('Helpful Answer:')[-1]
    st.session_state.my_chat_history.append(helpful_answer)
    log.debug(f'{helpful_answer = }')

    # Update the chat_history in the memory with the new helpful answer
    log.debug(f"Before: {st.session_state.conversation.memory.chat_memory.messages[-1].content = }")
    st.session_state.conversation.memory.chat_memory.messages[-1].content = helpful_answer
    log.debug(f"After: {st.session_state.conversation.memory.chat_memory.messages[-1].content = }")

    # Display the response in the chat
    for i, message in enumerate(st.session_state.my_chat_history):
        sender = "human" if i % 2 == 0 else "assistant"
        container.chat_message(sender).write(message)


def select_language_model(model_options_spinner):
    if model_options_spinner == 'Llama-2-13B-chat-GPTQ (GPU required)':
        return GptqModel()
    elif model_options_spinner == 'Llama-2-13B-chat-GGML (CPU only)':
        return GgmlModel()
    elif model_options_spinner == 'HuggingFace Hub (Online)':
        return HuggingFaceModel()
    elif model_options_spinner == 'OpenAI API (Online)':
        return OpenAIModel()
    elif model_options_spinner == 'Mistral-7B (CPU only)':
        return MistralModel()
    else:
        return None


def create_conversation_chain(vectorstore, language_model):
    if language_model:
        st.session_state.conversation = ConversationChainFactory.get_conversation_chain(vectorstore, language_model)


def clear_cache():
    keys = list(st.session_state.keys())
    for key in keys:
        st.session_state.pop(key)


def process_text(text, model_options_spinner):
    with st.spinner("Processing"):
        start_time = time.time()
        text_chunks = get_text_chunks(text)
        vectorstore = get_vectorstore(text_chunks)
        language_model = select_language_model(model_options_spinner)
        create_conversation_chain(vectorstore=vectorstore, language_model=language_model)
        show_temp_success_message(f"Processing done!\n Time taken: {round(time.time() - start_time, 2)} Seconds", 5)


def render_input_ui(input_option):
    """
    Render the input UI based on the selected input option.
    The input options are:
        1. Upload PDFs
        2. Enter URLs
        3. Enter text
        4. YouTube Video

    Parameters
    ----------
    input_option : str
        The selected input option from the radio group in the sidebar.

    Returns
    -------
    None (Updates the session state variables)

    """
    if input_option == "Upload PDFs":
        st.subheader("Your documents")
        uploaded_files = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        st.session_state.uploaded_files = uploaded_files

    elif input_option == "Enter URLs":
        st.subheader("Enter URLs")
        if st.button(label="add"):
            st.session_state.n_urls += 1
            st.experimental_rerun()

        urls_list = []
        for i in range(st.session_state.n_urls):
            urls_list.append(st.text_input("", placeholder=f"URL {i + 1}", label_visibility="collapsed"))
            print(urls_list)

        if st.button(label="remove"):
            if st.session_state.n_urls > 1:
                st.session_state.n_urls -= 1
                st.experimental_rerun()

        st.session_state.urls = urls_list

    elif input_option == 'Enter text':
        st.subheader("Enter text")
        st.session_state.text_area_input = st.text_area("Enter your text here", height=200)

    elif input_option == 'YouTube Video':
        st.subheader("YouTube Video")
        st.session_state.youtube_url = st.text_input("Enter a YouTube video URL or ID:")


# %%: Functions to get raw text from different sources
def get_raw_text_from_youtube_video():
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


def get_raw_text(input_option):
    raw_text = None

    if input_option == "Upload PDFs":
        raw_text = get_raw_text_from_pdfs()
    elif input_option == "Enter URLs":
        raw_text = get_raw_text_from_urls()
    elif input_option == 'Enter text':
        raw_text = st.session_state.text_area_input
        if not raw_text:
            st.warning("Please enter some text first")
    elif input_option == 'YouTube Video':
        raw_text = get_raw_text_from_youtube_video()

    return raw_text


# %%:
def initialize_session_state_defaults():
    for variable, default_value in session_state_defaults.items():
        if variable not in st.session_state:
            st.session_state[variable] = default_value


def main():
    load_dotenv()
    st.set_page_config(page_title="",
                       page_icon=":books:")

    # Define the state of the app
    initialize_session_state_defaults()

    st.header("Chat with your documents")
    chat_body = st.container()
    chat_body.markdown("---")

    # Create a toggle button to show/hide the chat history object in session state
    if st.session_state.conversation:
        view_messages = st.expander("View the chain's chat history object", expanded=False)
        with view_messages:
            st.write(st.session_state.my_chat_history)

        # Add a button to clear the chat history and the conversation chain
        st.button("Clear chat history", on_click=clear_cache)

    with st.sidebar:
        # create a radio group for the different input options
        st.subheader("Input options")
        input_option = st.radio("", input_options_rg, on_change=clear_cache)

        # create divider to separate the input options from the rest
        st.markdown("---")

        # render the input UI based on the selected input option
        render_input_ui(input_option)

        # create divider to separate the input options from the rest
        st.markdown("---")

        # Create a radio group for the different models
        st.subheader("Select a Model")
        model_options_spinner = st.selectbox("", model_options)

        # create divider to separate the input options from the rest
        st.markdown("---")

        if st.button("Process", use_container_width=True):
            raw_text = get_raw_text(input_option=input_option)
            if raw_text:
                process_text(text=raw_text, model_options_spinner=model_options_spinner)

    # Create a form to get the user question
    with st.form(key='my_form', clear_on_submit=True):
        user_question = st.text_input(label="Ask a question about your documents:")
        submit_button = st.form_submit_button(label='Submit')

    if (user_question or submit_button) and st.session_state.conversation:
        handle_userinput(user_question, chat_body)

    elif user_question and not st.session_state.conversation:
        st.error("Please upload your PDFs first and click on 'Process'")


if __name__ == '__main__':
    main()
