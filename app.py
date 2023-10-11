import os
import shutil
import sys
import time

import streamlit as st
import torch
import validators
from auto_gptq import AutoGPTQForCausalLM
from dotenv import load_dotenv
from langchain import FAISS, HuggingFacePipeline, HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFDirectoryLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger as log
from transformers import AutoTokenizer, TextStreamer, pipeline

from deal_with_urls import extract_text_from_urls

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(DEVICE)


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
    if os.path.isdir('uploaded_files'):
        shutil.rmtree('uploaded_files')

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


def get_pdf_text(pdf_docs):
    # Load the PDFs from the folder
    loder = PyPDFDirectoryLoader("uploaded_files")
    docs = loder.load()
    st.write(f"Loaded {len(docs)} PDFs")
    st.write(f'Type: {type(docs)}')
    return docs


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


def get_conversation_chain_ggml(vectorstore):
    # Download the model from https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/tree/main
    model_path = 'models/llama-2-13b-chat.Q4_K_M.gguf'

    model = CTransformers(
        model=model_path,
        model_type='llama',
        device=DEVICE,
        do_sample=True,
        config={
            'max_new_tokens': 4096,
            'temperature': 0.1,
            'top_p': 0.95,
            'repetition_penalty': 1.15,
        }
    )

    memory = ConversationBufferWindowMemory(k=1, memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        memory=memory,
        chain_type="stuff",
        verbose=True,
    )

    return conversation_chain


def get_conversation_chain_gptq(vectorstore):
    # Load your local model from disk
    model_name = 'TheBloke/Llama-2-13B-chat-GPTQ'
    model_basename = "model"

    model = AutoGPTQForCausalLM.from_quantized(
        model_name,
        revision="main",
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=True,
        inject_fused_attention=False,
        device=DEVICE,
        quantize_config=None,
    )

    # model_name = 'google/flan-t5-base'
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

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
                             streamer=streamer,
                             )

    llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0.1})

    memory = ConversationBufferWindowMemory(k=1, memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        memory=memory,
        chain_type="stuff",
        verbose=True,
    )

    return conversation_chain


def handle_userinput(user_question):
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
        st.chat_message(sender).write(message)


def validate_urls(urls):
    """Validate URLs and return a list of valid URLs."""
    valid_urls = []
    for url in urls:
        # Check if the URL is valid
        if validators.url(url):
            valid_urls.append(url)
        else:
            st.warning(f"Invalid URL: {url}")

    return valid_urls


def get_conversation_chain_huggingface(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.1, "max_length": 1024})

    memory = ConversationBufferWindowMemory(k=1, memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        verbose=True,
    )
    return conversation_chain


def get_conversation_chain_openai(vectorstore):
    llm = ChatOpenAI()

    memory = ConversationBufferWindowMemory(k=1, memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        verbose=True,
    )
    return conversation_chain


def get_conversation_chain_mistral(vectorstore):
    # https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/tree/main
    model_path = 'models/mistral-7b-instruct-v0.1.Q4_K_M.gguf'

    model = CTransformers(
        model=model_path,
        model_type='mistral',
        device=DEVICE,
        do_sample=True,
        config={
            'max_new_tokens': 4096,
            'temperature': 0.1,
            'top_p': 0.95,
            'repetition_penalty': 1.15,
        }
    )

    memory = ConversationBufferWindowMemory(k=1, memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        memory=memory,
        chain_type="stuff",
        verbose=True,
    )

    return conversation_chain


def create_conversation_chain_with_selected_model(vectorstore, model_options_spinner):
    if model_options_spinner == 'Llama-2-13B-chat-GPTQ (GPU required)':
        st.session_state.conversation = get_conversation_chain_gptq(vectorstore)
        print('the selected model is: Llama-2-13B-chat-GPTQ (GPU required)')
    elif model_options_spinner == 'Llama-2-13B-chat-GGML (CPU only)':
        st.session_state.conversation = get_conversation_chain_ggml(vectorstore)
        print('the selected model is: Llama-2-13B-chat-GGML (CPU only)')
    elif model_options_spinner == 'HuggingFace Hub (Online)':
        st.session_state.conversation = get_conversation_chain_huggingface(vectorstore)
        print('the selected model is: HuggingFace Hub (Online)')
    elif model_options_spinner == 'OpenAI API (Online)':
        st.session_state.conversation = get_conversation_chain_openai(vectorstore)
        print('the selected model is: OpenAI API (Online)')
    elif model_options_spinner == 'Mistral-7B (CPU only)':
        st.session_state.conversation = get_conversation_chain_mistral(vectorstore)
        print('the selected model is: Mistral-7B (CPU only)')


def main():
    load_dotenv()
    st.set_page_config(page_title="",
                       page_icon=":books:")

    # Define the state of the app
    pdf_docs = None
    urls_list = []
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "my_chat_history" not in st.session_state:
        st.session_state.my_chat_history = []
    if 'n_urls' not in st.session_state:
        st.session_state.n_urls = 1
    if 'urls' not in st.session_state:
        st.session_state.urls = []
    if 'text_area_input' not in st.session_state:
        st.session_state.text_area_input = None

    st.header("")
    user_question = st.text_input("Ask a question about your documents:")

    if user_question and st.session_state.conversation:
        handle_userinput(user_question)

    elif user_question and not st.session_state.conversation:
        st.error("Please upload your PDFs first and click on 'Process'")

    # Create a toggle button to show/hide the chat history object in session state
    if st.session_state.conversation:
        view_messages = st.expander("View the chain's chat history object", expanded=False)
        with view_messages:
            st.write(st.session_state.my_chat_history)

    with st.sidebar:
        # create a radio group for the different input options
        st.subheader("Input options")
        input_options_rg = ["Upload PDFs", "Enter URLs", "Enter text"]
        input_option = st.radio("", input_options_rg)

        # create divider to separate the input options from the rest
        st.markdown("---")

        if input_option == "Upload PDFs":
            st.subheader("Your documents")
            pdf_docs = st.file_uploader(
                "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

        elif input_option == "Enter URLs":
            st.subheader("Enter URLs")
            if st.button(label="add"):
                st.session_state.n_urls += 1
                st.experimental_rerun()

            for i in range(st.session_state.n_urls):
                urls_list.append(st.text_input("", placeholder=f"URL {i + 1}", label_visibility="collapsed"))
                print(urls_list)

            if st.button(label="remove"):
                if st.session_state.n_urls > 1:
                    st.session_state.n_urls -= 1
                    st.experimental_rerun()

        elif input_option == 'Enter text':
            st.subheader("Enter text")
            st.session_state.text_area_input = st.text_area("Enter your text here", height=200)

        # create divider to separate the input options from the rest
        st.markdown("---")

        # Create a radio group for the different models
        st.subheader("Select a Model")
        model_options = ['Mistral-7B (CPU only)', 'Llama-2-13B-chat-GPTQ (GPU required)', 
                         'Llama-2-13B-chat-GGML (CPU only)', 'HuggingFace Hub (Online)', 'OpenAI API (Online)']

        model_options_spinner = st.selectbox("", model_options)

        # create divider to separate the input options from the rest
        st.markdown("---")

        if st.button("Process", use_container_width=True):
            if input_option == "Upload PDFs":
                if pdf_docs:  # or link:
                    save_uploaded_files(pdf_docs)
                    with st.spinner("Processing"):

                        # start the timer
                        start_time = time.time()

                        # get pdf text
                        raw_text = []
                        for doc in pdf_docs:
                            if doc.type == 'application/pdf':
                                raw_text.extend(get_pdf_text([doc]))
                            elif doc.type == 'text/plain':
                                raw_text.extend(
                                    DirectoryLoader('uploaded_files/txt', glob='**/*.txt', show_progress=True).load())

                        # get the text chunks
                        text_chunks = get_text_chunks(raw_text)

                        # create vector store
                        vectorstore = get_vectorstore(text_chunks)

                        # create a conversation chain
                        create_conversation_chain_with_selected_model(vectorstore, model_options_spinner)

                        # Show a processing-done message and time taken for 5 seconds
                        show_temp_success_message(
                            f"Processing done!\n Time taken: {round(time.time() - start_time, 2)} Seconds", 5)

                else:
                    st.warning("Please upload your PDFs first and click on 'Process'")

            elif input_option == "Enter URLs":
                with st.spinner("Processing"):
                    # check if the user entered an url
                    print(len(urls_list))
                    if len(urls_list) == 1 and urls_list[0] == "":
                        st.warning("Please enter at least one URL")

                    else:

                        # start the timer
                        start_time = time.time()

                        # validate the urls
                        urls_list = validate_urls(urls_list)

                        # get the text from the urls
                        text = extract_text_from_urls(urls_list)

                        # get the text chunks
                        text_chunks = get_text_chunks(text)

                        # create vector store
                        vectorstore = get_vectorstore(text_chunks)

                        # create a conversation chain
                        create_conversation_chain_with_selected_model(vectorstore, model_options_spinner)

                        # Show a processing-done message and time taken for 5 seconds
                        show_temp_success_message(
                            f"Processing done!\n Time taken: {round(time.time() - start_time, 2)} Seconds", 5)

            elif input_option == 'Enter text':
                with st.spinner("Processing"):
                    # check if the user entered a text
                    print(f'the length of the text in the text area is: {len(st.session_state.text_area_input)}',
                          file=sys.stderr)
                    if st.session_state.text_area_input == "":
                        st.warning("Please enter some text first")

                    # start the timer
                    start_time = time.time()

                    text = st.session_state.text_area_input

                    # get the text chunks
                    text_chunks = get_text_chunks(text)

                    # create vector store
                    vectorstore = get_vectorstore(text_chunks)

                    # create a conversation chain
                    create_conversation_chain_with_selected_model(vectorstore, model_options_spinner)

                    # Show a processing-done message and time taken for 5 seconds
                    show_temp_success_message(
                        f"Processing done!\n Time taken: {round(time.time() - start_time, 2)} Seconds", 5)


if __name__ == '__main__':
    main()
