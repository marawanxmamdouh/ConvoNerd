import os
import time

import streamlit as st
import torch
import validators
from auto_gptq import AutoGPTQForCausalLM
from dotenv import load_dotenv
from langchain import PromptTemplate, FAISS, HuggingFacePipeline, HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, TextStreamer, pipeline

from deal_with_urls import extract_text_from_urls
from html_templates import css

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(DEVICE)

# %%
DEFAULT_SYSTEM_PROMPT = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
""".strip()


def generate_prompt(prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return f"""
[INST] <<SYS>>
{system_prompt}
<</SYS>>

{prompt} [/INST]
""".strip()


SYSTEM_PROMPT = ("Use the following pieces of context to answer the question at the end. If you don't know the answer, "
                 "just say that you don't know, don't try to make up an answer.")

template = generate_prompt(
    """
Question: {question}
""",
    system_prompt=SYSTEM_PROMPT,
)

prompt = PromptTemplate(template=template, input_variables=["question"])


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
def get_pdf_text(pdf_docs):
    # delete old files
    if os.path.isdir('uploaded_files'):
        for f in os.listdir('uploaded_files'):
            os.remove(os.path.join('uploaded_files', f))
    for uploaded_file in pdf_docs:
        if not os.path.isdir('uploaded_files'):
            os.mkdir('uploaded_files')
        with open(os.path.join('uploaded_files', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getvalue())

        # show a success message for 2 seconds and then hide it.
        show_temp_success_message(f"File uploaded successfully: {uploaded_file.name}", 2)

    # load the pdfs
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
    model_name = "BAAI/bge-small-en"
    model_kwargs = {'device': DEVICE}
    encode_kwargs = {'normalize_embeddings': True}  # set True to compute cosine similarity
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    st.write(f"Loaded embeddings model: {text_chunks}")

    # check if the text is a list of strings or a list of lists of strings
    if isinstance(text_chunks, list) and isinstance(text_chunks[0], str):
        vectorstore = FAISS.from_texts(text_chunks, embeddings)
    else:
        vectorstore = FAISS.from_documents(text_chunks, embeddings)

    return vectorstore


def get_conversation_chain_ggml(vectorstore):
    # Load your local model from disk
    # Download the model from https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/tree/main,
    # model_path = 'models/llama-2-13b-chat.ggmlv3.q2_K.bin'
    # model_path = 'models/llama-2-13b-chat.ggmlv3.q4_1.bin'
    model_path = 'models/llama-2-7b-chat.ggmlv3.q8_0.bin'

    model = CTransformers(
        model=model_path,
        model_type='llama',
        device=DEVICE,
        config={
            'max_new_tokens': 1024,
            'temperature': 0.0,
            'top_p': 0.95,
            'repetition_penalty': 1.15,
        }
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        memory=memory,
        chain_type="stuff",
        condense_question_prompt=prompt,
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
                             max_new_tokens=1024,
                             temperature=0,
                             top_p=0.95,
                             repetition_penalty=1.15,
                             streamer=streamer,
                             )

    llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        memory=memory,
        chain_type="stuff",
        condense_question_prompt=prompt,
    )

    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    print(response)
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.chat_message("human").write(message.content)
        else:
            # Only keep the part that comes after 'Helpful Answer: '
            answer = message.content.split('Helpful Answer:', 1)[-1]
            st.chat_message("assistant").write(answer)
            print(answer)


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

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def get_conversation_chain_openai(vectorstore):
    llm = ChatOpenAI()

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
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


def main():
    load_dotenv()
    st.set_page_config(page_title="",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Define the state of the app
    pdf_docs = None
    urls_list = []
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if 'n_urls' not in st.session_state:
        st.session_state.n_urls = 1
    if 'urls' not in st.session_state:
        st.session_state.urls = []

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
            st.write(st.session_state.chat_history)

    with st.sidebar:
        # create a radio group for the different input options
        st.subheader("Input options")
        input_options_rg = ["Upload PDFs", "Enter URLs"]
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

        # create divider to separate the input options from the rest
        st.markdown("---")

        # Create a radio group for the different models
        st.subheader("Select a Model")
        model_options = ["Llama-2-13B-chat-GPTQ (GPU required)", "Llama-2-13B-chat-GGML (CPU only)",
                         'HuggingFace Hub (Online)', 'OpenAI API (Online)']

        model_options_spinner = st.selectbox("", model_options)

        # create divider to separate the input options from the rest
        st.markdown("---")

        if st.button("Process", use_container_width=True):
            if input_option == "Upload PDFs":
                if pdf_docs:  # or link:
                    with st.spinner("Processing"):

                        # start the timer
                        start_time = time.time()

                        # get pdf text
                        raw_text = get_pdf_text(pdf_docs)

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


if __name__ == '__main__':
    main()
