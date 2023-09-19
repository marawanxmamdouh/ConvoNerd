import validators
import os

import streamlit as st
import torch
from dotenv import load_dotenv
from langchain import HuggingFacePipeline, PromptTemplate, FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline

from deal_with_urls import extract_text_from_urls
from html_templates import css, bot_template, user_template

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(DEVICE)

# %%
custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be 
a standalone question. At the end of standalone question add this 'Answer the question in German language.' If you do 
not know the answer reply with 'I am sorry'. Chat History: {chat_history} Follow Up Input: {question} Standalone 
question:"""

# %%

prompt = PromptTemplate.from_template(template=custom_template)  # , input_variables=["context", "question"]


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
        st.success('File has been uploaded and saved to the session.')

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

    # check if the text is a list
    if isinstance(text_chunks, list):
        vectorstore = FAISS.from_texts(text_chunks, embeddings)
    else:
        vectorstore = FAISS.from_documents(text_chunks, embeddings)

    return vectorstore


def get_conversation_chain(vectorstore):
    # Load your local model from disk
    # model_name = 'google/flan-t5-base'
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

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
    response = st.session_state.conversation({'question': user_question,
                                              'chat_history': st.session_state.chat_history})
    print(response)
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
            print(message.content)


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

        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing"):
                    # get pdf text
                    raw_text = get_pdf_text(pdf_docs)
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

        if st.button("Process", use_container_width=True):
            if input_options_rg == "Upload PDFs":
                if pdf_docs:  # or link:
                    with st.spinner("Processing"):
                        # get pdf text
                        raw_text = get_pdf_text(pdf_docs)

                        # get the text chunks
                        text_chunks = get_text_chunks(raw_text)

                        # create vector store
                        vectorstore = get_vectorstore(text_chunks)

                        # create conversation chain
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                else:
                    st.warning("Please upload your PDFs first and click on 'Process'")

            elif input_option == "Enter URLs":
                with st.spinner("Processing"):
                    # check if the user entered a url
                    print(len(urls_list))
                    if len(urls_list) == 1 and urls_list[0] == "":
                        st.warning("Please enter at least one URL")

                    else:
                        # validate the urls
                        urls_list = validate_urls(urls_list)

                        # get the text from the urls
                        text = extract_text_from_urls(urls_list)

                        # get the text chunks
                        text_chunks = get_text_chunks(text)

                        # create vector store
                        vectorstore = get_vectorstore(text_chunks)

                        # create conversation chain
                        st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()
