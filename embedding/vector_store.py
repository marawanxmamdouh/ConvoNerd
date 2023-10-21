import torch
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores.faiss import FAISS
from loguru import logger as log

# %%: Device to use
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# %%: Vector store
def get_vectorstore(text_chunks):
    # Initialize the vector store components
    vectorstore = None
    embeddings = initialize_embeddings()

    if text_chunks:
        vectorstore = create_vector_store(text_chunks, embeddings)
    else:
        log.warning(f'Your input is empty: {len(text_chunks)}. Please use a different input and try again.')

    return vectorstore


def initialize_embeddings():
    # Initialize the HuggingFaceBgeEmbeddings with specific model and parameters
    model_name = "BAAI/bge-small-en"
    model_kwargs = {'device': DEVICE}
    encode_kwargs = {'normalize_embeddings': True}
    return HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )


def create_vector_store(text_chunks, embeddings):
    # Check if the text is a list of strings or a list of lists of strings
    if isinstance(text_chunks, list) and isinstance(text_chunks[0], str):
        return FAISS.from_texts(text_chunks, embeddings)
    else:
        return FAISS.from_documents(text_chunks, embeddings)
