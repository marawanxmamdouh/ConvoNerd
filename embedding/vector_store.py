# Importing the necessary libraries
import torch
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores.faiss import FAISS
from loguru import logger as log

from utils.helpers import get_config

# Get the configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

cfg = get_config('embedding.yaml')


def get_vectorstore(text_chunks):
    """
    Retrieves a vector store created from the input text chunks using pre-trained embeddings.

    Parameters
    ----------
    text_chunks: List[str] or List[List[str]]
        A list of strings where each string is a piece of text to be processed,
        or a list of lists where each sub-list is a document represented by a list of sentences.

    Returns
    -------
    vectorstore : FAISS
        A FAISS vector store containing the sentence embeddings for each piece of text in text_chunks.

    Warnings
    --------
    - If the input text_chunks is empty.
    """
    vectorstore = None
    embeddings = initialize_embeddings()

    if text_chunks:
        vectorstore = create_vector_store(text_chunks, embeddings)
    else:
        log.warning(f'Your input is empty: {len(text_chunks)}. Please use a different input and try again.')

    return vectorstore


def initialize_embeddings():
    """
    Initializes the embeddings using a pre-trained HuggingFace BG E model.

    Returns
    -------
    HuggingFaceBgeEmbeddings
        The initialized HuggingFace BG E embeddings.

    """
    return HuggingFaceBgeEmbeddings(
        model_name=cfg.model_name,
        model_kwargs={'device': DEVICE},
        encode_kwargs=cfg.encode_kwargs
    )


def create_vector_store(text_chunks, embeddings):
    """
    Creates a FAISS vector store for the input text chunks using the provided embeddings.

    Parameters
    ----------
    text_chunks: List[str] or List[List[str]]
        A list of strings where each string is a piece of text to be processed,
        or a list of lists where each sub-list is a document represented by a list of sentences.
    embeddings: HuggingFaceBgeEmbeddings
        The initialized embeddings.

    Returns
    -------
    FAISS
        A FAISS vector store containing the sentence embeddings for each piece of text in text_chunks.
    """
    if isinstance(text_chunks, list) and isinstance(text_chunks[0], str):
        return FAISS.from_texts(text_chunks, embeddings)
    else:
        return FAISS.from_documents(text_chunks, embeddings)
