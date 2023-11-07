# Importing the necessary libraries
import torch
from box import Box
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.schema import Document
from langchain.vectorstores.faiss import FAISS
from loguru import logger as log

from utils.helpers import get_config

# Get the configuration
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Get the configuration
cfg: Box = get_config('embedding.yaml')


def get_vectorstore(text_chunks: list[str] | list[Document]) -> FAISS | None:
    """
    Retrieves a vector store created from the input text chunks using pre-trained embeddings.

    Parameters
    ----------
    text_chunks: list[str] | list[Document]
         A list of chunks from the original text.

    Returns
    -------
    vectorstore : FAISS
        A FAISS vector store containing the sentence embeddings for each piece of text in text_chunks.

    Warnings
    --------
    - If the input text_chunks is empty.
    """
    embeddings: HuggingFaceBgeEmbeddings = initialize_embeddings()

    if not text_chunks:
        log.warning(f'Your input is empty: {len(text_chunks)}. Please use a different input and try again.')
        return None

    vectorstore: FAISS = create_vector_store(text_chunks, embeddings)

    return vectorstore


def initialize_embeddings() -> HuggingFaceBgeEmbeddings:  # pragma: no cover
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


def create_vector_store(text_chunks: list[str] | list[Document], embeddings: HuggingFaceBgeEmbeddings) -> FAISS:
    """
    Creates a FAISS vector store for the input text chunks using the provided embeddings.

    Parameters
    ----------
    text_chunks: list[str] or list[Document]
         A list of chunks from the original text.
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
