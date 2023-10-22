# Importing the necessary libraries
from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_text_chunks(text):
    """
    Splits the input text into chunks using a text splitter.

    Parameters
    ----------
    text: str or List[List[str]]
        A string containing a document to be split,
        or a list of lists where each sublist contains documents represented as list of sentences.

    Returns
    -------
    chunks: List[str] or List[List[str]]
        A list of chunks from the original text if input text is a string,
        or list of lists where each sublist contains chunks from original document if input is list of lists.

    """
    # Initialize the text splitter
    text_splitter = initialize_text_splitter()

    # Check if the text is a string
    if isinstance(text, str):
        chunks = split_text(text, text_splitter)
    else:
        chunks = split_documents(text, text_splitter)

    return chunks


def initialize_text_splitter():
    """
    Initializes the text splitter using specific parameters.

    Returns
    -------
    RecursiveCharacterTextSplitter
        An instance of RecursiveCharacterTextSplitter initialized with specific parameters.

    """
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )


def split_text(text, text_splitter):
    """
    Splits a document into chunks.

    Parameters
    ----------
    text: str
        A string containing the document to be split.
    text_splitter: RecursiveCharacterTextSplitter
        An instance of RecursiveCharacterTextSplitter.

    Returns
    -------
    chunks : List[str]
        A list of chunks from the original text.

    """
    return text_splitter.split_text(text)


def split_documents(text, text_splitter):
    """
    Splits a list of documents into chunks.

    Parameters
    ----------
    text: List[List[str]]
        A list of documents where each document is represented as a list of sentences.
    text_splitter: RecursiveCharacterTextSplitter
        An instance of RecursiveCharacterTextSplitter.

    Returns
    -------
    chunks: List[List[str]]
        A list of chunks from the original documents.

    """
    return text_splitter.split_documents(text)
