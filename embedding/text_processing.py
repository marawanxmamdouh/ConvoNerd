from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_text_chunks(text):
    # Initialize the text splitter
    text_splitter = initialize_text_splitter()

    # Check if the text is a string
    if isinstance(text, str):
        chunks = split_text(text, text_splitter)
    else:
        chunks = split_documents(text, text_splitter)

    return chunks


def initialize_text_splitter():
    # Initialize the RecursiveCharacterTextSplitter with parameters
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )


def split_text(text, text_splitter):
    # Split a text document into chunks
    return text_splitter.split_text(text)


def split_documents(text, text_splitter):
    # Split a list of documents into chunks
    return text_splitter.split_documents(text)
