# Importing the necessary libraries
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory

from utils.helpers import get_config

# Get the configuration
cfg = get_config('conversation_chain.yaml')


def create_conversation_chain(vectorstore, language_model):
    """
    Create a conversation chain which consists of initialized memory, retriever
    defined by the vectorstore, and a language model.

    Parameters
    ----------
    vectorstore: Vectorstore object
        The vectorstore instance used to create the retriever.
    language_model: LanguageModel object
        The language model instance used to set up the conversation chain.

    Returns
    -------
    conversation_chain: ConversationRetrievalChain object
        The conversation chain built using the given language model, retriever and memory.
    """
    memory = initialize_memory()
    retriever = create_retriever(vectorstore)
    conversation_chain = build_conversation_chain(language_model, retriever, memory)
    return conversation_chain


def initialize_memory():
    """
    Initialize the ConversationBufferWindowMemory with certain parameters.

    Returns
    -------
    memory: ConversationBufferWindowMemory instance
        Initialized memory for the conversation chain.
    """
    return ConversationBufferWindowMemory(k=cfg.memory.k, memory_key='chat_history', return_messages=True)


def create_retriever(vectorstore):
    """
    Create a retriever with certain search parameters.

    Parameters
    ----------
    vectorstore: Vectorstore object
        The vectorstore instance used to create the retriever.

    Returns
    -------
    retriever: Retriever object
        A retriever created using the given vectorstore.
    """
    return vectorstore.as_retriever(search_kwargs={"k": cfg.retriever.k})


def build_conversation_chain(language_model, retriever, memory):
    """
    Build the ConversationalRetrievalChain using a given language model, retriever,
    and memory.

    Parameters
    ----------
    language_model: LanguageModel object
        The language model instance used to set up the conversation chain.
    retriever: Retriever object
        The retriever used in the conversation chain.
    memory: ConversationBufferWindowMemory instance
        The memory used in the conversation chain.

    Returns
    -------
    conversation_chain: ConversationRetrievalChain object
        The conversation chain built using the given language model, retriever and memory.
    """
    return ConversationalRetrievalChain.from_llm(
        llm=language_model,
        retriever=retriever,
        memory=memory,
        chain_type=cfg.conversation_chain.chain_type,
        verbose=cfg.conversation_chain.verbose,
    )
