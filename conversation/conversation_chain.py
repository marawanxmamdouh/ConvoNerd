from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory


def create_conversation_chain(vectorstore, language_model):
    memory = initialize_memory()
    retriever = create_retriever(vectorstore)
    conversation_chain = build_conversation_chain(language_model, retriever, memory)
    return conversation_chain


def initialize_memory():
    # Initialize the ConversationBufferWindowMemory
    return ConversationBufferWindowMemory(k=1, memory_key='chat_history', return_messages=True)


def create_retriever(vectorstore):
    # Create the retriever with specific search parameters
    return vectorstore.as_retriever(search_kwargs={"k": 2})


def build_conversation_chain(language_model, retriever, memory):
    # Build the ConversationalRetrievalChain
    return ConversationalRetrievalChain.from_llm(
        llm=language_model,
        retriever=retriever,
        memory=memory,
        chain_type="stuff",
        verbose=True
    )
