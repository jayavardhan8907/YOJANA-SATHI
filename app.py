import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
import os
import tempfile

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

    # Track whether chat history has been displayed
    if 'chat_history_displayed' not in st.session_state:
        st.session_state['chat_history_displayed'] = False

def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your PDF", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

            # Set chat history displayed to False to show it next time
            st.session_state['chat_history_displayed'] = False

    if st.session_state['generated']:
        with reply_container:
            if not st.session_state['chat_history_displayed']:
                # Display chat history if it hasn't been displayed yet
                for i in range(len(st.session_state['generated'])):
                    messages = [{"role": "user", "content": st.session_state["past"][i]},
                                {"role": "chatbot", "content": st.session_state["generated"][i]}]
                    for message in messages:
                        avatar = "logo2.png" if message["role"] == "chatbot" else None
                        with st.chat_message(message["role"], avatar=avatar):
                            st.markdown(message["content"])

                # Set chat history displayed to True
                st.session_state['chat_history_displayed'] = True

def create_conversational_chain(vector_store):
    # Create llm
    llm = LlamaCpp(
        streaming=True,
        model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        temperature=0.75,
        top_p=1, 
        verbose=True,
        n_ctx=4096
    )
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                 retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                 memory=memory)
    return chain

def main():
    # Initialize session state
    initialize_session_state()
    
    # Set the title and logo
    st.title("YOJANA SATHI")
    st.image("flag.jpeg", width=50)  # Adjust width as needed

    # Default folder path containing PDFs
    default_folder_path = r'C:\Users\vardh\OneDrive\Documents\GitHub\MultiPDFchatMistral-7B\database'

    # Initialize Streamlit
    #st.sidebar.title("Document Processing")

    # Process PDFs from the default folder path
    text = []
    for filename in os.listdir(default_folder_path):
        file_path = os.path.join(default_folder_path, filename)
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            text.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(text)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                       model_kwargs={'device': 'cpu'})

    # Create vector store
    vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

    # Create the chain object
    chain = create_conversational_chain(vector_store)

    display_chat_history(chain)

if __name__ == "__main__":
    main()
