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
import PyPDF2

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def load_pdf_texts_from_folder(folder_path):
    pdf_texts = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pdf"):
                file_path = os.path.join(root, file)
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfFileReader(f)
                    for page_num in range(reader.numPages):
                        page = reader.getPage(page_num)
                        pdf_texts.append(page.extractText())
    return pdf_texts

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
    st.image("flag.jpeg", width=100)  # Adjust width as needed

    # Initialize Streamlit
    st.sidebar.title("Document Processing")
    folder_path = 'C:/Users/vardh/OneDrive/Documents/GitHub/MultiPDFchatMistral-7B/database'

    # Check if the specified folder path exists
    if not os.path.exists(folder_path):
        st.error("The specified folder path does not exist.")
        return

    # Load PDF texts from the specified folder
    pdf_texts = load_pdf_texts_from_folder(folder_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(pdf_texts)

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
