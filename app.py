import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
import requests
import pdfplumber
from io import BytesIO
import faiss  # Explicitly import faiss

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about "]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! "]

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

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

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

    # Direct link to the PDF file
    pdf_url = "https://github.com/jayavardhan8907/YOJANA-SATHI/raw/main/database/AP.pdf"

    # Download the PDF file
    response = requests.get(pdf_url)
    pdf_bytes = BytesIO(response.content)

    # Fetch text from the PDF using pdfplumber
    with pdfplumber.open(pdf_bytes) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()

    # Split text into chunks (adjust chunk size as needed)
    chunk_size = 10000
    text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                     model_kwargs={'device': 'cpu'})

    try:
        # Create vector store directly using embeddings (no need for page_content)
        vector_store = faiss.IndexFlatL2(len(embeddings[0]))  # Adjust dimensionality
        vector_store.add(embeddings)

        # Create the chain object
        chain = create_conversational_chain(vector_store)

        display_chat_history(chain)

    except Exception as e:
        print(f"Error encountered: {e}")
        # Handle the error appropriately (e.g., display an error message to the user)

if __name__ == "__main__":
    main()
