import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging
from requests.exceptions import Timeout
import tempfile
from datetime import datetime

# Konfigurasi logging
logging.basicConfig(
    filename='chatbot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Konstanta
MAX_MESSAGES = 50
MAX_INPUT_LENGTH = 500
TIMEOUT_SECONDS = 30

# Konfigurasi Streamlit
st.set_page_config(page_title="AI Assistant", layout="wide")

def initialize_session_state():
    """Inisialisasi session state"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

def setup_groq():
    """Setup model Groq"""
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
        llm = ChatGroq(
            temperature=0.3,
            groq_api_key=groq_api_key,
            model_name="mixtral-8x7b-32768"
        )
        return llm
    except Exception as e:
        logging.error(f"Error setting up Groq: {str(e)}")
        st.error("Error initializing AI model. Please check your API key.")
        return None

def process_documents(uploaded_files):
    """Proses dokumen PDF yang diupload"""
    try:
        logging.info("Starting document processing")
        documents = []
        
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            loader = PyPDFLoader(tmp_file_path)
            documents.extend(loader.load())
            os.unlink(tmp_file_path)

        # Text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(documents)

        # Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Create vector store
        vector_store = FAISS.from_documents(splits, embeddings)
        st.session_state.vector_store = vector_store
        
        logging.info("Document processing completed successfully")
        return True

    except Exception as e:
        logging.error(f"Error in document processing: {str(e)}")
        st.error(f"Error processing documents: {str(e)}")
        return False

def setup_conversation(llm, vector_store):
    """Setup conversation chain"""
    try:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        conversation = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory,
            verbose=True
        )
        
        st.session_state.conversation = conversation
        logging.info("Conversation chain setup successfully")
        return True

    except Exception as e:
        logging.error(f"Error setting up conversation: {str(e)}")
        st.error("Error setting up conversation system")
        return False

def export_chat_history():
    """Export riwayat chat ke file"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_history_{timestamp}.txt"
        
        with open(filename, "w", encoding="utf-8") as f:
            for message in st.session_state.messages:
                f.write(f"{message['role']}: {message['content']}\n\n")
        
        return filename
    except Exception as e:
        logging.error(f"Error exporting chat history: {str(e)}")
        return None

def main():
    initialize_session_state()
    
    st.title("🤖 LCV Assistant")
    st.markdown(
        """
        <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;'>
        <p style='font-size: 18px; margin: 0;'>
        <strong>Selamat datang !</strong> - Chatbot ini adalah Asisten yang dilatih oleh CCM untuk membantu AoC dalam implementasi LCV AKHLAK 2025.
        Pastikanlah Anda memiliki koneksi internet yang baik dan stabil. Terimakasih atas kesabarannya menunggu chatbot siap untuk digunakan.
        </p>
        </div>
        """,
         unsafe_allow_html=True
     )
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        uploaded_files = st.file_uploader(
            "Upload PDF Documents",
            type=["pdf"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            with st.spinner("Processing documents..."):
                if process_documents(uploaded_files):
                    st.success("Documents processed successfully!")
                    
                    # Setup LLM and conversation
                    llm = setup_groq()
                    if llm and setup_conversation(llm, st.session_state.vector_store):
                        st.success("AI Assistant is ready!")
        
        if st.button("Reset Conversation"):
            st.session_state.messages = []
            st.session_state.conversation = None
            st.session_state.vector_store = None
            logging.info("Conversation reset")
            st.experimental_rerun()
            
        if st.button("Export Chat History"):
            if st.session_state.messages:
                filename = export_chat_history()
                if filename:
                    st.success(f"Chat history exported to {filename}")
                else:
                    st.error("Failed to export chat history")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Tuliskan pertanyaan Anda disini secara spesifik"):
        if len(prompt) > MAX_INPUT_LENGTH:
            st.warning(f"Question too long. Maximum {MAX_INPUT_LENGTH} characters allowed.")
            return
            
        if not st.session_state.conversation:
            st.warning("Please upload documents first!")
            return
            
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
            
        # Generate AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            try:
                with st.spinner("🤔 Sedang berpikir..."):
                    response = st.session_state.conversation.invoke(
                        {"question": prompt},
                        timeout=TIMEOUT_SECONDS
                    )
                    ai_response = response["answer"]
                    
                message_placeholder.write(ai_response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": ai_response}
                )
                
                # Limit message history
                if len(st.session_state.messages) > MAX_MESSAGES:
                    st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]
                    
            except Timeout:
                message_placeholder.error("Request timed out. Please try again.")
                logging.error("Request timeout occurred")
            except Exception as e:
                message_placeholder.error(f"An error occurred: {str(e)}")
                logging.error(f"Error generating response: {str(e)}")
    # Disclaimer
    st.markdown(
        """
        <p style='font-size: 12px; font-style: italic; color: gray;'>
        ⚠️ <strong>Disclaimer:</strong> AI-LLM model dapat saja membuat kesalahan. CEK KEMBALI INFO PENTING.
        </p>
        """,
        unsafe_allow_html=True
    )
if __name__ == "__main__":
    main()
