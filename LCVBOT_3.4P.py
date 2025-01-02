import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
import logging
import toml

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)

# Load config dari secrets.toml
def load_api_key():
    try:
        with open('.streamlit/secrets.toml', 'r') as f:
            config = toml.load(f)
            return config.get('GROQ_API_KEY')
    except FileNotFoundError:
        st.error("File secrets.toml tidak ditemukan di folder .streamlit/")
        return None
    except Exception as e:
        st.error(f"Error membaca API key: {str(e)}")
        return None

# Definisikan path folder dokumen
DOCUMENTS_PATH = "./documents"

def initialize_llm(api_key):
    """Inisialisasi model LLM"""
    if not api_key:
        raise ValueError("API key tidak ditemukan")
    
    return ChatGroq(
        api_key=api_key,
        model_name="mixtral-8x7b-32768",
        temperature=0.27,
        max_tokens=3699,
        top_p=0.9,
        presence_penalty=0.1,
        frequency_penalty=0.1
    )

def process_documents():
    """Memproses dokumen dari folder documents"""
    if not os.path.exists(DOCUMENTS_PATH):
        raise FileNotFoundError(f"Folder {DOCUMENTS_PATH} tidak ditemukan")
    
    # Cek apakah ada file PDF dalam folder
    pdf_files = [f for f in os.listdir(DOCUMENTS_PATH) if f.endswith('.pdf')]
    if not pdf_files:
        raise FileNotFoundError("Tidak ada file PDF yang ditemukan dalam folder documents")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    
    loader = DirectoryLoader(
        DOCUMENTS_PATH,
        glob="*.pdf",  # Ubah pattern untuk PDF
        loader_cls=PyPDFLoader  # Gunakan PyPDFLoader
    )
    
    documents = loader.load()
    chunks = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    return FAISS.from_documents(chunks, embeddings)

def get_conversation_chain(vector_store, api_key):
    llm = initialize_llm(api_key)
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )

def main():
    st.title(":blue[🤖] LCV ASSISTANT")
    st.markdown(
        """
        <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;'>
        <p style='font-size: 18px; margin: 0;'>
        <strong>Selamat datang !</strong> - Chatbot ini adalah Asisten yang dilatih untuk membantu AoC dalam implementasi LCV AKHLAK 2025.
        Pastikanlah Anda memiliki koneksi internet yang baik dan stabil. Terimakasih atas kesabarannya menunggu chatbot siap digunakan.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Load API key
    api_key = load_api_key()
    if not api_key:
        st.error("Tidak dapat melanjutkan tanpa API key")
        return

    # Initialize vector store and conversation chain
    if 'vector_store' not in st.session_state:
        try:
            with st.spinner('Mempersiapkan sistem...'):
                st.session_state.vector_store = process_documents()
                st.session_state.conversation = get_conversation_chain(
                    st.session_state.vector_store,
                    api_key
                )
        except Exception as e:
            st.error(f"Error saat inisialisasi: {str(e)}")
            return

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Tuliskan pertanyaan Anda disini secara spesifik"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                modified_prompt = "Berikan jawaban selalu dalam bahasa Indonesia yang baik dan terstruktur. Jika tidak tahu, katakan Mohon Maaf saya tidak mendapatkan hal ini dalam pelatihan saya: " + prompt
                with st.spinner("🤔 Sedang berpikir..."):
                    response = st.session_state.conversation({"question": modified_prompt})
                st.markdown(response['answer'])
                st.session_state.messages.append({"role": "assistant", "content": response['answer']})
            except Exception as e:
                error_message = f"Error saat memproses pertanyaan: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

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
