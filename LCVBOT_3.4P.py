import os
import sys
import subprocess
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
import toml

# Konstanta
DOCUMENTS_PATH = "documents"
MODEL_NAME = "mixtral-8x7b-32768"

# Instalasi package yang diperlukan
def install_missing_packages():
    required_packages = [
        'streamlit',
        'langchain_groq',
        'langchain_community',
        'langchain',
        'huggingface_hub',
        'faiss-cpu',
        'PyPDF2',
        'sentence_transformers',
        'toml'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Load API key
def load_api_key():
    try:
        # Coba ambil dari environment variable dulu
        api_key = os.getenv('GROQ_API_KEY')
        if api_key:
            return api_key
            
        # Jika tidak ada, coba dari secrets.toml
        with open('.streamlit/secrets.toml', 'r') as f:
            config = toml.load(f)
            return config.get('GROQ_API_KEY')
    except FileNotFoundError:
        st.error("File secrets.toml tidak ditemukan. Pastikan file ada di folder .streamlit/")
        return None
    except Exception as e:
        st.error(f"Error membaca API key: {str(e)}")
        return None

# Proses dokumen PDF
def process_documents():
    try:
        if not os.path.exists(DOCUMENTS_PATH):
            os.makedirs(DOCUMENTS_PATH)  # Buat folder jika tidak ada
            st.warning(f"Folder {DOCUMENTS_PATH} telah dibuat. Silakan tambahkan file PDF Anda.")
            return None
        
        pdf_files = [f for f in os.listdir(DOCUMENTS_PATH) if f.endswith('.pdf')]
        if not pdf_files:
            st.warning("Tidak ada file PDF yang ditemukan dalam folder documents")
            return None

        documents = []
        for pdf_file in pdf_files:
            loader = PyPDFLoader(os.path.join(DOCUMENTS_PATH, pdf_file))
            documents.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        chunks = text_splitter.split_documents(documents)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store
        
    except Exception as e:
        st.error(f"Error dalam memproses dokumen: {str(e)}")
        return None

# Inisialisasi conversation chain
def get_conversation_chain(vector_store, api_key):
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=MODEL_NAME,
        temperature=0.3,
        max_tokens=1024,
        top_p=1
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )
    
    return conversation_chain

def check_required_packages():
    required_packages = [
        'selenium',
        'beautifulsoup4',
        'pandas',
        'requests',
        'python-dotenv'
        # tambahkan package lain yang dibutuhkan
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Please add these packages to requirements.txt")
        return False
    return True

# Fungsi utama
def main():
    try:
        st.title(":blue[🤖] LCV ASSISTANT")
        
        # Initialize session state untuk menyimpan riwayat chat
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        # Load API key
        api_key = load_api_key()
        if not api_key:
            st.error("API key tidak ditemukan. Pastikan file secrets.toml berisi GROQ_API_KEY")
            return

        # Initialize vector store
        if 'vector_store' not in st.session_state:
            with st.spinner('Mempersiapkan sistem...'):
                vector_store = process_documents()
                if vector_store is None:
                    return
                st.session_state.vector_store = vector_store
                st.session_state.conversation = get_conversation_chain(
                    st.session_state.vector_store,
                    api_key
                )

        # Tampilkan riwayat chat
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Input chat
        if prompt := st.chat_input("Tuliskan pertanyaan Anda disini secara spesifik"):
            try:
                if not prompt.strip():  # Cek input kosong
                    st.warning("Mohon masukkan pertanyaan yang valid")
                    return
                    
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
                        error_message = "Maaf, terjadi kesalahan saat memproses pertanyaan Anda. Silakan coba lagi."
                        st.error(f"{error_message}\nDetail error: {str(e)}")
                        st.session_state.messages.append({"role": "assistant", "content": error_message})
            except Exception as e:
                st.error(f"Terjadi kesalahan: {str(e)}")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")

if __name__ == "__main__":
    try:
        if not check_required_packages():  # Ganti install_missing_packages() dengan check_required_packages()
            st.error("Beberapa package yang dibutuhkan belum terinstall!")
            st.stop()
        main()
    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")
