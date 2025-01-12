mport streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging
from requests.exceptions import Timeout
from datetime import datetime
import glob

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
DOCUMENTS_PATH = "documents/"  # Folder untuk menyimpan PDF

# Pustaka Data 
PUSTAKA_DATA = {
    "files": [
        {
            "title": "9 Parameter 2025",
            "description": "9 Parameter Penilaian LCV AKHLAK 2025.",
            "url": "https://drive.google.com/file/d/11DOL9kG0ttp0ilJ_5ykLrSjKVEI9IJlV/view?usp=sharing"
        },
        {
            "title": "10 Fokus Keberlanjutan Pertamina",
            "description": "Guideline 10 Fokus Keberlanjutan Pertamina beserta contohnya.",
            "url": "https://drive.google.com/file/d/1FTIttFp17nGh5Pfc_w-wS-Xf7D_aLUrg/view?usp=sharing"
        },
        {
            "title": "Contoh PCB dengan Allignment terhadap 10 Fokus Keberlanjutan Pertamina",
            "description": "Berbagai contoh PCB yang memiliki program budaya menyasar pada 10 Fokus Keberlnajutan Pertamina.",
            "url": "https://drive.google.com/file/d/17Bx_ha1o01UsrovI6TVadmupP9LxN-J5/view?usp=sharing"
        },
        {
            "title": "Contoh Klasifikasi Program",
            "description": "Contoh-contoh klasifikasi Program Strategis, Taktikal, Operasional.",
            "url": "https://docs.google.com/spreadsheets/d/1irDS2zSD8yavfEf5uLDSpuCY65T_UIe0/edit?usp=sharing"
        },
        {
            "title": "Form Kuantifikasi Impact to Business",
            "description": "Form kuantifikasi impact to business dan contoh pengisian.",
            "url": "https://docs.google.com/spreadsheets/d/1W2jlrIhiJac_1oLd86dSSXLMihgV1KO4/edit?usp=sharing"
        },
        {
            "title": "Sosialisasi LCV 2025.",
            "description": "Materi sosialisasi LCV 2025",
            "url": "https://drive.google.com/file/d/1iXtwOtd0BCF4tQnz2R2Obek3I2E0h5cM/view?usp=sharing"
        },
        {
            "title": "Dashboard PowerBI",
            "description": "Nilai Kualitatif Evidence Bulanan.",
            "url": "https://ptm.id/skorlivingcorevaluesAKHLAK"
        },
        {
            "title": "Konfirmasi Evidence",
            "description": "Form Konfirmasi Evidence LCV 2025.",
            "url": "https://ptm.id/FormKonfirmasiEvidenceLCV2025"
        }
    ]
}

# Fungsi untuk menampilkan Pustaka Dokumen
def display_pustaka():
    st.sidebar.title("Referensi Penunjang")
    st.sidebar.write("Daftar Referensi yang Tersedia:")

    for file in PUSTAKA_DATA['files']:
        st.sidebar.write(f"**{file['title']}**")
        st.sidebar.write(f"Deskripsi: {file['description']}")
        st.sidebar.write(f"URL: {file['url']}")
        st.sidebar.write("---")

# Konfigurasi Streamlit
st.set_page_config(page_title="🤖LCV ASSISTANT", layout="centered")

def initialize_session_state():
    """Inisialisasi session state"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False

def setup_groq():
    """Setup model Groq"""
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
        llm = ChatGroq(
            temperature=0.27,
            groq_api_key=groq_api_key,
            model_name="mixtral-8x7b-32768"
        )
        return llm
    except Exception as e:
        logging.error(f"Error setting up Groq: {str(e)}")
        st.error("Error dalam inisialisasi model AI. Mohon periksa API key.")
        return None

def process_documents():
    """Proses dokumen PDF dari folder documents"""
    try:
        if not os.path.exists(DOCUMENTS_PATH):
            os.makedirs(DOCUMENTS_PATH)
            return False

        pdf_files = glob.glob(os.path.join(DOCUMENTS_PATH, "*.pdf"))
        if not pdf_files:
            return False

        documents = []
        for pdf_path in pdf_files:
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2700,
            chunk_overlap=180
        )
        splits = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vector_store = FAISS.from_documents(splits, embeddings)
        st.session_state.vector_store = vector_store
        st.session_state.documents_processed = True
        
        logging.info("Dokumen berhasil diproses")
        return True

    except Exception as e:
        logging.error(f"Error dalam pemrosesan dokumen: {str(e)}")
        st.error(f"Error dalam pemrosesan dokumen: {str(e)}")
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
        return True

    except Exception as e:
        logging.error(f"Error dalam setup conversation: {str(e)}")
        st.error("Error dalam setup sistem percakapan")
        return False

def main():
    initialize_session_state()
    
    st.title("🤖 LCV Assistant-DEMO")
    st.markdown(
        """
        <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;'>
        <p style='font-size: 18px; margin: 0;'>
        <strong>Selamat datang !</strong> - Chatbot ini adalah Asisten yang dilatih  untuk membantu AoC dalam implementasi LCV AKHLAK 2025.
        Pastikanlah Anda memiliki koneksi internet yang baik dan stabil. Terimakasih atas kesabarannya menunggu chatbot siap untuk digunakan 🙏
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    # Header
    st.markdown("""
    ---
    **Disclaimer:**
    - Sistem ini menggunakan AI dan dapat menghasilkan jawaban yang tidak selalu akurat
    - ketik : LANJUTKAN JAWABANMU untuk kemungkinan mendapatkan jawaban yang lebih baik dan utuh.
    - Mohon verifikasi informasi penting dengan sumber terpercaya
    """)

    # Proses dokumen otomatis
    if not st.session_state.documents_processed:
        with st.spinner("Memproses dokumen..."):
            if process_documents():
                llm = setup_groq()
                if llm and setup_conversation(llm, st.session_state.vector_store):
                    st.success("Sistem siap digunakan!")
            else:
                st.info("Silakan tambahkan file PDF ke folder 'documents'")
                return

    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("✍️ Tuliskan pertanyaan Anda"):
        if len(prompt) > MAX_INPUT_LENGTH:
            st.warning(f"Pertanyaan terlalu panjang. Maksimal {MAX_INPUT_LENGTH} karakter.")
            return

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            try:
                with st.spinner("🤔 Sedang berpikir..."):
                    response = st.session_state.conversation.invoke(
                        {"question": prompt + " (Tolong jawab dalam Bahasa Indonesia)"},
                        timeout=TIMEOUT_SECONDS
                    )
                    ai_response = response["answer"]
                    
                message_placeholder.write(ai_response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": ai_response}
                )
                
                if len(st.session_state.messages) > MAX_MESSAGES:
                    st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]
                    
            except Timeout:
                message_placeholder.error("Waktu permintaan habis. Silakan coba lagi.")
            except Exception as e:
                message_placeholder.error(f"Terjadi kesalahan: {str(e)}")

if __name__ == "__main__":
    try:
        display_pustaka()
        main()
    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")
