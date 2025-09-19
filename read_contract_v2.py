import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os

# Fungsi untuk mengekstrak teks dari PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Fungsi utama aplikasi Streamlit
def main():
    st.set_page_config(page_title="Analisis Dokumen Kontrak", layout="wide")
    st.header("Analisis Dokumen Kontrak dengan OpenAI 💬")

    # Sidebar untuk input API Key
    with st.sidebar:
        st.title("Pengaturan")
        st.write("Masukkan Kunci API OpenAI Anda untuk memulai.")
        api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.success("API Key berhasil disimpan!", icon="✅")

    # Area utama untuk upload file dan tanya jawab
    st.subheader("1. Unggah Dokumen Kontrak Anda")
    pdf_docs = st.file_uploader("Pilih file PDF kontrak", type="pdf", accept_multiple_files=True)

    if not api_key:
        st.warning("Mohon masukkan OpenAI API Key Anda di sidebar.")
        return

    if pdf_docs:
        # Ekstrak teks dari PDF yang diunggah
        raw_text = get_pdf_text(pdf_docs)

        if raw_text:
            # Bagi teks menjadi chunk yang lebih kecil
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            texts = text_splitter.split_text(raw_text)

            # Buat embeddings dan vector store
            try:
                embeddings = OpenAIEmbeddings()
                knowledge_base = FAISS.from_texts(texts, embeddings)
                st.success("Dokumen berhasil diproses. Silakan ajukan pertanyaan.")

                st.subheader("2. Ajukan Pertanyaan Tentang Kontrak")
                user_question = st.text_input("Apa yang ingin Anda ketahui dari dokumen ini?")

                if user_question:
                    # Lakukan pencarian similaritas dan dapatkan jawaban
                    with st.spinner("Menganalisis dokumen..."):
                        docs = knowledge_base.similarity_search(user_question)
                        
                        llm = OpenAI()
                        chain = load_qa_chain(llm, chain_type="stuff")
                        response = chain.run(input_documents=docs, question=user_question)

                        st.subheader("Jawaban:")
                        st.write(response)
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses dokumen: {e}")
                st.info("Pastikan API Key Anda valid dan memiliki cukup kuota.")

        else:
            st.warning("Gagal mengekstrak teks dari PDF. Mohon coba file lain.")

if __name__ == '__main__':
    main()
