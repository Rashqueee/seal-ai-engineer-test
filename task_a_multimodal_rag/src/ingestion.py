import os
import fitz
import pdfplumber
import base64
from fastapi import UploadFile
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load API Key dari .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Inisialisasi Gemini 2.5 Flash untuk Vision dan Teks
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=GOOGLE_API_KEY)
# Model Embedding untuk mengubah teks menjadi vektor
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=GOOGLE_API_KEY)
# Inisialisasi ChromaDB lokal
vector_store = Chroma(collection_name="mandiri_report", embedding_function=embeddings, persist_directory="./chroma_db")

def process_image_with_vision(image_bytes: bytes) -> str:
    """Mengirim gambar ke Gemini untuk dideskripsikan."""
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Jelaskan secara detail informasi, data, dan persentase yang ada pada grafik/gambar ini. Jika ini adalah struktur visual, jelaskan alurnya."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
        ]
    )
    try:
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        print(f"Error processing image: {e}")
        return ""

def extract_tables_from_page(pdf_path: str, page_num: int) -> str:
    """Mengekstrak tabel menjadi teks berformat menggunakan pdfplumber."""
    table_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        tables = page.extract_tables()
        for table in tables:
            for row in table:
                # Menghapus nilai None dan menggabungkan baris dengan pemisah |
                clean_row = [str(cell).replace('\n', ' ') if cell else "" for cell in row]
                table_text += " | ".join(clean_row) + "\n"
            table_text += "\n"
    return table_text

async def process_and_ingest_pdf(file: UploadFile) -> dict:
    """Fungsi utama untuk memproses PDF dan memasukkannya ke Vector DB."""
    # Simpan file PDF sementara
    temp_pdf_path = f"temp_{file.filename}"
    with open(temp_pdf_path, "wb") as f:
        f.write(await file.read())

    documents = []
    
    # Buka PDF dengan PyMuPDF untuk teks dan gambar
    doc = fitz.open(temp_pdf_path)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Ekstraksi Teks Standar
        raw_text = page.get_text("text")
        
        # Ekstraksi Tabel (menggunakan index pdfplumber)
        table_text = extract_tables_from_page(temp_pdf_path, page_num)
        
        # Ekstraksi Gambar/Grafik
        image_summaries = ""
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Deskripsikan gambar menggunakan Gemini Vision
            summary = process_image_with_vision(image_bytes)
            if summary:
                image_summaries += f"\n[Deskripsi Gambar {img_index + 1} di Halaman {page_num + 1}]:\n{summary}\n"

        # Gabungkan semua konteks dalam satu halaman
        combined_page_content = f"--- Teks Halaman ---\n{raw_text}\n\n--- Data Tabel ---\n{table_text}\n\n--- Data Visual/Grafik ---\n{image_summaries}"
        
        # Buat objek Document dengan metadata (sangat penting untuk evaluasi soal)
        doc_obj = Document(
            page_content=combined_page_content,
            metadata={"page_number": page_num + 1, "source": file.filename}
        )
        documents.append(doc_obj)

    doc.close()
    os.remove(temp_pdf_path) # Hapus file sementara

    # Chunking (Pemotongan Teks)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, # Ukuran teks per potongan
        chunk_overlap=200, # Overlap agar konteks antar potongan tidak terputus
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    # Masukkan ke Vector Database (ChromaDB)
    vector_store.add_documents(chunks)

    return {
        "message": "Dokumen berhasil diproses dan disimpan ke Vector Database.",
        "total_pages_processed": len(documents),
        "total_chunks_created": len(chunks)
    }