import os
import fitz
import pdfplumber
import base64
from fastapi import UploadFile
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load API Key dari .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Inisialisasi Gemini untuk Vision dan Teks
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", google_api_key=GOOGLE_API_KEY)
# Model Embedding untuk mengubah teks menjadi vektor
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=GOOGLE_API_KEY)
# Inisialisasi ChromaDB lokal
vector_store = Chroma(collection_name="mandiri_report", embedding_function=embeddings, persist_directory="./chroma_db")

def extract_table_chart(page) -> str:
    """Merender halaman menjadi gambar dan meminta LLM mengekstrak tabel & grafik."""
    # Render halaman PDF menjadi gambar (Pixmap) dengan resolusi sedang (DPI 150 cukup)
    pix = page.get_pixmap(dpi=150)
    image_bytes = pix.tobytes("png")
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    
    # Prompt khusus untuk mengabaikan teks biasa dan fokus pada data visual/tabular
    prompt = """
    Anda adalah asisten data extraction dengan penglihatan yang sangat detail. Analisis gambar halaman laporan ini:
    
    1. Jika terdapat TABEL: ekstrak seluruh isinya ke dalam format Markdown.
    2. Jika terdapat GRAFIK/CHART: Anda WAJIB menganalisisnya dengan langkah berikut:
       - Langkah A: Identifikasi secara akurat nama kategori di legenda beserta warnanya (Misal: "Giro = Biru Muda", "Tabungan = Biru Tua"). Perhatikan baik-baik perbedaan gradasi warna.
       - Langkah B: Lihat ke dalam grafik, cari potongan warna yang SESUAI dengan warna di legenda tersebut.
       - Langkah C: Tuliskan persentase atau angka yang menempel pada warna tersebut. Jangan sampai angka biru muda tertukar ke biru tua!
    3. Abaikan paragraf teks biasa.
    4. Jika tidak ada tabel atau grafik, balas: "TIDAK ADA VISUAL".
    """
    
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
        ]
    )
    
    try:
        response = llm.invoke([message])
        if "TIDAK ADA VISUAL" in response.content:
            return "" # Kosongkan jika tidak ada visual
        return response.content
    except Exception as e:
        print(f"Error Vision LLM: {e}")
        return ""

async def process_and_ingest(file: UploadFile) -> dict:
    temp_pdf_path = f"temp_{file.filename}"
    with open(temp_pdf_path, "wb") as f:
        f.write(await file.read())

    documents = []
    doc = fitz.open(temp_pdf_path)
    
    # Folder menyimpan txt (debugging saja)
    debug_folder = "ingestion"
    os.makedirs(debug_folder, exist_ok=True)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Ekstrak Teks Standar
        raw_text = page.get_text("text")
        
        # Ekstrak Visual & Tabel via Rendered Image (Mengatasi vector & borderless table)
        table_chart_data = extract_table_chart(page)
        
        # Gabungkan konteks
        combined_page_content = f"--- DATA TABEL & GRAFIK ---\n{table_chart_data}\n\n--- TEKS HALAMAN ---\n{raw_text}"
        
        # Hasil Txt (untuk debugging saja)
        with open(f"{debug_folder}/page_{page_num + 1}.txt", "w", encoding="utf-8") as f:
            f.write(combined_page_content)
            
        doc_obj = Document(
            page_content=combined_page_content,
            metadata={"page_number": page_num + 1, "source": file.filename}
        )
        documents.append(doc_obj)

    doc.close()
    os.remove(temp_pdf_path)

    # Chunking & Embedding
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=6000,
        chunk_overlap=500,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
        
    vector_store.add_documents(chunks)

    return {
        "message": "Dokumen berhasil diproses dan disimpan ke Vector Database.",
        "total_pages_processed": len(documents),
        "total_chunks_created": len(chunks)
    }