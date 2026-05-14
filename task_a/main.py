from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from src.ingestion import process_and_ingest_pdf
from src.query import get_answer_from_query

app = FastAPI(title="Multimodal RAG API - Bank Mandiri", version="1.0")

class QueryRequest(BaseModel):
    question: str

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """
    Endpoint untuk mengunggah file PDF.
    Sistem akan mengekstrak teks, tabel, dan gambar, lalu menyimpannya ke Vector DB.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Hanya file PDF yang diperbolehkan.")
    
    try:
        result = await process_and_ingest_pdf(file)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
def query_document(request: QueryRequest):
    """
    Endpoint untuk mengajukan pertanyaan terkait Laporan Keuangan Bank Mandiri 2025.
    Sistem akan mencari informasi relevan dari Vector DB dan memberikan jawaban beserta sumber halaman.
    """
    try:
        result = get_answer_from_query(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Multimodal RAG API Active. Go to /docs to test endpoints."}