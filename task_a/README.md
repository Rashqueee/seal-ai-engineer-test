# Multimodal Retrieval-Augmented Generation (RAG)

A FastAPI-based REST API that implements an end-to-end Multimodal RAG pipeline. This system is capable of converting complex PDF documents—containing plain text, borderless tables, and vector graphics—into a queryable knowledge base, and accurately answering user questions while providing the exact source page metadata.

## 📂 Project Structure

```text
task_a/
├─ notebooks/        # Jupyter notebooks for experimentation and testing
├─ src/
│  ├─ ingestion.py   # Logic for PDF parsing, Vision extraction, and chunking
│  └─ query.py       # Logic for Vector retrieval and LLM synthesis
├─ .env              # Environment variables (API Keys)
├─ main.py           # FastAPI main application and endpoints
├─ requirements.txt  # List of required Python libraries
└─ README.md         # Project documentation

```

*(Note: A `chroma_db/` folder and `ingestion/` debug folder will be automatically generated upon running the ingestion endpoint).*

## ✨ Key Features

* **Full-Page Vision Extraction**: Bypasses traditional OCR limitations by rendering entire PDF pages as images and using Gemini Vision (`gemini-2.5-flash-lite`) to perfectly extract tables into Markdown and describe complex graphs.
* **Smart Context Formatting**: Employs a dual-context strategy (Vision Extraction + Raw Text) with prioritized prompt engineering to eliminate LLM hallucinations and data overlap.
* **End-to-End REST API**: Built with FastAPI, providing seamless endpoints for uploading documents (`/ingest`) and asking questions (`/query`).
* **Source Tracking**: Automatically retrieves and returns the exact page numbers used to synthesize the answer for easy auditing and debugging.

## 🚀 How to Run

### 1. Navigate to the Directory

```bash
cd task_a
```

### 2. Setup Environment Variables

Create a `.env` file in the `task_a/` directory (if you haven't already) and add your Google AI Studio API Key:

```env
GOOGLE_API_KEY="your_google_api_key_here"
```

### 3. Install Dependencies

It is highly recommended to use a virtual environment. Install the required packages using:

```bash
pip install -r requirements.txt
```

### 4. Run the Server

Start the FastAPI server using Uvicorn:

```bash
uvicorn main:app --reload
```

### 5. View and Test the API

1. Open your browser and navigate to the Swagger UI: **`http://localhost:8000/docs`**
2. **Ingest Document:** Use the `POST /ingest` endpoint to upload your PDF file (e.g., *Laporan Keuangan Bank Mandiri 2025.pdf*). Wait for the process to finish and populate the local ChromaDB.
3. **Query Document:** Use the `POST /query` endpoint to ask questions based on the document. Pass a JSON payload like this:
    ```json
    {
    "question": "Sebutkan presentase komposisi dana pihak ketiga (DPK) di Bank Mandiri pada tahun 2024 dan 2025?"
    }
    ```