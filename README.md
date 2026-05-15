# AI Engineer Intern Technical Test - SEAL Internship

This repository contains the complete technical assessment for the **AI Engineer Intern** position at SEAL. It demonstrates proficiency in building end-to-end **Multimodal RAG** pipelines and advanced **Computer Vision** systems for layout-aware text extraction.


## 📌 Project Overview

The project is divided into two specialized tasks, each focusing on distinct AI capabilities:

### **Task A: Multimodal Retrieval-Augmented Generation (RAG)**

A robust pipeline designed to ingest complex PDF documents—including text, borderless tables, and vector graphics—into a queryable knowledge base.

* **Input**: Bank Mandiri 2025 Annual Report.

* **Key Features**:
   * **Multimodal Parsing**: Uses Vision Language Models to interpret charts and extract tables into clean Markdown.
   * **Smart Retrieval**: Implements chunking strategies that preserve context for synthesis by an LLM.
   * **REST API**: Built with FastAPI to provide `/ingest` and `/query` endpoints.


### **Task B: Layout Aware Text Extraction (Computer Vision)**

A vision pipeline that converts static presentation slides into interactive, editable HTML pages.

* **Input**: Presentation slide images (.jpg/.png).
* **Key Features**:
   * **Style Extraction**: Automatically detects font sizes, colors, and bounding box coordinates.
   * **Inpainting Technology**: Removes original text from the background image to prevent visual ghosting/duplication.
   * **Interactive UI**: Overlays extracted text as editable `<span>` elements precisely over the original layout.


## 📂 Project Structure

```text
seal-ai-engineer-task
├─ task_a/                # Multimodal RAG Project
│  ├─ main.py             # FastAPI entry point
│  ├─ src/                # Core logic for ingestion and querying
│  └─ notebooks/          # R&D and experimentation
├─ task_b/                # Computer Vision Project
│  ├─ main.py             # Main extraction script
│  ├─ data/               # Input slide images
│  ├─ output/             # Generated interactive HTML files
│  └─ notebooks/          # R&D and experimentation
└─ README.md              # Main documentation

```


## 🛠️ Tech Stack

* **Language**: Python 3.10+ 
* **AI Orchestration**: LangChain / LangGraph 
* **Web Framework**: FastAPI 
* **Models**:
   * **LLM & Vision**: Google Gemini (Flash 2.5 Lite)
   * **OCR**: EasyOCR
* **Database**: ChromaDB (Vector Storage)
* **Image Processing**: OpenCV (Inpainting & Manipulation)


## 🚀 Getting Started

### 1. Installation

Clone the repository and install the dependencies for each task:

```bash
# Task A
cd task_a && pip install -r requirements.txt

# Task B
cd ../task_b && pip install -r requirements.txt
```

### 2. Configuration (Task A)

Create a `.env` file in the `task_a` directory:

```env
GOOGLE_API_KEY="your_api_key_here"
```

### 3. Execution

* **Task A (API)**: Run `uvicorn main:app --reload` and visit `http://localhost:8000/docs`.
* **Task B (Vision)**: Place images in `/data` and run `python main.py`.


## 🎓 Evaluation Criteria Met

* **Accuracy**: Successfully answers complex financial queries regarding POJK regulations, debt collection windows, and sectoral credit growth.
* **Metadata**: Includes source page tracking for all RAG responses to ensure auditability.
* **Fidelity**: Preserves original layout, font size, and text color in Task B's HTML output.



---

This project was created to fulfill the Technical Test for the SEAL Independent Internship Registration Batch 1 2026.