import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Inisialisasi Model & DB (Gunakan model terbaru yang sudah Anda verifikasi)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=GOOGLE_API_KEY, temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=GOOGLE_API_KEY)
vector_store = Chroma(collection_name="mandiri_report", embedding_function=embeddings, persist_directory="./chroma_db")

# Definisikan Custom Prompt untuk menjawab pertanyaan dengan konteks yang diberikan
prompt_template = """
Anda adalah asisten AI profesional yang membantu menganalisis Laporan Keuangan Bank Mandiri 2025. 
Gunakan potongan konteks berikut untuk menjawab pertanyaan di akhir. 

Jika Anda tidak tahu jawabannya berdasarkan konteks yang diberikan, katakan saja bahwa Anda tidak tahu, jangan mencoba mengarang jawaban.
Selalu jawab dalam bahasa Indonesia yang formal dan mudah dipahami.

KONTEKS:
{context}

PERTANYAAN: 
{input}

JAWABAN:
"""

def get_answer_from_query(question: str):
    # Mencari 5 chunk paling relevan
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Membuat Chain untuk menggabungkan dokumen ke prompt
    combine_docs_chain = create_stuff_documents_chain(
        llm, 
        PromptTemplate.from_template(prompt_template)
    )

    # Membuat Retrieval Chain utama
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    # Jalankan Query
    response = rag_chain.invoke({"input": question})

    # Ekstraksi Jawaban dan Metadata Halaman
    answer = response["answer"]
    # Mengambil nomor halaman unik dari dokumen yang ditemukan
    sources = list(set([doc.metadata.get("page_number") for doc in response["context"]]))
    sources.sort()

    return {
        "answer": answer,
        "source_pages": sources
    }