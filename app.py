import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PyMuPDFReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.opensearch import OpensearchVectorStore, OpensearchVectorClient
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.llms.ollama import Ollama

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ==========================================
# 1. RAG AYARLARI VE İNDEKS KURULUMU
# ==========================================

# CUDA hatasını önlemek için device="cpu" ayarlı (GPU var ise "cuda" yapabilirsiniz)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-m3",
    device="cpu" 
)

Settings.llm = Ollama(
    model="llama3.1:8b ",
    base_url="http://localhost:11434",
    request_timeout=180.0
)

Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

PDF_PATH = os.getenv("RAG_PDF_PATH", "./pollen.pdf")
if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(
        f"PDF bulunamadı: {PDF_PATH}. Önce 'python dowloand_data.py' çalıştırın "
        f"veya RAG_PDF_PATH ortam değişkenini ayarlayın."
    )

print(f" PDF yükleniyor: {PDF_PATH}")
reader = PyMuPDFReader()
documents = reader.load_data(file_path=PDF_PATH)

opensearch_endpoint = "https://localhost:9200"
index_name = "rag_bge_web"

# OpenSearch güvenlik eklentisi varsayılan olarak açık gelir; kimlik bilgisi
# verilmezse her istek 401 Unauthorized döner. Kullanıcı adı/şifreyi ortam
# değişkenlerinden okuyoruz — koda gömmüyoruz.
opensearch_user = os.getenv("OPENSEARCH_USER", "admin")
opensearch_password = os.getenv("OPENSEARCH_PASSWORD")
if not opensearch_password:
    raise RuntimeError(
        "OPENSEARCH_PASSWORD ortam değişkeni ayarlanmamış. "
        "OpenSearch'e bağlanmak için kullanıcı adı/şifre gerekiyor "
        "(varsayılan dev kurulumunda güvenlik eklentisi açıktır)."
    )

client = OpensearchVectorClient(
    endpoint=opensearch_endpoint,
    index=index_name,
    dim=1024,
    text_field="content",
    embedding_field="embedding",
    http_auth=(opensearch_user, opensearch_password),
    verify_certs=False,        # self-signed sertifika kullanan localhost dev kurulumu için
    ssl_show_warn=False
)

vector_store = OpensearchVectorStore(client)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# NOT: Bu her process başlangıcında (uvicorn restart dahil) PDF'i yeniden
# embed edip OpenSearch'e yazar. OpenSearch'teki index kalıcı olduğundan
# bu, restart başına gereksiz işlem + olası duplike doküman anlamına gelir.
# Üretimde: index_name zaten var mı kontrol edip varsa from_documents yerine
# doğrudan VectorStoreIndex.from_vector_store(vector_store) kullanın.
print(" İndeks oluşturuluyor...")
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

reranker = FlagEmbeddingReranker(model="BAAI/bge-reranker-large", top_n=3)

query_engine = index.as_query_engine(
    similarity_top_k=15,
    node_postprocessors=[reranker]
)

# ==========================================
# 2. API VE SAYFA ROUTE'LARI
# ==========================================

class QueryRequest(BaseModel):
    question: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {})

@app.post("/api/query")
async def query_endpoint(data: QueryRequest):
    question = data.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Soru boş olamaz.")

    print(f" Gelen Soru: {question}")
    try:
        response = query_engine.query(question)
    except Exception as exc:
        print(f" Sorgu hatası: {exc}")
        raise HTTPException(
            status_code=502,
            detail="Sorgu işlenirken bir hata oluştu. Ollama/OpenSearch servislerinin çalıştığından emin olun.",
        )
    return {"answer": str(response)}