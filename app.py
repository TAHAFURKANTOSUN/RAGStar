import os
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PyMuPDFReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.opensearch import (
    OpensearchVectorStore,
    OpensearchVectorClient,
)
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
# OpenAI yerine Ollama paketini yüklüyoruz
from llama_index.llms.ollama import Ollama

# ==========================================
# 1. MODEL VE AYARLARIN YAPILANDIRILMASI
# ==========================================

# Embedding Modeli (Metinleri vektöre dönüştürür)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-m3",
    device="cuda"  # GPU yoksa "cpu" yapabilirsiniz
)

# LLM Ayarı: Ollama üzerinde çalışan Llama 3.1 8B
Settings.llm = Ollama(
    model="llama3.1",        # veya "llama3.1:8b" (ollama list çıktısındaki adı)
    base_url="http://localhost:11434",
    request_timeout=180.0,   # Büyük yanıtlarda zaman aşımına uğramaması için
    context_window=8000      # Llama 3.1 geniş bağlam penceresini destekler
)

# Metin Bölümleme (Chunking) Stratejisi
Settings.node_parser = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50
)

# ==========================================
# 2. PDF OKUMA (PyMuPDF)
# ==========================================
print("📄 PDF belgesi yükleniyor...")
reader = PyMuPDFReader()
documents = reader.load_data(file_path="./pollen.pdf")

# ==========================================
# 3. OPENSEARCH BAĞLANTISI VE VEKTÖR DEPOSU
# ==========================================
opensearch_endpoint = "http://localhost:9200"
index_name = "rag_bge_llama3"

client = OpensearchVectorClient(
    endpoint=opensearch_endpoint,
    idx=index_name,
    dim=1024, # BGE-M3 boyutu
    text_field="content",
    embedding_field="embedding"
)

vector_store = OpensearchVectorStore(client)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# ==========================================
# 4. İNDEKS OLUŞTURMA
# ==========================================
print("⚡ İndeks oluşturuluyor ve OpenSearch'e aktarılıyor...")
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True
)

# ==========================================
# 5. RERANKER (BGE-Reranker)
# ==========================================
reranker = FlagEmbeddingReranker(
    model="BAAI/bge-reranker-large",
    top_n=3,          # Llama 3.1'e gönderilecek en alakalı 3 bölüm
    use_fp16=True
)

# ==========================================
# 6. SORGULAMA MOTORU
# ==========================================
query_engine = index.as_query_engine(
    similarity_top_k=15, # OpenSearch'ten ilk aşamada çekilecek aday sayısı
    node_postprocessors=[reranker]
)

# ==========================================
# 7. SORU SORMA
# ==========================================
query = "Dokümandaki ana konular ve dikkat edilmesi gereken şartlar nelerdir?"
print(f"\n❓ Soru: {query}\n")

response = query_engine.query(query)

print("🦙 Llama 3.1 Yanıtı:")
print(response)
