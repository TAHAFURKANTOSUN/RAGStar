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
from llama_index.llms.openai import OpenAI # Veya Ollama, Anthropic vb.

# ==========================================
# 1. MODEL VE AYARLARIN YAPILANDIRILMASI
# ==========================================

# Embedding Modeli (Çok dilli destek için bge-m3 idealdir)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-m3",
    device="cuda" # GPU yoksa "cpu" yapabilirsiniz
)

# LLM Ayarı (Örn: OpenAI veya Ollama / Local LLM)
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)

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
# OpenSearch endpoint bilgileri
opensearch_endpoint = "http://localhost:9200" # veya https URL'iniz
index_name = "rag_bge_demo"

# BGE-M3 çıktı boyutu 1024'tür (bge-large-en kullanırsanız 1024, bge-base 768'dir)
client = OpensearchVectorClient(
    endpoint=opensearch_endpoint,
    idx=index_name,
    dim=1024,
    text_field="content",
    embedding_field="embedding",
    # OpenSearch tarafında Hybrid arama aktifse:
    # search_pipeline="hybrid-search-pipeline" 
)

vector_store = OpensearchVectorStore(client)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# ==========================================
# 4. İNDEKS OLUŞTURMA (Indexing)
# ==========================================
print("⚡ İndeks oluşturuluyor ve OpenSearch'e aktarılıyor...")
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True
)

# ==========================================
# 5. RERANKER (BGE-Reranker) TANIMLAMA
# ==========================================
# Ilk etapta OpenSearch'ten 10-20 dokuman çekip, Reranker ile en iyi 3 tanesini seçeceğiz.
reranker = FlagEmbeddingReranker(
    model="BAAI/bge-reranker-large",
    top_n=3,          # LLM'e gidecek nihai doküman sayısı
    use_fp16=True     # GPU belleğini optimize etmek için
)

# ==========================================
# 6. SORU-CEVAP SORGULAMA MOTORU (Query Engine)
# ==========================================
query_engine = index.as_query_engine(
    similarity_top_k=15, # OpenSearch'ten ilk aşamada çekilecek aday sayısı
    node_postprocessors=[reranker] # Çekilen adayları yeniden sıralayan katman
)

# ==========================================
# 7. SORGULAMA
# ==========================================
query = "Sözleşmedeki fesih şartları ve ceza koşulları nelerdir?"
print(f"\n❓ Soru: {query}\n")

response = query_engine.query(query)

print("🤖 Yanıt:")
print(response)

print("\n🔍 LLM'e Gönderilen En Alakalı Bağlamlar (Rerank Sonrası):")
for node in response.source_nodes:
    print(f"- [Skor: {node.score:.4f}] {node.node.get_content()[:150]}...")