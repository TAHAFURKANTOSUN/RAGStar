"""pgvector tarafinin konfigurasyonu.

Embedding boyutu, sorgu talimati gibi ORTAK ayarlar ana projedeki config.py'den
gelir - iki sistemin ayni vektor uzayini kullandigindan emin olmak icin.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as base  # noqa: E402  (ana projenin config'i)

# Ana .env dosyasi zaten base tarafindan yuklendi; buradaki degiskenler de okunur.

# ------------------------------------------------------------- baglanti
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DB = os.getenv("PG_DB", "ragbench")
PG_USER = os.getenv("PG_USER", "rag")
PG_PASSWORD = os.getenv("PG_PASSWORD", "ragpass")

PG_DSN = os.getenv(
    "PG_DSN",
    f"host={PG_HOST} port={PG_PORT} dbname={PG_DB} user={PG_USER} password={PG_PASSWORD}",
)

PG_TABLE = os.getenv("PG_TABLE", "chunks")

# ---------------------------------------------------------------- index
# HNSW parametreleri - OpenSearch tarafindaki degerlerle ayni tutuldu ki
# karsilastirma ANN ayarlarindan degil sistemden kaynaklansin.
HNSW_M = int(os.getenv("HNSW_M", "16"))
HNSW_EF_CONSTRUCTION = int(os.getenv("HNSW_EF_CONSTRUCTION", "256"))
HNSW_EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", "100"))

# Tam metin arama dili (to_tsvector konfigurasyonu)
FTS_CONFIG = os.getenv("FTS_CONFIG", "english")

# ------------------------------------------------------------- hibrit
# Postgres tarafinda skorlari RRF ile birlestiriyoruz: ts_rank ile kosinus
# benzerligi ayni olcekte olmadigi icin sira-tabanli fuzyon daha saglikli.
RRF_K = int(os.getenv("RRF_K", "60"))
PG_FTS_WEIGHT = float(os.getenv("PG_FTS_WEIGHT", str(base.BM25_WEIGHT)))
PG_VECTOR_WEIGHT = float(os.getenv("PG_VECTOR_WEIGHT", str(base.KNN_WEIGHT)))

# --------------------------------------------------------------- ortak
EMBEDDING_DIM = base.EMBEDDING_DIM
EMBEDDING_MODEL = base.EMBEDDING_MODEL
QUERY_INSTRUCTION = base.QUERY_INSTRUCTION
OPENSEARCH_INDEX = base.INDEX_NAME
