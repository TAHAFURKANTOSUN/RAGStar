"""pgvector: sema, indeksleme ve arama (vector / fts / hybrid).

OpenSearch tarafiyla esdeger kurulum:
  vector -> HNSW, cosine  (OpenSearch: hnsw / cosinesimil / lucene)
  fts    -> GIN + tsvector (OpenSearch: BM25 'english' analyzer)
  hybrid -> RRF fuzyonu    (OpenSearch: hybrid query + search pipeline)

NOT: Postgres'in ts_rank'i BM25 DEGILDIR (dokuman uzunlugu normalizasyonu ve
saturasyon davranisi farklidir). Bu yuzden hibritte sira-tabanli RRF kullaniyoruz;
ham skorlari toplamak iki sistemi haksiz sekilde ayirirdi.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

import pg_config as pcfg


class ConnectionProblem(RuntimeError):
    """Baglanti kurulamadi; mesaj ne yapilacagini soyler."""


# ---------------------------------------------------------------- client
def connect(verbose: bool = True) -> psycopg.Connection:
    try:
        conn = psycopg.connect(pcfg.PG_DSN, autocommit=True)
    except Exception as exc:  # noqa: BLE001
        raise ConnectionProblem(_diagnose(exc)) from exc

    try:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    except Exception as exc:  # noqa: BLE001
        raise ConnectionProblem(
            f"'vector' uzantisi kurulamadi: {exc}\n"
            "  pgvector iceren bir imaj kullanin: pgvector/pgvector:pg17\n"
            "  (duz 'postgres' imajinda uzanti yoktur)"
        ) from exc

    register_vector(conn)

    if verbose:
        ver = conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()
        pg = conn.execute("SHOW server_version").fetchone()
        print(
            f"[pg] baglanti tamam: {pcfg.PG_HOST}:{pcfg.PG_PORT}/{pcfg.PG_DB} "
            f"(PostgreSQL {pg[0]}, pgvector {ver[0] if ver else '?'})"
        )
    return conn


def _diagnose(exc: Exception) -> str:
    s = str(exc)
    head = (
        f"PostgreSQL'e baglanilamadi: {pcfg.PG_HOST}:{pcfg.PG_PORT}/{pcfg.PG_DB}\n"
        f"  Ham hata: {s[:160]}\n"
    )
    if "Connection refused" in s or "could not connect" in s:
        return head + (
            "\n  TESHIS: Bu adreste dinleyen bir servis yok.\n\n"
            "      cd pgvector && docker compose up -d\n"
            "      docker compose logs -f postgres"
        )
    if "password authentication failed" in s or "authentication" in s:
        return head + (
            "\n  TESHIS: Kullanici adi/parola hatali.\n\n"
            "  .env -> PG_USER / PG_PASSWORD (compose varsayilani: rag / ragpass)"
        )
    if "does not exist" in s:
        return head + (
            "\n  TESHIS: Veritabani yok.\n\n"
            "  .env -> PG_DB (compose varsayilani: ragbench)"
        )
    return head


# ------------------------------------------------------------------ sema
def recreate_table(conn: psycopg.Connection, table: Optional[str] = None) -> str:
    table = table or pcfg.PG_TABLE
    conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.execute(
        f"""
        CREATE TABLE {table} (
            chunk_id        text PRIMARY KEY,
            doc_id          text,
            passage_id      integer,
            position        integer,
            original_text   text NOT NULL,
            context_prefix  text,
            contextual_text text NOT NULL,
            embedding       vector({pcfg.EMBEDDING_DIM}) NOT NULL,
            tsv tsvector GENERATED ALWAYS AS (
                to_tsvector('{pcfg.FTS_CONFIG}', contextual_text)
            ) STORED
        )
        """
    )
    print(f"[pg] '{table}' tablosu olusturuldu (dim={pcfg.EMBEDDING_DIM}).")
    return table


def create_indexes(conn: psycopg.Connection, table: Optional[str] = None) -> Dict[str, float]:
    """Index'leri VERI YUKLENDIKTEN SONRA kurar (HNSW icin cok daha hizli)."""
    table = table or pcfg.PG_TABLE
    timings: Dict[str, float] = {}

    t0 = time.perf_counter()
    conn.execute(
        f"""
        CREATE INDEX {table}_embedding_hnsw ON {table}
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = {pcfg.HNSW_M}, ef_construction = {pcfg.HNSW_EF_CONSTRUCTION})
        """
    )
    timings["hnsw"] = time.perf_counter() - t0
    print(f"[pg] HNSW index kuruldu ({timings['hnsw']:.1f}s)")

    t0 = time.perf_counter()
    conn.execute(f"CREATE INDEX {table}_tsv_gin ON {table} USING GIN (tsv)")
    timings["gin"] = time.perf_counter() - t0
    print(f"[pg] GIN (tam metin) index kuruldu ({timings['gin']:.1f}s)")

    conn.execute(f"ANALYZE {table}")
    return timings


def table_size(conn: psycopg.Connection, table: Optional[str] = None) -> Dict[str, Any]:
    table = table or pcfg.PG_TABLE
    row = conn.execute(
        f"""
        SELECT pg_total_relation_size('{table}'),
               pg_relation_size('{table}'),
               (SELECT count(*) FROM {table})
        """
    ).fetchone()
    return {"total_bytes": row[0], "table_bytes": row[1], "count": row[2]}


# -------------------------------------------------------------- indexing
def bulk_insert(
    conn: psycopg.Connection,
    records: Sequence[Dict[str, Any]],
    table: Optional[str] = None,
    batch_size: int = 500,
) -> int:
    """COPY ile toplu yukleme (INSERT'ten belirgin sekilde hizli)."""
    table = table or pcfg.PG_TABLE
    cols = (
        "chunk_id",
        "doc_id",
        "passage_id",
        "position",
        "original_text",
        "context_prefix",
        "contextual_text",
        "embedding",
    )
    n = 0
    with conn.cursor() as cur:
        with cur.copy(f"COPY {table} ({','.join(cols)}) FROM STDIN (FORMAT BINARY)") as copy:
            copy.set_types(
                ["text", "text", "integer", "integer", "text", "text", "text", "vector"]
            )
            for rec in records:
                emb = rec["embedding"]
                if not isinstance(emb, np.ndarray):
                    emb = np.asarray(emb, dtype=np.float32)
                copy.write_row(
                    (
                        rec["chunk_id"],
                        rec.get("doc_id"),
                        rec.get("passage_id"),
                        rec.get("position"),
                        rec["original_text"],
                        rec.get("context_prefix") or "",
                        rec["contextual_text"],
                        emb,
                    )
                )
                n += 1
    return n


# ---------------------------------------------------------------- search
_SELECT = "chunk_id, doc_id, passage_id, original_text, context_prefix, contextual_text"


def set_ef_search(conn: psycopg.Connection, ef: Optional[int] = None) -> None:
    conn.execute(f"SET hnsw.ef_search = {ef or pcfg.HNSW_EF_SEARCH}")


def vector_search(
    conn: psycopg.Connection,
    query_vector: np.ndarray,
    k: int = 10,
    table: Optional[str] = None,
) -> List[Dict[str, Any]]:
    table = table or pcfg.PG_TABLE
    rows = conn.execute(
        f"""
        SELECT {_SELECT}, 1 - (embedding <=> %s) AS score
        FROM {table}
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        (query_vector, query_vector, k),
    ).fetchall()
    return _rows_to_dicts(rows)


def fts_search(
    conn: psycopg.Connection,
    query: str,
    k: int = 10,
    table: Optional[str] = None,
) -> List[Dict[str, Any]]:
    table = table or pcfg.PG_TABLE
    rows = conn.execute(
        f"""
        SELECT {_SELECT}, ts_rank_cd(tsv, q) AS score
        FROM {table}, websearch_to_tsquery('{pcfg.FTS_CONFIG}', %s) q
        WHERE tsv @@ q
        ORDER BY score DESC
        LIMIT %s
        """,
        (query, k),
    ).fetchall()
    return _rows_to_dicts(rows)


def hybrid_search(
    conn: psycopg.Connection,
    query: str,
    query_vector: np.ndarray,
    k: int = 10,
    table: Optional[str] = None,
    candidates: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """RRF: score = w_v/(K+rank_v) + w_f/(K+rank_f)

    Iki alt sorgu da 'candidates' kadar aday cikarir, FULL OUTER JOIN ile
    birlestirilir; birinde olup digerinde olmayan sonuclar kaybolmaz.
    """
    table = table or pcfg.PG_TABLE
    cand = candidates or max(k * 5, 50)

    rows = conn.execute(
        f"""
        WITH vec AS (
            SELECT chunk_id, ROW_NUMBER() OVER (ORDER BY embedding <=> %s) AS rnk
            FROM {table}
            ORDER BY embedding <=> %s
            LIMIT %s
        ),
        fts AS (
            SELECT chunk_id,
                   ROW_NUMBER() OVER (ORDER BY ts_rank_cd(tsv, q) DESC) AS rnk
            FROM {table}, websearch_to_tsquery('{pcfg.FTS_CONFIG}', %s) q
            WHERE tsv @@ q
            ORDER BY ts_rank_cd(tsv, q) DESC
            LIMIT %s
        ),
        fused AS (
            SELECT COALESCE(vec.chunk_id, fts.chunk_id) AS chunk_id,
                   COALESCE(%s / (%s + vec.rnk), 0)
                 + COALESCE(%s / (%s + fts.rnk), 0) AS score
            FROM vec FULL OUTER JOIN fts USING (chunk_id)
        )
        SELECT c.chunk_id, c.doc_id, c.passage_id, c.original_text,
               c.context_prefix, c.contextual_text, f.score
        FROM fused f JOIN {table} c USING (chunk_id)
        ORDER BY f.score DESC
        LIMIT %s
        """,
        (
            query_vector,
            query_vector,
            cand,
            query,
            cand,
            pcfg.PG_VECTOR_WEIGHT,
            pcfg.RRF_K,
            pcfg.PG_FTS_WEIGHT,
            pcfg.RRF_K,
            k,
        ),
    ).fetchall()
    return _rows_to_dicts(rows)


def _rows_to_dicts(rows) -> List[Dict[str, Any]]:
    keys = (
        "chunk_id",
        "doc_id",
        "passage_id",
        "original_text",
        "context_prefix",
        "contextual_text",
        "score",
    )
    return [dict(zip(keys, r)) for r in rows]
