"""pgvector'e ingest.

Iki kaynak:

  --from-opensearch  (VARSAYILAN)
      Chunk'lari ve embedding'leri dogrudan OpenSearch index'inden kopyalar.
      Benchmark icin dogru olan budur: iki sistemde birebir AYNI metin ve
      AYNI vektorler bulunur, fark yalnizca arama motorundan kaynaklanir.
      Gemini cagrisi yok, embedding yeniden hesaplanmaz.

  --from-dataset
      Dataset'ten bagimsiz kurar (dokuman gruplama -> on soz -> embedding).
      OpenSearch olmadan tek basina kullanmak icin.

Kullanim:
    python pg_ingest.py
    python pg_ingest.py --from-dataset --limit 500
    python pg_ingest.py --table chunks_baseline --os-index mini-wiki-contextual-baseline
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pg_config as pcfg  # noqa: E402
import pg_store as pg  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="pgvector ingest")
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--from-opensearch",
        action="store_true",
        default=True,
        help="Chunk+embedding'leri OpenSearch'ten kopyala (varsayilan).",
    )
    src.add_argument(
        "--from-dataset",
        action="store_true",
        help="Dataset'ten bastan kur (on soz + embedding uretilir).",
    )
    p.add_argument("--table", default=None, help="Hedef tablo adi.")
    p.add_argument("--os-index", default=None, help="Kaynak OpenSearch index'i.")
    p.add_argument("--limit", type=int, default=None, help="Ilk N kayit.")
    p.add_argument(
        "--no-context", action="store_true", help="(--from-dataset ile) on soz uretme."
    )
    return p.parse_args()


# ------------------------------------------------- kaynak: OpenSearch
def from_opensearch(index: str, limit: Optional[int]) -> Iterator[Dict]:
    from opensearchpy.helpers import scan

    import opensearch_store as store

    try:
        client, _ = store.connect(verbose=False)
    except store.ConnectionProblem as exc:
        print(f"\n{exc}\n")
        raise SystemExit(1)

    if not client.indices.exists(index=index):
        print(
            f"\n[pg] '{index}' index'i yok. Once ana projede ingest calistirin:\n"
            f"    python ingest.py\n"
        )
        raise SystemExit(1)

    total = client.count(index=index)["count"]
    print(f"[pg] OpenSearch '{index}' index'inden {total} kayit kopyalanacak.")

    n = 0
    for hit in scan(
        client,
        index=index,
        query={"query": {"match_all": {}}},
        preserve_order=False,
        size=500,
    ):
        src = hit["_source"]
        yield {
            "chunk_id": src["chunk_id"],
            "doc_id": src.get("doc_id"),
            "passage_id": src.get("passage_id"),
            "position": src.get("position"),
            "original_text": src["original_text"],
            "context_prefix": src.get("context_prefix") or "",
            "contextual_text": src["contextual_text"],
            "embedding": np.asarray(src["embedding"], dtype=np.float32),
        }
        n += 1
        if limit and n >= limit:
            break


# ---------------------------------------------------- kaynak: dataset
def from_dataset(limit: Optional[int], no_context: bool) -> List[Dict]:
    import dataset as ds
    import embedder

    passages = ds.load_passages()
    if limit:
        passages = passages[:limit]
    print(f"[pg] {len(passages)} passage yuklendi.")

    docs = ds.build_documents(passages)
    chunks = ds.all_chunks(docs)

    if not no_context:
        import contextualize

        try:
            contextualize.contextualize(docs, use_cache=True)
        except contextualize.QuotaExhausted:
            filled = sum(1 for c in chunks if c.context)
            print(f"[pg] Kota bitti, {filled}/{len(chunks)} on soz hazir. Durduruluyor.")
            raise SystemExit(1)

    print("[pg] embedding hesaplaniyor...")
    vectors = embedder.embed_passages([c.contextual_text for c in chunks])

    return [
        {
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "passage_id": c.passage_id,
            "position": c.position,
            "original_text": c.text,
            "context_prefix": c.context,
            "contextual_text": c.contextual_text,
            "embedding": vec.astype(np.float32),
        }
        for c, vec in zip(chunks, vectors)
    ]


def main() -> None:
    args = parse_args()
    table = args.table or pcfg.PG_TABLE
    t0 = time.time()

    # Once baglan - pahali islerden onceki hizli basarisizlik
    try:
        conn = pg.connect()
    except pg.ConnectionProblem as exc:
        print(f"\n{exc}\n")
        raise SystemExit(1)

    if args.from_dataset:
        records = from_dataset(args.limit, args.no_context)
    else:
        index = args.os_index or pcfg.OPENSEARCH_INDEX
        records = list(from_opensearch(index, args.limit))

    if not records:
        print("[pg] Kaynak bos, cikiliyor.")
        raise SystemExit(1)

    dim = len(records[0]["embedding"])
    if dim != pcfg.EMBEDDING_DIM:
        print(
            f"\n[pg] HATA: kaynak vektorler {dim} boyutlu ama tablo "
            f"{pcfg.EMBEDDING_DIM} bekliyor.\n"
            f"     .env -> EMBEDDING_DIM={dim} yapin.\n"
        )
        raise SystemExit(1)

    pg.recreate_table(conn, table)

    t_load = time.perf_counter()
    n = pg.bulk_insert(conn, records, table)
    load_secs = time.perf_counter() - t_load
    print(f"[pg] {n} kayit yuklendi ({load_secs:.1f}s).")

    idx_timings = pg.create_indexes(conn, table)

    size = pg.table_size(conn, table)
    print(
        f"[pg] tablo '{table}': {size['count']} satir, "
        f"{size['total_bytes'] / 1024 / 1024:.1f} MB (index'ler dahil)"
    )
    print(
        f"[pg] toplam {time.time() - t0:.1f}s "
        f"(yukleme {load_secs:.1f}s, hnsw {idx_timings['hnsw']:.1f}s, "
        f"gin {idx_timings['gin']:.1f}s)"
    )

    row = conn.execute(
        f"SELECT chunk_id, context_prefix, original_text FROM {table} LIMIT 1"
    ).fetchone()
    if row:
        print("\n--- ornek kayit ---")
        print(f"chunk_id : {row[0]}")
        print(f"prefix   : {row[1] or '(yok)'}")
        print(f"original : {row[2][:200]}...")


if __name__ == "__main__":
    main()
