"""pgvector arama CLI'i.

    python pg_search.py "Who assassinated Lincoln?"
    python pg_search.py "beetle defense" --mode vector -k 5
    python pg_search.py                     # interaktif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pg_config as pcfg  # noqa: E402
import pg_store as pg  # noqa: E402


def run(conn, query: str, mode: str, k: int, table: str) -> List[Dict[str, Any]]:
    if mode == "fts":
        return pg.fts_search(conn, query, k=k, table=table)

    import embedder
    import numpy as np

    vec = np.asarray(embedder.embed_query(query), dtype=np.float32)
    if mode == "vector":
        return pg.vector_search(conn, vec, k=k, table=table)
    return pg.hybrid_search(conn, query, vec, k=k, table=table)


def show(results: List[Dict[str, Any]]) -> None:
    if not results:
        print("  (sonuc yok)")
        return
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] score={r['score']:.4f}  {r['chunk_id']}  (passage {r['passage_id']})")
        if r.get("context_prefix"):
            print(f"    ctx : {r['context_prefix']}")
        text = r["original_text"].replace("\n", " ")
        print(f"    text: {text[:300]}{'...' if len(text) > 300 else ''}")


def main() -> None:
    p = argparse.ArgumentParser(description="pgvector arama")
    p.add_argument("query", nargs="?", help="Sorgu. Bos birakilirsa interaktif mod.")
    p.add_argument("--mode", choices=["hybrid", "vector", "fts"], default="hybrid")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--table", default=pcfg.PG_TABLE)
    args = p.parse_args()

    try:
        conn = pg.connect(verbose=False)
    except pg.ConnectionProblem as exc:
        print(f"\n{exc}\n")
        raise SystemExit(1)
    pg.set_ef_search(conn)

    if args.query:
        show(run(conn, args.query, args.mode, args.k, args.table))
        return

    print(f"Interaktif arama ({args.mode}, tablo={args.table}). Cikmak icin bos Enter.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        show(run(conn, q, args.mode, args.k, args.table))


if __name__ == "__main__":
    main()
