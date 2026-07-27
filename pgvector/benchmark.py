"""OpenSearch vs pgvector karsilastirmali benchmark.

Olculenler
----------
Kalite : recall@k, MRR  (proxy metrik - asagidaki nota bakin)
Hiz    : sorgu gecikmesi p50 / p95 / ortalama, QPS
Boyut  : index / tablo disk kullanimi
Ortusme: iki sistemin top-k sonuclarinin ne kadar ortustugu

Adil karsilastirma icin
-----------------------
* Sorgu embedding'leri BIR KEZ hesaplanir, iki sisteme de ayni vektor gider.
* Gecikme olcumu yalnizca arama cagrisini kapsar (embedding haric).
* Her kosudan once isinma turu yapilir (JIT, cache, baglanti kurulumu).
* pg_ingest.py --from-opensearch kullandiysaniz iki sistemde birebir ayni
  metin ve vektorler vardir; fark yalnizca arama motorundan gelir.

Kullanim
--------
    python benchmark.py                       # 200 soru, tum modlar
    python benchmark.py -n 500 -k 1 5 10 20
    python benchmark.py --systems pgvector    # tek sistem
    python benchmark.py --repeat 3            # gecikme icin 3 tur
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pg_config as pcfg  # noqa: E402
import pg_store as pg  # noqa: E402

# ana projeden yeniden kullanilanlar
import config as base  # noqa: E402
import embedder  # noqa: E402
import evaluate as ev  # noqa: E402  (normalize / is_relevant / build_queries)
import opensearch_store as store  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# --------------------------------------------------------------- olcum
def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, q))


def run_suite(
    name: str,
    search_fn: Callable[[int], List[Dict[str, Any]]],
    qa: List[Dict[str, str]],
    ks: List[int],
    repeat: int,
) -> Dict[str, Any]:
    """search_fn(i) -> i. sorgunun sonuclari. Kalite + gecikme olcer."""
    max_k = max(ks)
    hits_at = {k: 0 for k in ks}
    rr_sum = 0.0
    latencies: List[float] = []
    topk_ids: List[List[str]] = []

    # isinma
    for i in range(min(5, len(qa))):
        search_fn(i)

    for i, row in enumerate(tqdm(qa, desc=name, leave=False)):
        results: List[Dict[str, Any]] = []
        for r in range(repeat):
            t0 = time.perf_counter()
            results = search_fn(i)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        topk_ids.append([r["chunk_id"] for r in results[:max_k]])

        rank = next(
            (j for j, h in enumerate(results, 1) if ev.is_relevant(row["answer"], h)), None
        )
        if rank:
            rr_sum += 1.0 / rank
            for k in ks:
                if rank <= k:
                    hits_at[k] += 1

    n = len(qa)
    total_secs = sum(latencies) / 1000.0
    return {
        "name": name,
        **{f"recall@{k}": hits_at[k] / n for k in ks},
        "mrr": rr_sum / n,
        "p50_ms": _percentile(latencies, 50),
        "p95_ms": _percentile(latencies, 95),
        "mean_ms": statistics.fmean(latencies) if latencies else 0.0,
        "qps": (len(latencies) / total_secs) if total_secs else 0.0,
        "_topk": topk_ids,
    }


def overlap_at_k(a: List[List[str]], b: List[List[str]], k: int) -> float:
    """Iki sistemin top-k sonuclarinin ortalama ortusme orani."""
    if not a or not b:
        return 0.0
    scores = []
    for ra, rb in zip(a, b):
        sa, sb = set(ra[:k]), set(rb[:k])
        if not sa and not sb:
            continue
        scores.append(len(sa & sb) / max(len(sa | sb), 1))
    return statistics.fmean(scores) if scores else 0.0


# ------------------------------------------------------------- raporlama
def format_table(rows: List[Dict[str, Any]], ks: List[int]) -> str:
    cols = (
        ["sistem / mod"]
        + [f"recall@{k}" for k in ks]
        + ["mrr", "p50 ms", "p95 ms", "QPS"]
    )
    widths = [max(len(cols[0]), max(len(r["name"]) for r in rows))] + [
        max(len(c), 9) for c in cols[1:]
    ]

    def line(cells: List[str]) -> str:
        return "  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(cells, widths)))

    out = [line(cols), "-" * (sum(widths) + 2 * (len(widths) - 1))]
    for r in rows:
        out.append(
            line(
                [r["name"]]
                + [f"{r[f'recall@{k}']:.4f}" for k in ks]
                + [
                    f"{r['mrr']:.4f}",
                    f"{r['p50_ms']:.1f}",
                    f"{r['p95_ms']:.1f}",
                    f"{r['qps']:.1f}",
                ]
            )
        )
    return "\n".join(out)


def markdown_table(rows: List[Dict[str, Any]], ks: List[int]) -> str:
    head = ["sistem / mod"] + [f"recall@{k}" for k in ks] + ["MRR", "p50 ms", "p95 ms", "QPS"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = (
            [r["name"]]
            + [f"{r[f'recall@{k}']:.4f}" for k in ks]
            + [f"{r['mrr']:.4f}", f"{r['p50_ms']:.1f}", f"{r['p95_ms']:.1f}", f"{r['qps']:.1f}"]
        )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_report(rows: List[Dict[str, Any]], ks: List[int], meta: Dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RESULTS_DIR / f"benchmark-{stamp}.md"

    body = [
        f"# OpenSearch vs pgvector — {stamp}",
        "",
        "## Kurulum",
        "",
        f"- Embedding modeli: `{base.EMBEDDING_MODEL}` ({base.EMBEDDING_DIM} boyut)",
        f"- Soru sayisi: {meta['n_queries']} (yes/no elendi)",
        f"- Tur sayisi (gecikme icin): {meta['repeat']}",
        f"- HNSW: m={pcfg.HNSW_M}, ef_construction={pcfg.HNSW_EF_CONSTRUCTION}, "
        f"ef_search={pcfg.HNSW_EF_SEARCH}",
        "",
        "## Sonuclar",
        "",
        markdown_table(rows, ks),
        "",
    ]

    if meta.get("sizes"):
        body += ["## Disk kullanimi", ""]
        for label, mb in meta["sizes"].items():
            body.append(f"- {label}: {mb:.1f} MB")
        body.append("")

    if meta.get("overlap"):
        body += [
            "## Sonuc ortusmesi (hybrid)",
            "",
            "Iki sistemin ayni sorguya dondurdugu top-k kumelerinin Jaccard benzerligi.",
            "",
        ]
        for k, v in meta["overlap"].items():
            body.append(f"- top-{k}: {v:.3f}")
        body.append("")

    body += [
        "## Notlar",
        "",
        "- Kalite metrigi proxy'dir: dataset'te gold passage etiketi yok, bir sonuc",
        "  cevap metnini iceriyorsa ilgili sayiliyor. Mutlak degerlerden cok",
        "  satirlar arasi FARKA bakin.",
        "- Postgres `ts_rank_cd` BM25 degildir; leksik siralama davranisi",
        "  OpenSearch BM25'ten farklidir. Hibritte bu yuzden sira-tabanli RRF",
        "  kullaniliyor.",
        "- Gecikmeler tek istemciden, seri olarak olculdu; es zamanli yuk altindaki",
        "  davranisi yansitmaz.",
    ]

    path.write_text("\n".join(body), encoding="utf-8")
    return path


# ------------------------------------------------------------------ main
def main() -> None:
    p = argparse.ArgumentParser(description="OpenSearch vs pgvector benchmark")
    p.add_argument("-n", type=int, default=200, help="Soru sayisi (0 = hepsi).")
    p.add_argument("-k", nargs="+", type=int, default=[1, 5, 10])
    p.add_argument(
        "--systems",
        nargs="+",
        default=["opensearch", "pgvector"],
        choices=["opensearch", "pgvector"],
    )
    p.add_argument(
        "--modes",
        nargs="+",
        default=["lexical", "vector", "hybrid"],
        choices=["lexical", "vector", "hybrid"],
        help="lexical = BM25 / ts_rank, vector = kNN / pgvector",
    )
    p.add_argument("--repeat", type=int, default=1, help="Her sorgu kac kez calissin.")
    p.add_argument("--index", default=base.INDEX_NAME, help="OpenSearch index'i.")
    p.add_argument("--table", default=pcfg.PG_TABLE, help="Postgres tablosu.")
    p.add_argument("--include-yesno", action="store_true")
    p.add_argument("--no-save", action="store_true", help="Rapor dosyasi yazma.")
    args = p.parse_args()

    ks = sorted(args.k)
    max_k = max(ks)

    qa = ev.build_queries(args.n or None, args.include_yesno)
    print(f"[bench] {len(qa)} soru, k={ks}, repeat={args.repeat}")

    # Sorgu embedding'leri BIR KEZ - iki sisteme de ayni vektorler
    print("[bench] sorgu embedding'leri hesaplaniyor...")
    qvecs = embedder.embed_queries([r["question"] for r in qa], show_progress=True)
    qvecs32 = [np.asarray(v, dtype=np.float32) for v in qvecs]
    qtexts = [r["question"] for r in qa]

    rows: List[Dict[str, Any]] = []
    sizes: Dict[str, float] = {}
    topk_by_system: Dict[str, List[List[str]]] = {}

    # ---------------------------------------------------------- OpenSearch
    if "opensearch" in args.systems:
        try:
            client, _ = store.connect(verbose=False)
        except store.ConnectionProblem as exc:
            print(f"\n{exc}\n")
            raise SystemExit(1)
        if not client.indices.exists(index=args.index):
            print(f"[bench] '{args.index}' yok. Once: python ingest.py")
            raise SystemExit(1)

        st = client.indices.stats(index=args.index)["_all"]["primaries"]["store"]
        sizes[f"OpenSearch `{args.index}`"] = st["size_in_bytes"] / 1024 / 1024

        fns = {
            "lexical": lambda i: store.bm25_search(client, qtexts[i], k=max_k, index=args.index),
            "vector": lambda i: store.knn_search(
                client, qvecs32[i].tolist(), k=max_k, index=args.index
            ),
            "hybrid": lambda i: store.hybrid_search(
                client, qtexts[i], qvecs32[i].tolist(), k=max_k, index=args.index
            ),
        }
        for mode in args.modes:
            label = {"lexical": "bm25", "vector": "knn", "hybrid": "hybrid"}[mode]
            res = run_suite(f"opensearch / {label}", fns[mode], qa, ks, args.repeat)
            if mode == "hybrid":
                topk_by_system["opensearch"] = res["_topk"]
            rows.append(res)

    # ------------------------------------------------------------ pgvector
    if "pgvector" in args.systems:
        try:
            conn = pg.connect(verbose=False)
        except pg.ConnectionProblem as exc:
            print(f"\n{exc}\n")
            raise SystemExit(1)

        exists = conn.execute("SELECT to_regclass(%s)", (args.table,)).fetchone()[0]
        if not exists:
            print(f"[bench] '{args.table}' tablosu yok. Once: python pg_ingest.py")
            raise SystemExit(1)
        pg.set_ef_search(conn)

        size = pg.table_size(conn, args.table)
        sizes[f"pgvector `{args.table}`"] = size["total_bytes"] / 1024 / 1024

        fns = {
            "lexical": lambda i: pg.fts_search(conn, qtexts[i], k=max_k, table=args.table),
            "vector": lambda i: pg.vector_search(conn, qvecs32[i], k=max_k, table=args.table),
            "hybrid": lambda i: pg.hybrid_search(
                conn, qtexts[i], qvecs32[i], k=max_k, table=args.table
            ),
        }
        for mode in args.modes:
            label = {"lexical": "ts_rank", "vector": "vector", "hybrid": "hybrid"}[mode]
            res = run_suite(f"pgvector / {label}", fns[mode], qa, ks, args.repeat)
            if mode == "hybrid":
                topk_by_system["pgvector"] = res["_topk"]
            rows.append(res)

    # ------------------------------------------------------------- ciktilar
    print("\n" + format_table(rows, ks))

    if sizes:
        print("\nDisk kullanimi:")
        for label, mb in sizes.items():
            print(f"  {label}: {mb:.1f} MB")

    overlap: Dict[int, float] = {}
    if len(topk_by_system) == 2:
        a, b = topk_by_system["opensearch"], topk_by_system["pgvector"]
        overlap = {k: overlap_at_k(a, b, k) for k in ks}
        print("\nHybrid sonuc ortusmesi (Jaccard):")
        for k, v in overlap.items():
            print(f"  top-{k}: {v:.3f}")

    if not args.no_save:
        path = save_report(
            rows,
            ks,
            {
                "n_queries": len(qa),
                "repeat": args.repeat,
                "sizes": sizes,
                "overlap": overlap,
            },
        )
        print(f"\nRapor: {path}")

    print(
        "\nNot: kalite metrigi proxy (cevap metni sonucta geciyor mu). "
        "Sistemler arasi FARKA bakin, mutlak degere degil."
    )


if __name__ == "__main__":
    main()
