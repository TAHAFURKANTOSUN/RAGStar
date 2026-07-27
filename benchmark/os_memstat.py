"""
OpenSearch sunucu tarafı bellek anlık görüntüsü.
================================================

VectorDBBench RAM ölçmez. Bu script koşu öncesi ve sonrası OpenSearch'ten
bellek istatistiklerini çeker; report.py farkı alıp "Ek RAM Tüketimi" olarak
raporlar.

Ölçülenler:
  jvm_heap_used      JVM heap kullanımı (_nodes/stats)
  segments_memory    Lucene segment belleği (_nodes/stats)
  knn_graph_memory   HNSW graph'ının native bellek kullanımı
                     (_plugins/_knn/stats — asıl vektör indeksi maliyeti)

NOT: JVM heap GC'ye bağlı dalgalanır, tek başına gürültülüdür. Vektör
indeksinin gerçek maliyeti knn_graph_memory'dir; faiss/lucene motoru graph'ı
heap dışında tutar.

Kullanım:
    python benchmark/os_memstat.py --out benchmark/dataset/mem_before.json
    ... benchmark ...
    python benchmark/os_memstat.py --out benchmark/dataset/mem_after.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib3
from pathlib import Path

import requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def fetch(session, base: str, path: str, auth, verify: bool) -> tuple[dict | None, str]:
    """(veri, hata_aciklamasi) döner. Hata boşsa istek başarılı."""
    try:
        r = session.get(f"{base}{path}", auth=auth, verify=verify, timeout=30)
        if not r.ok:
            return None, f"HTTP {r.status_code}: {r.text[:200]}"
        return r.json(), ""
    except requests.exceptions.SSLError as exc:
        return None, f"SSL hatasi: {exc}"
    except requests.exceptions.ConnectionError:
        return None, "baglanti kurulamadi (servis kapali ya da yanlis sema/port)"
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=9200)
    p.add_argument("--no-ssl", action="store_true")
    p.add_argument("--user", default=None)
    p.add_argument("--password", default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--label", default="", help="snapshot etiketi (before/after)")
    args = p.parse_args()

    load_dotenv(ROOT / ".env")
    user = args.user or os.getenv("OPENSEARCH_USER") or "admin"
    password = args.password or os.getenv("OPENSEARCH_PASSWORD")
    auth = (user, password) if password else None

    scheme = "http" if args.no_ssl else "https"
    base = f"{scheme}://{args.host}:{args.port}"
    session = requests.Session()

    snap: dict = {"label": args.label, "endpoint": base}

    nodes, err = fetch(session, base, "/_nodes/stats/jvm,indices", auth, verify=False)
    if nodes is None:
        # Bu uc calismiyorsa OpenSearch'e hic ulasilamiyor demektir. Sessizce
        # bos snapshot yazip devam etmek, sonraki raporu yaniltir.
        print(f"HATA: OpenSearch'e ulasilamadi -> {base}", file=sys.stderr)
        print(f"       {err}\n", file=sys.stderr)
        print("  Kontrol et:", file=sys.stderr)
        print(f"    1) Servis ayakta mi?   curl -k {base}", file=sys.stderr)
        alt = base.replace("https://", "http://") if base.startswith("https") else base.replace("http://", "https://")
        print(f"    2) Sema dogru mu?      {alt} dene ({'--no-ssl' if base.startswith('https') else 'ssl acik'})",
              file=sys.stderr)
        print("    3) Kimlik bilgileri?   .env icindeki OPENSEARCH_USER/PASSWORD", file=sys.stderr)
        snap["error"] = err
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(snap, indent=2), encoding="utf-8")
        sys.exit(1)

    if nodes:
        heap = seg = 0
        node_names = []
        for node in nodes.get("nodes", {}).values():
            node_names.append(node.get("name", "?"))
            heap += node.get("jvm", {}).get("mem", {}).get("heap_used_in_bytes", 0)
            segments = node.get("indices", {}).get("segments", {})
            # OpenSearch 2.x'te segment bellek alanları kaldırılmış olabilir.
            seg += segments.get("memory_in_bytes", 0)
        snap["nodes"] = node_names
        snap["jvm_heap_used_bytes"] = heap
        snap["segments_memory_bytes"] = seg

    knn, knn_err = fetch(session, base, "/_plugins/_knn/stats", auth, verify=False)
    if knn_err:
        print(f"  UYARI /_plugins/_knn/stats -> {knn_err}", file=sys.stderr)
    if knn:
        graph_kb = 0
        for node in knn.get("nodes", {}).values():
            # graph_memory_usage KB cinsindendir
            graph_kb += node.get("graph_memory_usage", 0) or 0
        snap["knn_graph_memory_bytes"] = int(graph_kb) * 1024
        snap["knn_stats_available"] = True
    else:
        snap["knn_stats_available"] = False

    # İndeks boyutu (disk) — bellek değil ama karşılaştırma için faydalı
    cat, _ = fetch(session, base, "/_stats/store", auth, verify=False)
    if cat:
        snap["store_size_bytes"] = (
            cat.get("_all", {}).get("total", {}).get("store", {}).get("size_in_bytes", 0)
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(snap, indent=2), encoding="utf-8")

    def mb(key: str) -> str:
        v = snap.get(key)
        # 0 gecerli bir deger (henuz index yuklenmemis); "veri yok"tan farkli.
        if v is None:
            return "olculemedi"
        return f"{v / 1e6:.2f} MB"

    print(f"snapshot [{args.label or 'unlabeled'}] -> {args.out}")
    print(f"  JVM heap        : {mb('jvm_heap_used_bytes')}")
    seg_note = ""
    if snap.get("segments_memory_bytes") == 0:
        seg_note = "   (OpenSearch 2.x bu alani kaldirdi — normal)"
    print(f"  Segment bellegi : {mb('segments_memory_bytes')}{seg_note}")
    knn_note = ""
    if snap.get("knn_graph_memory_bytes") == 0:
        knn_note = "   (henuz kNN index yuklenmemis)"
    print(f"  kNN graph       : {mb('knn_graph_memory_bytes')}{knn_note}")
    print(f"  Disk (store)    : {mb('store_size_bytes')}")


if __name__ == "__main__":
    main()
