"""
VectorDBBench Custom Dataset Builder — DeneyselRAG
==================================================

PDF'lerden chunk çıkarır, embed eder ve VectorDBBench'in "Custom Dataset"
formatında parquet dosyaları üretir:

    <out>/train.parquet      id:int64, emb:list<float>
    <out>/test.parquet       id:int64, emb:list<float>   (sorgu vektörleri)
    <out>/neighbors.parquet  id:int64, neighbors_id:list<int64>  (exact kNN)
    <out>/chunks.jsonl       id -> metin/kaynak/sayfa  (izlenebilirlik için)
    <out>/dataset_meta.json  size/dim/metric özeti

Embedding backend'i seçilebilir:
  --backend api    OpenAI-uyumlu /embeddings endpoint (Cordatus vb.)
  --backend local  lokal sentence-transformers (app.py ile aynı: BAAI/bge-m3)

Kullanım (API):
    set EMBED_BASE_URL=https://<cordatus-endpoint>/v1
    set EMBED_API_KEY=sk-...
    python benchmark/build_dataset.py --backend api --embed-model bge-m3

Kullanım (lokal):
    python benchmark/build_dataset.py --backend local
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_dataset")

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# 1. PDF -> chunk
# --------------------------------------------------------------------------
def extract_chunks(
    pdf_paths: list[Path],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    """PDF'leri okuyup chunk'lara böler. İçerik hash'iyle tekilleştirir.

    birlesmis.pdf, docs/ klasörünün birleştirilmiş hali olduğundan aynı metin
    iki kez gelebilir; hash tabanlı dedupe bunu temizler.
    """
    import fitz  # PyMuPDF
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.schema import Document

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    seen: set[str] = set()
    chunks: list[dict] = []

    for pdf in pdf_paths:
        try:
            doc = fitz.open(pdf)
        except Exception as exc:  # noqa: BLE001
            log.warning("Açılamadı, atlanıyor: %s (%s)", pdf.name, exc)
            continue

        page_docs = []
        for page_no, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if not text:
                continue
            page_docs.append(
                Document(text=text, metadata={"source": pdf.name, "page": page_no})
            )
        doc.close()

        if not page_docs:
            log.warning("Metin çıkmadı: %s", pdf.name)
            continue

        nodes = splitter.get_nodes_from_documents(page_docs)
        kept = 0
        for node in nodes:
            text = re.sub(r"\s+", " ", node.get_content()).strip()
            # Çok kısa parçalar gürültü; embed etmeye değmez.
            if len(text) < 80:
                continue
            digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            chunks.append(
                {
                    "text": text,
                    "source": node.metadata.get("source", pdf.name),
                    "page": node.metadata.get("page", -1),
                }
            )
            kept += 1
        log.info("%-42s %4d chunk (%d tekil)", pdf.name, len(nodes), kept)

    return chunks


# --------------------------------------------------------------------------
# 2. Embedding backend'leri
# --------------------------------------------------------------------------
class ApiEmbedder:
    """OpenAI-uyumlu POST {base_url}/embeddings backend'i.

    Cordatus, vLLM, TEI, Ollama (/v1), LocalAI ve LM Studio bu şemayı konuşur.
    """

    def __init__(self, base_url: str, api_key: str | None, model: str, batch_size: int):
        import requests

        self.session = requests.Session()
        self.url = base_url.rstrip("/") + "/embeddings"
        self.model = model
        self.batch_size = batch_size
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def _post(self, batch: list[str], attempt: int = 0) -> list[list[float]]:
        payload = {"model": self.model, "input": batch}
        try:
            resp = self.session.post(
                self.url, json=payload, headers=self.headers, timeout=180
            )
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            if attempt >= 4:
                raise
            wait = 2**attempt
            log.warning("Embedding isteği başarısız (%s). %ss sonra tekrar.", exc, wait)
            time.sleep(wait)
            return self._post(batch, attempt + 1)

        data = resp.json()["data"]
        # Endpoint sırayı bozabilir; 'index' alanı varsa ona göre sırala.
        data.sort(key=lambda d: d.get("index", 0))
        return [d["embedding"] for d in data]

    def encode(self, texts: list[str]) -> np.ndarray:
        out: list[list[float]] = []
        total = len(texts)
        for i in range(0, total, self.batch_size):
            batch = texts[i : i + self.batch_size]
            out.extend(self._post(batch))
            log.info("  embed %d/%d", min(i + self.batch_size, total), total)
        return np.asarray(out, dtype=np.float32)


class LocalEmbedder:
    """app.py'deki SafeHuggingFaceEmbedding ile aynı davranış (NaN/Inf temizliği)."""

    def __init__(self, model: str, device: str, batch_size: int):
        import torch
        from sentence_transformers import SentenceTransformer

        if device == "cpu":
            # app.py'deki OpenMP/MKL kaynaklı NaN sorununa karşı aynı önlem.
            torch.set_num_threads(1)

        self.model = SentenceTransformer(model, device=device)
        self.batch_size = batch_size

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        vecs = np.asarray(vecs, dtype=np.float32)
        bad = ~np.isfinite(vecs)
        if bad.any():
            log.warning("%d adet NaN/Inf değer 0.0 ile değiştirildi.", int(bad.sum()))
            vecs[bad] = 0.0
        return vecs


def build_embedder(args) -> object:
    if args.backend == "api":
        base_url = args.embed_base_url or os.getenv("EMBED_BASE_URL")
        if not base_url:
            sys.exit(
                "HATA: --backend api için EMBED_BASE_URL ortam değişkeni ya da "
                "--embed-base-url gerekli (örn. https://host/v1)."
            )
        api_key = args.embed_api_key or os.getenv("EMBED_API_KEY")
        log.info("Embedding backend: API -> %s (model=%s)", base_url, args.embed_model)
        return ApiEmbedder(base_url, api_key, args.embed_model, args.batch_size)

    log.info(
        "Embedding backend: lokal %s (device=%s)", args.local_model, args.device
    )
    return LocalEmbedder(args.local_model, args.device, args.batch_size)


# --------------------------------------------------------------------------
# 3. Ground truth (exact kNN, brute force)
# --------------------------------------------------------------------------
def exact_knn(
    train: np.ndarray,
    queries: np.ndarray,
    k: int,
    metric: str,
    block: int = 512,
) -> np.ndarray:
    """Kesin (exact) en yakın k komşuyu hesaplar — ANN değil, referans doğru cevap.

    Bellek patlamasın diye sorguları bloklar halinde işler.
    """
    n_train = train.shape[0]
    k = min(k, n_train)
    out = np.empty((queries.shape[0], k), dtype=np.int64)

    if metric in ("cosine", "ip"):
        # Vektörler normalize edilmişse iç çarpım = kosinüs benzerliği.
        # Büyük skor = yakın.
        for i in range(0, queries.shape[0], block):
            sims = queries[i : i + block] @ train.T
            idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
            rows = np.arange(idx.shape[0])[:, None]
            order = np.argsort(-sims[rows, idx], axis=1)
            out[i : i + block] = idx[rows, order]
    else:  # l2 — küçük mesafe = yakın
        train_sq = (train**2).sum(axis=1)
        for i in range(0, queries.shape[0], block):
            q = queries[i : i + block]
            d = train_sq[None, :] - 2.0 * (q @ train.T) + (q**2).sum(axis=1)[:, None]
            idx = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
            rows = np.arange(idx.shape[0])[:, None]
            order = np.argsort(d[rows, idx], axis=1)
            out[i : i + block] = idx[rows, order]

    return out


# --------------------------------------------------------------------------
# 4. Ana akış
# --------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="VectorDBBench custom dataset üretici (DeneyselRAG)"
    )
    p.add_argument(
        "--pdf-dir",
        type=Path,
        default=ROOT / "docs",
        help="PDF klasörü (varsayılan: ./docs)",
    )
    p.add_argument(
        "--extra-pdf",
        type=Path,
        nargs="*",
        default=[],
        help="Ek PDF dosyaları. birlesmis.pdf docs/ ile aynı içerik olduğu için "
        "varsayılan olarak DAHİL EDİLMEZ; dedupe zaten temizler.",
    )
    p.add_argument("--out", type=Path, default=ROOT / "benchmark" / "dataset")
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--chunk-overlap", type=int, default=20)
    p.add_argument(
        "--num-queries",
        type=int,
        default=100,
        help="Sorgu (test) seti büyüklüğü. Train'den ayrılır.",
    )
    p.add_argument(
        "--gt-k",
        type=int,
        default=100,
        help="Ground truth'ta saklanacak komşu sayısı. Benchmark --k bundan büyük olamaz.",
    )
    p.add_argument(
        "--metric",
        choices=["cosine", "l2", "ip"],
        default="cosine",
        help="bge-m3 kosinüs için eğitildi; varsayılan cosine.",
    )
    p.add_argument(
        "--no-normalize",
        action="store_true",
        help="Vektörleri L2-normalize etme (cosine/ip için normalize önerilir).",
    )
    p.add_argument("--backend", choices=["api", "local"], default="api")
    p.add_argument("--embed-base-url", default=None)
    p.add_argument("--embed-api-key", default=None)
    p.add_argument("--embed-model", default="bge-m3")
    p.add_argument("--local-model", default="BAAI/bge-m3")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # ---- 1. Chunk ----
    pdfs = sorted(args.pdf_dir.glob("*.pdf")) if args.pdf_dir.exists() else []
    pdfs += [Path(x) for x in args.extra_pdf]
    if not pdfs:
        sys.exit(f"HATA: {args.pdf_dir} içinde PDF bulunamadı.")

    log.info("=== 1/4  Chunk çıkarımı (size=%d, overlap=%d) ===", args.chunk_size, args.chunk_overlap)
    chunks = extract_chunks(pdfs, args.chunk_size, args.chunk_overlap)
    if len(chunks) < args.num_queries * 3:
        log.warning(
            "Sadece %d chunk çıktı. num_queries=%d ile train seti çok küçük kalır; "
            "--chunk-size değerini düşürmeyi düşün.",
            len(chunks),
            args.num_queries,
        )
    log.info("Toplam tekil chunk: %d", len(chunks))

    # ---- 2. Embed ----
    log.info("=== 2/4  Embedding ===")
    embedder = build_embedder(args)
    t0 = time.time()
    vectors = embedder.encode([c["text"] for c in chunks])
    log.info(
        "Embedding bitti: %s, %.1f sn (%.1f chunk/sn)",
        vectors.shape,
        time.time() - t0,
        len(chunks) / max(time.time() - t0, 1e-6),
    )

    dim = int(vectors.shape[1])
    if not args.no_normalize and args.metric in ("cosine", "ip"):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms
        log.info("Vektörler L2-normalize edildi (metric=%s).", args.metric)

    # ---- 3. Train / test ayrımı + ground truth ----
    log.info("=== 3/4  Train/test ayrımı ve exact ground truth ===")
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(chunks))
    n_q = min(args.num_queries, max(1, len(chunks) // 5))
    q_idx, tr_idx = perm[:n_q], perm[n_q:]

    train_vecs = np.ascontiguousarray(vectors[tr_idx])
    query_vecs = np.ascontiguousarray(vectors[q_idx])
    log.info("train=%d  test=%d  dim=%d", len(tr_idx), len(q_idx), dim)

    gt_k = min(args.gt_k, len(tr_idx))
    if gt_k < args.gt_k:
        log.warning(
            "gt_k %d -> %d düşürüldü (train seti daha küçük). "
            "Benchmark'ı --k %d veya altıyla çalıştır.",
            args.gt_k,
            gt_k,
            gt_k,
        )
    gt = exact_knn(train_vecs, query_vecs, gt_k, args.metric)

    # ---- 4. Parquet yazımı ----
    log.info("=== 4/4  Parquet yazımı -> %s ===", args.out)
    import pyarrow as pa
    import pyarrow.parquet as pq

    def write_vectors(path: Path, vecs: np.ndarray) -> None:
        table = pa.table(
            {
                "id": pa.array(np.arange(len(vecs), dtype=np.int64)),
                "emb": pa.array(vecs.tolist(), type=pa.list_(pa.float32())),
            }
        )
        pq.write_table(table, path)

    write_vectors(args.out / "train.parquet", train_vecs)
    write_vectors(args.out / "test.parquet", query_vecs)

    pq.write_table(
        pa.table(
            {
                "id": pa.array(np.arange(len(gt), dtype=np.int64)),
                "neighbors_id": pa.array(gt.tolist(), type=pa.list_(pa.int64())),
            }
        ),
        args.out / "neighbors.parquet",
    )

    # İzlenebilirlik: hangi id hangi metin?
    with (args.out / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for new_id, old in enumerate(tr_idx):
            rec = dict(chunks[old])
            rec["id"] = new_id
            rec["split"] = "train"
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        for new_id, old in enumerate(q_idx):
            rec = dict(chunks[old])
            rec["id"] = new_id
            rec["split"] = "test"
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    meta = {
        "size": int(len(tr_idx)),
        "dim": dim,
        "num_queries": int(len(q_idx)),
        "gt_k": int(gt_k),
        "metric_type": {"cosine": "Cosine", "l2": "L2", "ip": "IP"}[args.metric],
        "normalized": not args.no_normalize and args.metric in ("cosine", "ip"),
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "embed_backend": args.backend,
        "embed_model": args.embed_model if args.backend == "api" else args.local_model,
        "source_pdfs": [p.name for p in pdfs],
    }
    (args.out / "dataset_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log.info("")
    log.info("BİTTİ.  %s", args.out)
    log.info("  size=%d  dim=%d  metric=%s  gt_k=%d", meta["size"], dim, meta["metric_type"], gt_k)
    log.info("")
    log.info("Sıradaki adım:  python benchmark/install_dataset.py")


if __name__ == "__main__":
    main()
