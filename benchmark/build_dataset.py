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

ASİMETRİK MODELLER
------------------
nvidia/nv-embedqa-e5-v5 gibi QA-retrieval modelleri asimetriktir: dokümanlar
`input_type="passage"`, sorgular `input_type="query"` ile embed edilmelidir.
Bu yüzden train/test ayrımı embedding'den ÖNCE yapılır. Simetrik bir model
(bge-m3 gibi) kullanıyorsan --no-input-type ile bu alanı kapat.

Kullanım (NVIDIA NIM — varsayılan):
    set EMBED_BASE_URL=https://integrate.api.nvidia.com/v1
    set EMBED_API_KEY=nvapi-...
    python benchmark/build_dataset.py

Kullanım (lokal NIM konteyneri):
    python benchmark/build_dataset.py --embed-base-url http://localhost:8000/v1

Kullanım (lokal bge-m3, app.py ile aynı):
    python benchmark/build_dataset.py --backend local --no-input-type
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

# Bilinen modellerin sınırları. Bilinmeyen model adı uyarı üretmez.
MODEL_LIMITS = {
    "nvidia/nv-embedqa-e5-v5": {"max_tokens": 512, "dim": 1024, "asymmetric": True},
    "nvidia/nv-embedqa-mistral-7b-v2": {"max_tokens": 512, "dim": 4096, "asymmetric": True},
    "nvidia/llama-3.2-nv-embedqa-1b-v2": {"max_tokens": 8192, "dim": 2048, "asymmetric": True},
    "baai/bge-m3": {"max_tokens": 8192, "dim": 1024, "asymmetric": False},
}


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

    NVIDIA NeMo Retriever NIM, vLLM, TEI, Ollama (/v1), LocalAI ve LM Studio
    bu şemayı konuşur. NVIDIA NIM ek olarak `input_type` ve `truncate`
    alanlarını kabul eder.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        batch_size: int,
        truncate: str | None = "END",
        encoding_format: str = "float",
    ):
        import requests

        self.session = requests.Session()
        self.url = base_url.rstrip("/") + "/embeddings"
        self.model = model
        self.batch_size = batch_size
        self.truncate = truncate
        self.encoding_format = encoding_format
        self.headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def _post(self, batch: list[str], input_type: str | None, attempt: int = 0) -> list[list[float]]:
        payload: dict = {
            "model": self.model,
            "input": batch,
            "encoding_format": self.encoding_format,
        }
        if input_type:
            payload["input_type"] = input_type
        if self.truncate:
            payload["truncate"] = self.truncate

        try:
            resp = self.session.post(
                self.url, json=payload, headers=self.headers, timeout=180
            )
            if resp.status_code >= 400:
                # Sunucunun hata gövdesi teşhis için kritik — yutma.
                detail = f"HTTP {resp.status_code}: {resp.text[:500]}"
                # 4xx istemci hatasıdır (yanlış model adı, eksik input_type,
                # geçersiz key). Tekrar denemek bir şeyi düzeltmez — hemen dur.
                # 429 hariç: o gerçekten geçici.
                if 400 <= resp.status_code < 500 and resp.status_code != 429:
                    raise SystemExit(f"\nEmbedding endpoint isteği reddetti.\n  {detail}\n")
                raise RuntimeError(detail)
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            if attempt >= 4:
                raise
            wait = 2**attempt
            log.warning("Embedding isteği başarısız (%s). %ss sonra tekrar.", exc, wait)
            time.sleep(wait)
            return self._post(batch, input_type, attempt + 1)

        data = resp.json()["data"]
        # Endpoint sırayı bozabilir; 'index' alanı varsa ona göre sırala.
        data.sort(key=lambda d: d.get("index", 0))
        return [d["embedding"] for d in data]

    def encode(self, texts: list[str], input_type: str | None = None) -> np.ndarray:
        out: list[list[float]] = []
        total = len(texts)
        label = input_type or "default"
        for i in range(0, total, self.batch_size):
            batch = texts[i : i + self.batch_size]
            out.extend(self._post(batch, input_type))
            done = min(i + self.batch_size, total)
            if done % (self.batch_size * 10) == 0 or done == total:
                log.info("  [%s] %d/%d", label, done, total)
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

    def encode(self, texts: list[str], input_type: str | None = None) -> np.ndarray:
        # E5 ailesi metin öneki bekler; bge-m3 beklemez.
        if input_type == "query":
            texts = [f"query: {t}" for t in texts]
        elif input_type == "passage":
            texts = [f"passage: {t}" for t in texts]

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
                "HATA: --backend api için EMBED_BASE_URL ortam değişkeni ya da\n"
                "      --embed-base-url gerekli.\n"
                "      NVIDIA barındırmalı : https://integrate.api.nvidia.com/v1\n"
                "      Lokal NIM konteyneri: http://localhost:8000/v1"
            )
        api_key = args.embed_api_key or os.getenv("EMBED_API_KEY")
        if not api_key and "integrate.api.nvidia.com" in base_url:
            sys.exit("HATA: NVIDIA barındırmalı API için EMBED_API_KEY gerekli (nvapi-...).")
        log.info("Embedding backend: API -> %s (model=%s)", base_url, args.embed_model)
        return ApiEmbedder(
            base_url, api_key, args.embed_model, args.batch_size, truncate=args.truncate
        )

    log.info("Embedding backend: lokal %s (device=%s)", args.local_model, args.device)
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
    p.add_argument("--pdf-dir", type=Path, default=ROOT / "docs", help="PDF klasörü")
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
        help="E5 ve bge ailesi kosinüs için eğitildi; varsayılan cosine.",
    )
    p.add_argument(
        "--no-normalize",
        action="store_true",
        help="Vektörleri L2-normalize etme (cosine/ip için normalize önerilir).",
    )

    p.add_argument("--backend", choices=["api", "local"], default="api")
    p.add_argument("--embed-base-url", default=None)
    p.add_argument("--embed-api-key", default=None)
    p.add_argument("--embed-model", default="nvidia/nv-embedqa-e5-v5")
    p.add_argument("--local-model", default="BAAI/bge-m3")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--truncate",
        choices=["END", "START", "NONE"],
        default="END",
        help="Model token sınırını aşan girdiyi nasıl kessin (NVIDIA NIM alanı).",
    )
    p.add_argument(
        "--no-input-type",
        action="store_true",
        help="input_type alanını hiç gönderme. Simetrik modeller (bge-m3) için.",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    model_key = (args.embed_model if args.backend == "api" else args.local_model).lower()
    limits = MODEL_LIMITS.get(model_key)

    # Asimetrik model + input_type kapalı = sessiz kalite kaybı. Uyar.
    if limits and limits["asymmetric"] and args.no_input_type:
        log.warning(
            "%s asimetrik bir modeldir; --no-input-type ile sorgu ve pasajlar "
            "aynı uzayda kodlanır ve retrieval kalitesi düşer.",
            model_key,
        )
    if limits and args.chunk_size > limits["max_tokens"]:
        log.warning(
            "chunk_size=%d, %s modelinin %d token sınırını aşıyor. "
            "Fazlası truncate=%s ile kesilecek.",
            args.chunk_size,
            model_key,
            limits["max_tokens"],
            args.truncate,
        )

    # ---- 1. Chunk ----
    pdfs = sorted(args.pdf_dir.glob("*.pdf")) if args.pdf_dir.exists() else []
    pdfs += [Path(x) for x in args.extra_pdf]
    if not pdfs:
        sys.exit(f"HATA: {args.pdf_dir} içinde PDF bulunamadı.")

    log.info(
        "=== 1/4  Chunk çıkarımı (size=%d, overlap=%d) ===",
        args.chunk_size,
        args.chunk_overlap,
    )
    chunks = extract_chunks(pdfs, args.chunk_size, args.chunk_overlap)
    if not chunks:
        sys.exit("HATA: hiç chunk çıkmadı.")
    log.info("Toplam tekil chunk: %d", len(chunks))

    # ---- 2. Train/test ayrımı (embedding'den ÖNCE — asimetrik model gereği) ----
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(chunks))
    n_q = min(args.num_queries, max(1, len(chunks) // 5))
    q_idx, tr_idx = perm[:n_q], perm[n_q:]
    log.info("Ayrım: train=%d  test=%d", len(tr_idx), len(q_idx))

    # ---- 3. Embedding ----
    log.info("=== 2/4  Embedding ===")
    embedder = build_embedder(args)
    passage_type = None if args.no_input_type else "passage"
    query_type = None if args.no_input_type else "query"

    t0 = time.time()
    train_vecs = embedder.encode([chunks[i]["text"] for i in tr_idx], passage_type)
    query_vecs = embedder.encode([chunks[i]["text"] for i in q_idx], query_type)
    elapsed = time.time() - t0
    log.info(
        "Embedding bitti: train=%s query=%s, %.1f sn (%.1f chunk/sn)",
        train_vecs.shape,
        query_vecs.shape,
        elapsed,
        len(chunks) / max(elapsed, 1e-6),
    )

    if train_vecs.shape[1] != query_vecs.shape[1]:
        sys.exit(
            f"HATA: train ({train_vecs.shape[1]}) ve query ({query_vecs.shape[1]}) "
            "boyutları uyuşmuyor."
        )
    dim = int(train_vecs.shape[1])
    if limits and dim != limits["dim"]:
        log.warning(
            "Beklenen boyut %d, gelen %d. Model adı ile sunucudaki model farklı olabilir.",
            limits["dim"],
            dim,
        )

    if not args.no_normalize and args.metric in ("cosine", "ip"):
        for name, v in (("train", train_vecs), ("query", query_vecs)):
            norms = np.linalg.norm(v, axis=1, keepdims=True)
            if np.allclose(norms, 1.0, atol=1e-3):
                log.info("%s vektörleri zaten normalize.", name)
            norms[norms == 0] = 1.0
            v /= norms
        log.info("Vektörler L2-normalize edildi (metric=%s).", args.metric)

    train_vecs = np.ascontiguousarray(train_vecs)
    query_vecs = np.ascontiguousarray(query_vecs)

    # ---- 4. Ground truth ----
    log.info("=== 3/4  Exact ground truth ===")
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

    # ---- 5. Parquet yazımı ----
    log.info("=== 4/4  Parquet yazımı -> %s ===", args.out)
    import pyarrow as pa
    import pyarrow.parquet as pq

    def write_vectors(path: Path, vecs: np.ndarray) -> None:
        pq.write_table(
            pa.table(
                {
                    "id": pa.array(np.arange(len(vecs), dtype=np.int64)),
                    "emb": pa.array(vecs.tolist(), type=pa.list_(pa.float32())),
                }
            ),
            path,
        )

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
        for split, idx_list in (("train", tr_idx), ("test", q_idx)):
            for new_id, old in enumerate(idx_list):
                rec = dict(chunks[old])
                rec["id"] = new_id
                rec["split"] = split
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
        "input_type_used": not args.no_input_type,
        "truncate": args.truncate,
        "source_pdfs": [p.name for p in pdfs],
    }
    (args.out / "dataset_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log.info("")
    log.info("BİTTİ.  %s", args.out)
    log.info(
        "  size=%d  dim=%d  metric=%s  gt_k=%d  input_type=%s",
        meta["size"],
        dim,
        meta["metric_type"],
        gt_k,
        "passage/query" if not args.no_input_type else "yok",
    )
    log.info("")
    log.info("Sıradaki adım:  python benchmark/install_dataset.py")


if __name__ == "__main__":
    main()
