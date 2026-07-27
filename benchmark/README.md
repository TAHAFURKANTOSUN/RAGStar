# VectorDBBench ile DeneyselRAG OpenSearch Benchmark'ı

Bu klasör, [zilliztech/VectorDBBench](https://github.com/zilliztech/VectorDBBench)'i
projenin kendi PDF korpusu üzerinde, lokal OpenSearch'e karşı çalıştırmak için gereken
her şeyi içerir.

---

## Önce şunu netleştirelim: bu ne ölçer, ne ölçmez

VectorDBBench **vektör veritabanının** performansını ölçer:

| Ölçer | Ölçmez |
| --- | --- |
| Recall@k (ANN indeksinin kesin cevaba yakınlığı) | Cevabın doğru olup olmadığı |
| QPS (eşzamanlı sorgu verimi) | Reranker'ın katkısı |
| p95 / p99 latency | Chunk boyutunun cevap kalitesine etkisi |
| Index build ve optimize süresi | LLM (Llama 3.1 / Cordatus) kalitesi |
| Yükleme (insert) verimi | Hallucination oranı |

`app.py` pipeline'ının **retrieval katmanının altındaki** OpenSearch'ü test eder.
Reranker ve LLM bu ölçüme hiç girmez. RAG cevap kalitesi için ayrı bir eval
harness'ı (RAGAS benzeri) gerekir — bu ayrı bir iş.

---

## Korpus gerçeği — bunu bilerek başla

`docs/` klasöründeki 14 PDF ve `birlesmis.pdf` **aynı içerik** (birlesmis.pdf,
docs'un birleştirilmiş hali; 144 sayfa = docs toplamı). Tekil içerik:

```
~578.000 karakter  ≈  145.000 token
```

Bu, chunk boyutuna göre şu kadar vektör eder:

| chunk_size | yaklaşık vektör sayısı |
| --- | --- |
| 512 (app.py ayarı) | ~310 |
| 256 | ~620 |
| **128 (bu benchmark'ın varsayılanı)** | **~1.200** |

**1.200 vektör, ANN benchmark'ı için küçüktür.** Şu anlama gelir:

- `k=100` ile recall ölçmek anlamsız — korpusun %8'ini getiriyor olursun.
  Bu yüzden varsayılan `k=10`.
- HNSW bu ölçekte neredeyse her zaman recall≈1.0 verir; parametre taraması
  (`-Sweep`) düz bir çizgi çıkarabilir. Ayrım görmek için ef_search'ü
  agresif biçimde düşürmek gerekir (32 ve altı).
- QPS ve latency sayıları **anlamlıdır** — bunlar korpus boyutundan görece
  bağımsız, senin donanımını ve OpenSearch ayarlarını ölçer.
- Index build süresi de anlamlı ama saniyeler mertebesinde kalacaktır.

Yani bu kurulum **pipeline doğrulaması + latency/QPS ölçümü** için sağlam,
**recall karşılaştırması** için zayıf. Recall eğrisi istiyorsan korpusu
100K+ pasaja çıkarman gerekir (bkz. en alttaki "Ölçeği büyütmek").

---

## Kurulum

### 0. İki ayrı sanal ortam kur

VectorDBBench, `llama_index` ile çakışan büyük bir bağımlılık ağacı çeker
(streamlit, pymilvus, polars, s3fs...). Projenin `env/` ortamına kurma.

Ayrıca **VectorDBBench Python >= 3.11 ister.** Senin `env/` ortamın Python
3.14.6 — sürüm olarak uygun ama 3.14 çok yeni, bazı bağımlılıkların wheel'i
olmayabilir. Sorun çıkarsa 3.11 veya 3.12 ile ayrı ortam aç.

```powershell
# RAG ortamı (zaten var) — build_dataset.py burada çalışır
.\env\Scripts\activate
pip install pyarrow          # muhtemelen tek eksik

# Benchmark ortamı (yeni) — geri kalan her şey burada
python -m venv bench-env
.\bench-env\Scripts\activate
pip install "vectordb-bench[opensearch]"
```

### 1. Dataset üret (RAG ortamında)

`build_dataset.py` PDF'leri chunk'lar, embed eder, train/test ayırır ve
**kesin (exact) kNN ile ground truth** hesaplar.

Cordatus veya OpenAI-uyumlu bir embedding endpoint'i ile:

```powershell
.\env\Scripts\activate
$env:EMBED_BASE_URL = "https://<cordatus-host>/v1"
$env:EMBED_API_KEY  = "<key>"
python benchmark\build_dataset.py --backend api --embed-model bge-m3
```

Lokal bge-m3 ile (app.py ile birebir aynı model, CPU'da birkaç dakika):

```powershell
python benchmark\build_dataset.py --backend local
```

Faydalı bayraklar:

```
--chunk-size 128        varsayılan; 64 yaparsan ~2.400 vektör
--num-queries 100       sorgu seti büyüklüğü
--gt-k 100              ground truth'ta saklanan komşu sayısı
--metric cosine         bge-m3 kosinüs için eğitildi
--device cuda           lokal backend'de GPU
```

Çıktı `benchmark/dataset/` altına düşer:
`train.parquet`, `test.parquet`, `neighbors.parquet`, `chunks.jsonl`,
`dataset_meta.json`.

> `chunks.jsonl` her vektör id'sinin hangi metin/kaynak/sayfadan geldiğini
> tutar. Recall düşük çıkarsa hangi chunk'ların kaçırıldığına bakabilirsin.

### 2. VectorDBBench'i yamala (bench ortamında)

Upstream'de lokal OpenSearch'e bağlanmayı engelleyen iki sorun var:

**Yama 1 — `metric_type_name` geçirilmiyor.**
`oss_opensearch/cli.py`, `OSSOpenSearchIndexConfig`'i kurarken bu alanı
doldurmuyor (`aws_opensearch/cli.py` dolduruyor). Sonuç: `parse_metric()`
içinde `None.upper()` çağrılıyor ve koşu `AttributeError` ile düşüyor.
Bunu sandbox'ta doğruladım.

**Yama 2 — SSL sadece port 443'te açılıyor.**
`config.py::to_dict()` içinde `use_ssl = self.port == 443` ve
`verify_certs = use_ssl` sabit. OpenSearch'ün varsayılan dev kurulumu
9200'de HTTPS konuşur ve sertifikası self-signed'dır — bu haliyle ne
bağlanılabilir ne de sertifika doğrulaması atlanabilir. Yama, davranışı
`OSS_OPENSEARCH_USE_SSL` / `OSS_OPENSEARCH_VERIFY_CERTS` ortam
değişkenleriyle kontrol edilebilir yapar; değişken yoksa eski davranış aynen
kalır.

```powershell
.\bench-env\Scripts\activate
python benchmark\patch_vdbb.py            # uygula
python benchmark\patch_vdbb.py --check    # durumu gör
python benchmark\patch_vdbb.py --revert   # .bak'tan geri al
```

### 3. Dataset'i yerine kur (bench ortamında)

VectorDBBench dosyaları `{DATASET_LOCAL_DIR}/{name}/{dir}/` altından okur.
Varsayılan `DATASET_LOCAL_DIR` = `/tmp/vectordb_bench/dataset` — Windows'ta
bunu değiştirmek isteyebilirsin:

```powershell
$env:DATASET_LOCAL_DIR = "C:\vdbb\dataset"
python benchmark\install_dataset.py
```

Bu ayrıca `vectordb_bench/custom/custom_case.json` dosyasına case tanımını
yazar; Streamlit arayüzünden koşmak istersen orada görünür.

### 4. Çalıştır

```powershell
.\benchmark\run_opensearch.ps1 -DryRun     # önce yapılandırmayı gör
.\benchmark\run_opensearch.ps1             # tek koşu
.\benchmark\run_opensearch.ps1 -Sweep      # M x ef_search taraması (12 koşu)
```

Linux/macOS: `./benchmark/run_opensearch.sh` (`DRY_RUN=1`, `SWEEP=1`).

Script `.env` dosyasından `OPENSEARCH_USER` / `OPENSEARCH_PASSWORD` okur,
`dataset_meta.json`'dan size/dim/metric alır ve `k > gt_k` ise baştan durur.

Parametreler:

```
-K 10                    aranacak komşu sayısı (gt_k'yı aşamaz)
-M 16 -EfConstruction 256 -EfSearch 100
-Engine faiss|lucene
-NumConcurrency "1,4,8"  eşzamanlılık seviyeleri
-Port 9200 -NoSsl        güvenlik eklentisi kapalıysa
```

### 5. Sonuçlar

JSON olarak `vectordb_bench/results/` altına yazılır. Görsel karşılaştırma:

```powershell
init_bench      # Streamlit arayüzünü açar
```

---

## Sorun giderme

| Belirti | Sebep / çözüm |
| --- | --- |
| `AttributeError: 'NoneType' object has no attribute 'upper'` | Yama 1 uygulanmamış. `python benchmark\patch_vdbb.py` |
| `ConnectionError` / `Connection refused` 9200'de | SSL uyuşmazlığı. Güvenlik eklentisi açıksa `OSS_OPENSEARCH_USE_SSL=true`, kapalıysa `-NoSsl` |
| `certificate verify failed` | `OSS_OPENSEARCH_VERIFY_CERTS=false` (script zaten ayarlıyor) |
| `401 Unauthorized` | `.env` içindeki `OPENSEARCH_USER` / `OPENSEARCH_PASSWORD` yanlış |
| `Could not find a version that satisfies vectordb-bench` | Python < 3.11. Ayrı venv aç |
| `KeyError` / gt boyut hatası | `--k` değeri `gt_k`'dan büyük |
| Recall hep 1.0 | Korpus küçük — beklenen. `-EfSearch 16` gibi agresif değerlerle ayrım aranabilir |
| Embedding isteği timeout | `--batch-size 8` ile küçült |

---

## Ölçeği büyütmek

Recall eğrisi anlamlı olsun istiyorsan korpusu büyütmen gerekir. Sıra ile
en az müdahaleden en fazlasına:

1. **`--chunk-size 64`** → ~2.400 vektör. Ucuz ama hâlâ küçük.
2. **Aynı alandan açık korpus ekle.** `docs/` alerji/immünoloji ağırlıklı;
   PubMed abstract'ları veya BEIR'in `nfcorpus`/`scifact` setleri doğal bir
   genişletme olur. `build_dataset.py`'ye bir loader eklemek yeterli —
   geri kalan akış aynı çalışır.
3. **GPU'da embed et.** bge-m3 CPU'da ~5-15 chunk/sn; 100K pasaj günler
   sürer. `--device cuda` ile 50-100x hızlanır.
4. **Hazır ANN dataset'i kullan.** Sadece OpenSearch ayarlarını
   karşılaştırmak istiyorsan kendi verinden vazgeçip
   `--case-type Performance768D1M` (Cohere 1M) yeterli — VectorDBBench
   dataset'i kendisi indirir, embedding üretmen gerekmez.

---

## Dosyalar

| Dosya | Ne yapar | Hangi ortamda |
| --- | --- | --- |
| `build_dataset.py` | PDF → chunk → embed → parquet + exact ground truth | RAG (`env`) |
| `patch_vdbb.py` | VectorDBBench'in iki OpenSearch hatasını yamalar | bench |
| `install_dataset.py` | Parquet'leri yerine kopyalar, `custom_case.json` yazar | bench |
| `run_opensearch.ps1` / `.sh` | Benchmark'ı çalıştırır | bench |
