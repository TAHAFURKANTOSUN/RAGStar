# pgvector karşılaştırması

Aynı contextual chunk'ları PostgreSQL + pgvector'e yükleyip OpenSearch ile
karşılaştıran benchmark.

## Neden ayrı bir ingest değil de kopyalama

`pg_ingest.py` varsayılan olarak chunk'ları **doğrudan OpenSearch index'inden**
çeker (`--from-opensearch`). Sebebi: benchmark'ın ölçmesi gereken şey arama
motoru farkı, veri farkı değil. Kopyalayınca iki sistemde birebir aynı metin ve
aynı vektörler bulunur — Gemini'ye tekrar gidilmez, embedding yeniden
hesaplanmaz.

Bağımsız kurmak isterseniz `--from-dataset` var, ama o zaman ön söz cache'i ve
embedding modeli aynı olmalı, yoksa karşılaştırma anlamını yitirir.

## Kurulum

```bash
pip install -r pgvector/requirements.txt      # proje kökünden

cd pgvector
docker compose up -d
docker compose logs -f postgres               # "ready to accept connections"
```

Bağlantı ayarları compose varsayılanlarıyla eşleşir (`localhost:5432`,
`rag/ragpass`, db `ragbench`). Değiştirmek isterseniz ana `.env` dosyasına
`PG_HOST` / `PG_PORT` / `PG_DB` / `PG_USER` / `PG_PASSWORD` ya da tek satırda
`PG_DSN` ekleyin.

## Çalıştırma

```bash
# 1. OpenSearch tarafı hazır olmalı
cd .. && python ingest.py && cd pgvector

# 2. pgvector'e kopyala
python pg_ingest.py

# 3. Karşılaştır
python benchmark.py -n 200 -k 1 5 10
```

Rapor `pgvector/results/benchmark-<zaman>.md` altına yazılır.

Tek tek arama denemek için:

```bash
python pg_search.py "Who assassinated Lincoln?"
python pg_search.py "beetle defense" --mode vector -k 5
```

## Şema

```sql
CREATE TABLE chunks (
    chunk_id        text PRIMARY KEY,
    ...
    contextual_text text NOT NULL,
    embedding       vector(384) NOT NULL,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', contextual_text)) STORED
);
CREATE INDEX ... USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=256);
CREATE INDEX ... USING GIN (tsv);
```

`tsv` **generated column**'dur — `contextual_text` değişince otomatik güncellenir,
elle senkron tutmak gerekmez. Index'ler veri yüklendikten **sonra** kurulur;
HNSW'yi boş tabloya kurup satır satır doldurmak belirgin şekilde yavaştır.

OpenSearch tarafıyla eşdeğerlik:

| | OpenSearch | pgvector |
|---|---|---|
| Vektör | HNSW, cosine, Lucene | HNSW, `vector_cosine_ops` |
| Leksik | BM25, `english` analyzer | `ts_rank_cd` + GIN, `english` config |
| Hibrit | `hybrid` query + search pipeline | SQL içinde RRF |
| ANN ayarı | `ef_search=100`, m=16, ef_construction=256 | aynı |

## Dürüst olunması gereken nokta: ts_rank BM25 değil

Postgres'in tam metin sıralaması BM25 uygulamaz — doküman uzunluğu
normalizasyonu ve terim frekansı doygunluğu farklı çalışır. Yani `pgvector /
ts_rank` satırı ile `opensearch / bm25` satırı **birebir kıyaslanamaz**; leksik
tarafta OpenSearch'ün avantajlı çıkması beklenir.

Bu yüzden hibritte ham skorları toplamak yerine **sıra tabanlı RRF**
kullanıyorum: `score = w_v/(K+rank_v) + w_f/(K+rank_f)`. Sıralar her iki sistemde
de aynı anlama gelir, skorlar gelmez. Gerçekten BM25 istiyorsanız ParadeDB
(`pg_search`) eklentisine bakın.

Vektör tarafında ise ikisi de aynı vektörler üzerinde aynı HNSW parametreleriyle
çalıştığı için `knn` ve `vector` satırları doğrudan karşılaştırılabilir —
farklar ANN implementasyonundan ve recall'dan gelir.

## Benchmark neyi nasıl ölçüyor

- **Sorgu embedding'leri bir kez** hesaplanır, iki sisteme de aynı vektör gider
- **Gecikme** yalnızca arama çağrısını kapsar (embedding hariç)
- Her koşudan önce **ısınma turu** yapılır
- `--repeat N` ile her sorgu N kez çalışır, p50/p95 daha kararlı olur
- **Ortüşme**: iki sistemin hybrid top-k sonuçlarının Jaccard benzerliği —
  1.0'a yakınsa aynı şeyi buluyorlar, düşükse sıralama davranışları ayrışıyor

Kalite metriği proxy'dir (dataset'te gold passage etiketi yok; bir sonuç cevap
metnini içeriyorsa ilgili sayılıyor). **Mutlak değerlere değil satırlar arası
farka bakın.**

Gecikmeler tek istemciden seri ölçülür; eşzamanlı yük altındaki davranışı
yansıtmaz. Onu istiyorsanız `--repeat` yerine gerçek bir yük aracı gerekir.

## Dosyalar

| Dosya | İş |
|---|---|
| `docker-compose.yml` | pgvector'lü PostgreSQL 17 |
| `pg_config.py` | Bağlantı + index ayarları (ortak ayarlar ana config'ten) |
| `pg_store.py` | Şema, COPY ile toplu yükleme, vector/fts/hybrid arama |
| `pg_ingest.py` | OpenSearch'ten kopyalar veya dataset'ten kurar |
| `pg_search.py` | Arama CLI |
| `benchmark.py` | Karşılaştırma + markdown rapor |
