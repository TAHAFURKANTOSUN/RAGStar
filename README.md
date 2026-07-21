# 🦙 RAGStar

Yüksek doğruluklu (**High Precision**) belge sorgulama ve yanıt üretme amacıyla geliştirilmiş, **tamamen yerel (Local) ve gizlilik odaklı** bir **Retrieval-Augmented Generation (RAG)** sistemidir.

Bu proje; gelişmiş metin işleme, hibrit arama (**Hybrid Search**), yeniden sıralama (**Reranking**) ve yerel büyük dil modellerini (**Local LLMs**) bir araya getirerek, **hallucination (illüzyon)** oranını minimuma indiren kurumsal düzeyde bir soru-cevap altyapısı sunar.

---

# 🏗️ Sistem Mimarisi

```text
                 ┌────────────────────┐
                 │     PDF Belgesi    │
                 └─────────┬──────────┘
                           │
                           ▼
              PyMuPDF (Metin Ayrıştırma)
                           │
                           ▼
                Metin Parçalama (Chunking)
                           │
                           ▼
       Embedding (BAAI/bge-m3 - 1024 Dimensions)
                           │
                           ▼
        OpenSearch Vector Database (Hybrid Search)
                           │
               İlk Arama (Top 15 Belgeler)
                           │
                           ▼
      BAAI/bge-reranker-large (Cross Encoder)
                           │
             Yeniden Sıralama (Top 3 Sonuç)
                           │
                           ▼
            Llama 3.1 8B (Ollama - Local LLM)
                           │
                           ▼
              Doğru ve Güvenilir Yanıt
```

---

# ✨ Özellikler

* 📄 **PDF Doküman Analizi**

  * PyMuPDF ile hızlı ve kaliteli metin çıkarımı
  * Sayfa yapısını mümkün olduğunca koruyan ayrıştırma

* 🔍 **Hybrid Search**

  * BM25 kelime tabanlı arama
  * Dense Vector Search
  * Daha yüksek doğruluk için hibrit sorgulama

* 🧠 **Semantic Embedding**

  * `BAAI/bge-m3`
  * Türkçe dahil çok dilli destek
  * 1024 boyutlu embedding

* 🎯 **Cross-Encoder Reranking**

  * `BAAI/bge-reranker-large`
  * İlk bulunan belgeleri yeniden puanlayarak en alakalı bağlamı seçer.

* 🤖 **Yerel LLM**

  * Llama 3.1 8B
  * Ollama üzerinden tamamen lokal çalışır.
  * Veriler hiçbir bulut servisine gönderilmez.

* 🔒 **Gizlilik Odaklı**

  * Tamamen offline çalışabilir.
  * Kurumsal veri güvenliği için uygundur.

---

# 🛠️ Kullanılan Teknolojiler

| Teknoloji                   | Açıklama                 |
| --------------------------- | ------------------------ |
| **PyMuPDF (fitz)**          | PDF metin ayrıştırma     |
| **BAAI/bge-m3**             | Embedding modeli         |
| **OpenSearch**              | Vektör veritabanı + BM25 |
| **BAAI/bge-reranker-large** | Cross Encoder Reranker   |
| **Llama 3.1 8B**            | Yerel Büyük Dil Modeli   |
| **Ollama**                  | Yerel LLM çalıştırma     |
| **LlamaIndex**              | RAG Pipeline yönetimi    |

---

# 🚀 Kurulum

## 1. Depoyu Klonlayın

```bash
git clone https://github.com/KULLANICI_ADI/REPO_ADI.git

cd REPO_ADI
```

---

## 2. Sanal Ortam Oluşturun

### Windows

```bash
python -m venv env

env\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv env

source env/bin/activate
```

---

## 3. Bağımlılıkları Kurun

```bash
pip install -r requirements.txt
```

---

## 4. Ollama Kurulumu

Ollama'nın sisteminizde kurulu olduğundan emin olun.

Ardından Llama 3.1 modelini indirin:

```bash
ollama run llama3.1
```

---

## 5. Uygulamayı Çalıştırın

Sorgulamak istediğiniz PDF dosyasını proje dizinine ekleyin.

Ardından:

```bash
python app.py
```

---

# 📂 Proje Akışı

```text
PDF
 │
 ▼
PyMuPDF
 │
 ▼
Chunking
 │
 ▼
Embedding (bge-m3)
 │
 ▼
OpenSearch
 │
 ▼
Top-15 Retrieval
 │
 ▼
BGE-Reranker
 │
 ▼
Top-3 Context
 │
 ▼
Llama 3.1
 │
 ▼
Final Answer
```

---

# 📄 Veri Seti ve Gizlilik

Bu projede kullanılan test PDF dosyaları telif hakları ve veri gizliliği nedeniyle depoya dahil edilmemiştir.

Kendi testlerinizi gerçekleştirmek için:

1. Analiz etmek istediğiniz akademik makaleyi veya PDF dokümanını indirin.
2. Dosyayı proje kök dizinine ekleyin.
3. `app.py` içerisindeki PDF dosya yolunu güncelleyin.
4. Uygulamayı çalıştırın.

Örnek test dokümanı:

```
PMC7105930.pdf
```

---

# 🎯 Kullanım Alanları

* Akademik makale analizi
* Kurumsal doküman sorgulama
* Teknik dokümantasyon
* Hukuki belgeler
* Şirket içi bilgi tabanı
* Ar-Ge dokümanları
* Yerel (Offline) yapay zekâ uygulamaları

---

# 📈 Performans Yaklaşımı

Sistem doğruluğunu artırmak amacıyla iki aşamalı arama stratejisi kullanır:

1. **Hybrid Retrieval**

   * BM25
   * Dense Vector Search

2. **Cross-Encoder Reranking**

   * İlk bulunan belgeleri yeniden değerlendirir.
   * En alakalı bağlamları seçerek LLM'e iletir.
   * Hallucination oranını azaltır.

---

# 🔒 Gizlilik

Bu proje tamamen yerel olarak çalışacak şekilde tasarlanmıştır.

* ✅ Veriler bilgisayarı terk etmez.
* ✅ Harici API kullanılmaz.
* ✅ Bulut servisine ihtiyaç duymaz.
* ✅ Hassas kurumsal belgeler için uygundur.

---

# 📝 Lisans

Bu proje **MIT License** kapsamında lisanslanmıştır.

Daha fazla bilgi için `LICENSE` dosyasını inceleyebilirsiniz.
