import uuid
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredPDFLoader

# PDF'i elementlerine (tablo, metin, görsel) ayırarak yükle
loader = UnstructuredPDFLoader("dokumaniniz.pdf", strategy="hi_res", extract_images_to_elements=True)
docs = loader.load()

# Elementleri kategorize etme
text_elements = []
table_elements = []

for doc in docs:
    # Unstructured kütüphanesinin döndürdüğü element tipine göre ayırma
    if "Table" in doc.metadata.get("filetype", "") or doc.metadata.get("category") == "Table":
        table_elements.append(doc.page_content)
    else:
        text_elements.append(doc.page_content)
