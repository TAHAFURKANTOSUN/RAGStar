import gemini # Veya Anthropic SDK'sı
from datasets import load_dataset

ds = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")
# 1. Adım: LLM ile Ön Söz Üret (Hızlı bir model kullanılması önerilir)
def generate_context_prefix(whole_document, chunk_content):
    prompt = f"""
    <document>{whole_document}</document>
    Here is the chunk we want to situate within the whole document:
    <chunk>{chunk_content}</chunk>
    Please give a short succinct context to situate this chunk within the overall document.
    """
    response = gemini.chat.completions.create(
        model="gemini-1.5-flash",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

# 2. Adım: Ön Söz ile Orijinal Chunk'ı Birleştir
prefix = generate_context_prefix(document, chunk)
contextualized_chunk = f"[{prefix}]: {chunk}"
