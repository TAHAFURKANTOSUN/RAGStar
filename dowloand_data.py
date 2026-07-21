import requests

# PMC7105930 PDF direkt indirme adresi
pdf_url = "https://pmc.ncbi.nlm.nih.gov/articles/PMC7105930/pdf/v22i3e16642.pdf"

response = requests.get(pdf_url, timeout=30)
response.raise_for_status()
with open("dokuman.pdf", "wb") as f:
    f.write(response.content)
print("PDF başarıyla indirildi!")