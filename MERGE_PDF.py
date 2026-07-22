import pymupdf
import glob
import os
klasor_yolu = r"C:\Users\tft\DeneyselRAG\docs"
pdf_dosyalari = []

for kok_dizin, alt_klasorler, dosyalar in os.walk(klasor_yolu):
    for dosya in dosyalar:
        if dosya.endswith(".pdf"):
            # Dosya adını tam yoluna dönüştürüp listeye ekler
            tam_yol = os.path.join(kok_dizin, dosya)
            pdf_dosyalari.append(tam_yol)

print(pdf_dosyalari)
    

doc_a = pymupdf.open(pdf_dosyalari[0]) # open the 1st document
for i in range(1, len(pdf_dosyalari)):
     doc_b = pymupdf.open(pdf_dosyalari[i]) # open the next document
     doc_a.insert_file(doc_b) 
#    doc_a.insert_pdf(doc_b) # merge the documents
#doc_a.save("merged.pdf") # save the merged document
doc_a.save("birlesmis.pdf")


# insert_pdf yerine insert_file deneyin
