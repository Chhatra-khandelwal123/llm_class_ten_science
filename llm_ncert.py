import PyPDF2
import os

import pdfplumber

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

print("reached")
# Directory containing the chapter-wise PDFs
pdf_dir = "./books"
all_text = ""

for pdf_file in sorted(os.listdir(pdf_dir)):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_dir, pdf_file)
        text = extract_text_from_pdf(pdf_path)
        print(text)
        all_text += text + "\n\n"

# Save the combined text
with open("ncert_10th_science.txt", "w") as f:
    f.write(all_text)

print("Text extraction complete!")

def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

with open("ncert_10th_science.txt", "r") as file:
    text = file.read()

chunks = chunk_text(text)

# Save chunks for later use
with open("text_chunks.txt", "w") as f:
    for chunk in chunks:
        f.write(chunk + "\n---\n")


from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = [model.encode(chunk) for chunk in chunks]

# Convert to numpy array
embeddings = np.array(embeddings)

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, "ncert_faiss.index")
print("Embeddings saved!")


from transformers import pipeline
import faiss

# Load the embedding model and FAISS index
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("ncert_faiss.index")

qa_pipeline = pipeline("text-generation", model="gpt-3.5-turbo")

def get_similar_chunk(query):
    query_vector = model.encode([query])
    _, I = index.search(np.array(query_vector), 1)
    return chunks[I[0][0]]

def answer_question(question):
    context = get_similar_chunk(question)
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    response = qa_pipeline(prompt, max_length=200, num_return_sequences=1)
    return response[0]["generated_text"]

user_query = input("Ask your question: ")
print("Answer:", answer_question(user_query))
