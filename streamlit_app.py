import streamlit as st
import os
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

# Define functions from the provided code
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Load embeddings and FAISS index
@st.cache_resource
def load_embeddings():
    pdf_dir = "./books"
    all_text = ""

    for pdf_file in sorted(os.listdir(pdf_dir)):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            text = extract_text_from_pdf(pdf_path)
            all_text += text + "\n\n"

    # Chunk the text
    chunks = chunk_text(all_text)

    # Load the model and create embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = [model.encode(chunk) for chunk in chunks]
    embeddings = np.array(embeddings)

    # Build FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return model, index, chunks

# Function to get the most similar chunk from the FAISS index
def get_similar_chunk(query, model, index, chunks):
    query_vector = model.encode([query])
    _, I = index.search(np.array(query_vector), 1)
    return chunks[I[0][0]]

# Load the QA pipeline
qa_pipeline = pipeline("text-generation", model="gpt-3.5-turbo")

# Function to generate an answer based on the query
def answer_question(query, model, index, chunks):
    context = get_similar_chunk(query, model, index, chunks)
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    response = qa_pipeline(prompt, max_length=200, num_return_sequences=1)
    return response[0]["generated_text"]

# Streamlit UI
st.title("NCERT 10th Science Textbook Q&A")
st.write(
    "Welcome to the NCERT 10th Science Textbook Question Answering App. "
    "Ask any question related to the 10th-grade science textbook and get an answer!"
)

# Ask for user query
user_query = st.text_input("Enter your question:")

# Load the model, index, and chunks
if user_query:
    model, index, chunks = load_embeddings()

    # Get the answer for the question
    answer = answer_question(user_query, model, index, chunks)

    # Show the answer
    st.write("Answer: ", answer)
