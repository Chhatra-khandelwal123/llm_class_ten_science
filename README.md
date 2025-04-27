# 📘 NCERT Class 10 Science QA App using Retrieval-Augmented Generation (RAG)

Welcome to the **NCERT 10th Science QA App** – an interactive question-answering system powered by **Retrieval-Augmented Generation (RAG)**. This project allows students and learners to ask natural language questions from any chapter of the NCERT Class 10 Science book and get intelligent, contextual answers.

---

## 🚀 Features

- ✅ Uses the **NCERT Class 10 Science textbook** (all chapters included)
- ✅ **PDF Parsing** and **Chunking** for efficient retrieval
- ✅ Semantic **embedding search** using [FAISS](https://github.com/facebookresearch/faiss)
- ✅ Real-time **Q&A generation** using HuggingFace Transformers
- ✅ Built with an easy-to-use **Streamlit interface**

---

## 🧠 How It Works

This app uses the **RAG (Retrieval-Augmented Generation)** architecture:

1. **Ingestion**:
   - All chapters from the NCERT Class 10 Science book are stored as PDFs in the `/books` directory.
   - PDFs are parsed using `pdfplumber`.

2. **Chunking**:
   - Text is split into overlapping word chunks (200 words with 50-word overlap) to maintain context.

3. **Embedding & Indexing**:
   - Each chunk is converted to a vector using the `all-MiniLM-L6-v2` SentenceTransformer model.
   - Vectors are indexed using **FAISS** for fast similarity search.

4. **Retrieval**:
   - Given a user query, the most relevant text chunk is retrieved via similarity search.

5. **Answer Generation**:
   - The question and retrieved context are passed into a **Question-Answering pipeline** (`distilbert-base-cased-distilled-squad`) to generate the answer.

---

## 📂 Project Structure

