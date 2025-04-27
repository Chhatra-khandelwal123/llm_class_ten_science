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

ncert-science-qa/
│
├── 📂 books/                  
│   └── *.pdf                 # All chapters of the Class 10 NCERT Science textbook (PDF files)
│
├── 📄 app.py                 
│   └── Main Streamlit application implementing RAG-based QA
│
├── 📄 requirement.text       
│   └── List of Python packages required to run the app (should be renamed to requirements.txt)
│
└── 📄 README.md              
    └── Project documentation (you’ll place the README I gave you here)


🧪 Example Questions You Can Ask
"What is the role of the placenta during pregnancy?"

"Explain the law of conservation of energy with an example."

"What happens during rusting of iron?"

"What is the difference between AC and DC current?"


🛠️ Tech Stack

Component	Technology
Interface	Streamlit
PDF Parsing	pdfplumber
Embeddings	SentenceTransformers (all-MiniLM-L6-v2)
Vector Search	FAISS
QA Model	distilBERT (SQuAD fine-tuned)
RAG Pipeline	Custom implementation


📌 To-Do / Future Improvements
 Switch to more powerful models (e.g. GPT-4 or mistral)

 Support for Hindi/other languages

 Chapter-wise navigation

 Upload custom textbooks

 Deploy to Hugging Face Spaces or Streamlit Cloud

🤝 Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request if you'd like to help improve the project.


🌟 Star this Repository
If you found this project helpful or interesting, please consider giving it a ⭐ on GitHub. It helps others discover it too!

