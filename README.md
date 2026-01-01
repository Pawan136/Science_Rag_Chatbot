#  Science RAG Chatbot  
A PDF-Grounded Scientific Question Answering System

## Project Architecture

![Science RAG Chatbot](science_rag_chatbot.png)

---

##  Overview

Science RAG Chatbot is a **Retrieval-Augmented Generation (RAG)** application that answers user questions strictly based on the content of uploaded scientific PDFs.

The system prevents hallucinations by retrieving relevant document chunks from a **vector database (Pinecone)** and generating answers only from the retrieved context using **Google Gemini**.

This project demonstrates **end-to-end GenAI engineering**, including document ingestion, embedding, vector search, query rewriting, and conversational UI.



##  Key Features

-  Upload and index scientific PDF documents  
-  Recursive text chunking with overlap for better retrieval  
-  Semantic embeddings using **Gemini `text-embedding-004`**  
-  Vector similarity search with **Pinecone**  
-  Follow-up question rewriting into standalone queries  
-  Conversational interface built with **Streamlit**  
-  Zero hallucination policy — answers only from document context  
-  Admin controls to clear chat and re-index documents  

---

## 🧠 Architecture (RAG Pipeline)

1. **PDF Upload**  
   PDF is uploaded through the Streamlit UI and temporarily stored.

2. **Document Processing**  
   - PDF pages are extracted  
   - Text is split into overlapping chunks  

3. **Embedding**  
   Each chunk is converted into a dense vector using Gemini embeddings.

4. **Vector Storage**  
   Vectors and metadata are stored in Pinecone.

5. **Query Rewriting**  
   Follow-up questions are rewritten into standalone search queries.

6. **Retrieval + Answering**  
   - Top-K relevant chunks are retrieved  
   - Gemini generates an answer strictly from retrieved context  

---

##  Tech Stack

| Layer | Technology |
|-----|-----------|
| UI | Streamlit |
| LLM | Google Gemini |
| Embeddings | Gemini `text-embedding-004` |
| Vector DB | Pinecone |
| Framework | LangChain |
| Language | Python |

---

##  Project Structure

```
Science_Rag_Chatbot/
├── .env.example
├── .gitignore
├── app.py
├── HEAT Chapter.pdf # Sample scientific document
├── README.md
├── science_rag_chatbot.png
├── Sciquery.py # Query + RAG logic 
└── scirag.py # Indexing / ingestion logic
```

---


---

## ⚙️ Setup Instructions

### Create Virtual Environment

```
python -m venv .venv
```
Activate it:

Windows
```
.venv\Scripts\activate
```
---
Linux / macOS
```
source .venv/bin/activate
```
###  Install Dependencies
```
pip install -r requirements.txt
```
---

## Environment Variables

Create a .env file (this file must NOT be committed):
```
GEMINI_API_KEY=your_gemini_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=your_index_name_here
```

A reference file is provided as ".env.example".

---

## Run the Application
```
streamlit run app.py
```
After running the command, Streamlit will display a local URL (usually http://localhost:8501). Open that URL in your browser.

---

**Use Cases**

Scientific paper Q&A

Research assistants

Academic document exploration

Domain-specific chatbots

---

**Author**

**Pawan Kumar Chaurasia**





