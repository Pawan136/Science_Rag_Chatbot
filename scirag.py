"""
This script:
1. Loads a PDF document
2. Splits it into chunks
3. Converts chunks into embeddings using Gemini
4. Stores embeddings into Pinecone vector database
"""

# 1. Environment setup

import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# 2. Import LangChain modules

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# 3. Import Pinecone client

from pinecone import Pinecone

# 4. Resolve file path

# Get current file directory (similar to __dirname in Node)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_PATH = os.path.join(BASE_DIR, "HEAT Chapter.pdf")

print("PDF Path:", PDF_PATH)

# 5. Load PDF

loader = PyPDFLoader(PDF_PATH)
raw_docs = loader.load()

print("PDF loaded successfully")

# 6. Chunk the documents

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunked_docs = text_splitter.split_documents(raw_docs)

print(f"Chunking completed: {len(chunked_docs)} chunks created")

# 7. Configure embedding model

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

print("Embedding model configured")

# 8. Initialize Pinecone

print("Using Pinecone index:", os.getenv("PINECONE_INDEX_NAME"))

index_name = os.getenv("PINECONE_INDEX_NAME")

print("Pinecone configured")

# 9. Store embeddings in Pinecone

vectorstore = PineconeVectorStore.from_documents(
    documents=chunked_docs,
    embedding=embeddings,
    index_name=os.getenv("PINECONE_INDEX_NAME"),
    pinecone_api_key=os.getenv("PINECONE_API_KEY"),
)


print("Data stored successfully in Pinecone")
