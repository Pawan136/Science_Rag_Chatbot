import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone  # New SDK
from google.genai import Client

# 1. Configuration (Developer Only - Hidden from UI)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# UI Settings
st.set_page_config(page_title="Science RAG Expert", page_icon="🔬", layout="centered")

# Attractive Custom Header
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>🔬 Science RAG Expert</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>I answer questions strictly based on your provided PDF.</p>", unsafe_allow_html=True)

# 2. Initialize Clients
if not (GEMINI_API_KEY and PINECONE_API_KEY and PINECONE_INDEX_NAME):
    st.error("Missing Environment Variables! Please check your .env file.")
    st.stop()

# Initialize Gemini Client
genai_client = Client(api_key=GEMINI_API_KEY)
# Initialize Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GEMINI_API_KEY
)

# 3. Utility Functions
def process_pdf(uploaded_file):
    """Processes PDF and uploads to Pinecone"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        raw_docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunked_docs = text_splitter.split_documents(raw_docs)
        
        # Upload to Pinecone
        PineconeVectorStore.from_documents(
            documents=chunked_docs,
            embedding=embeddings,
            index_name=PINECONE_INDEX_NAME,
            pinecone_api_key=PINECONE_API_KEY,
        )
        return len(chunked_docs)
    finally:
        os.remove(tmp_path)

def get_standalone_query(user_question, history):
    """Rewrites follow-up questions using Gemini"""
    history_context = "\n".join([f"{m['role']}: {m['content']}" for m in history[-3:]])
    prompt = (
        f"Conversation History:\n{history_context}\n\n"
        f"Follow-up Question: {user_question}\n\n"
        "Rephrase the follow-up question into a complete standalone search query. Output ONLY the query text."
    )
    response = genai_client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text.strip()

# 4. Sidebar for Document Management
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.title("Admin Console")
    st.markdown("---")
    
    st.subheader("📁 Knowledge Base")
    pdf_file = st.file_uploader("Upload PDF Document", type="pdf")
    if pdf_file and st.button("Index Document"):
        with st.spinner("Processing..."):
            count = process_pdf(pdf_file)
            st.success(f"Indexed {count} chunks successfully!")
            
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# 5. Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching document..."):
            # A. Rewrite query if there is history
            if len(st.session_state.messages) > 1:
                search_query = get_standalone_query(prompt, st.session_state.messages[:-1])
            else:
                search_query = prompt

            # B. Retrieve Context
            pc = Pinecone(api_key=PINECONE_API_KEY)
            index = pc.Index(PINECONE_INDEX_NAME)
            query_vector = embeddings.embed_query(search_query)
            
            search_results = index.query(
                vector=query_vector,
                top_k=5,
                include_metadata=True,
            )
            
            context = "\n\n---\n\n".join([match["metadata"]["text"] for match in search_results["matches"]])

            # C. Generate Response
            final_prompt = (
                "You are a Science Expert.\n"
                "Answer ONLY using the provided context. If the answer isn't there, say you don't know.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {search_query}"
            )
            
            response = genai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=final_prompt
            )
            
            full_response = response.text
            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})