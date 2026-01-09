"""
Interactive RAG query script:
1. Rewrites follow-up questions into standalone queries
2. Embeds the rewritten query using Gemini
3. Retrieves relevant chunks from Pinecone
4. Answers ONLY from retrieved context
"""

# 1. Environment setup

import os
from dotenv import load_dotenv

load_dotenv()

# 2. Imports

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from google.genai import Client 
llm = Client(api_key=os.getenv("GEMINI_API_KEY"))

# 3. Initialize Gemini (LLM)

# 3. Conversation memory (Gemini format)
HISTORY = []

# 4. Query Rewriting (follow-up → standalone)

def transform_query(question: str, llm) -> str:
    messages = [
        {
            "role": "user",
            "parts": [{
                "text": (
                    "You are a query rewriting expert. "
                    "Rephrase the follow-up user question into a complete, standalone question. "
                    "Only output the rewritten question and nothing else."
                    f"Question: {question}"
                )
            }]
        },
        
    ]

    response = llm.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=messages
    )

    return response.text.strip()

# 5. Main RAG logic

def answer_question(user_question: str, llm):
    # ---- Step 1: Rewrite query
    rewritten_query = transform_query(user_question,llm)

    # ---- Step 2: Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    query_vector = embeddings.embed_query(rewritten_query)

    # ---- Step 3: Pinecone retrieval
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    index_name = os.getenv("PINECONE_INDEX_NAME")
    index = pc.Index(index_name)

    search_results = index.query(
        vector=query_vector,
        top_k=10,
        include_metadata=True,
    )

    context = "\n\n---\n\n".join(
        match["metadata"]["text"] for match in search_results["matches"]
    )

    # ---- Step 4: Answer generation
    # --- retrieve context from Pinecone (unchanged) ---
    # context = ...

    messages = [
        {
            "role": "user",
            "parts": [{
                "text": (
                    "You are a Science Expert.\n"
                    "Answer ONLY using the context below.\n"
                    'If the answer is not present, say: '
                    '"I could not find the answer in the provided document."\n\n'
                    f"Context:\n{context}"
                )
            }]
        }
    ]

    # 🔑 Add conversation history
    messages.extend(HISTORY)

    # 🔑 Add current user question
    messages.append({
        "role": "user",
        "parts": [{"text": rewritten_query}]
    })

    response = llm.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=messages
    )

    answer = response.text.strip()

    # 🔑 Persist conversation
    HISTORY.append({
        "role": "user",
        "parts": [{"text": rewritten_query}]
    })

    HISTORY.append({
        "role": "model",
        "parts": [{"text": answer}]
    })

    print("\nAnswer:\n")
    print(answer)
    print("\n" + "=" * 60 + "\n")

# 6. Interactive loop (NO recursion)

def main():
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Ask me anything--> ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        answer_question(question,llm)

# 7. Entry point

if __name__ == "__main__":
    main()
