# src/rag_tool.py
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # ← FIX INI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
load_dotenv()

# Global variables to store the vector store
vectorstore = None
qa_chain = None

def load_vectorstore(persist_directory="./data/vectorstore"):
    """Load the FAISS vector store"""
    global vectorstore, qa_chain
    
    if vectorstore is None:
        embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
        vectorstore = FAISS.load_local(
            persist_directory, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        llm = ChatOpenAI(
            model_name=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            temperature=0
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True
        )
    
    return vectorstore, qa_chain

def rag_answer(query: str, top_k=5):
    """
    Answer a query using RAG (Retrieval Augmented Generation)
    
    Args:
        query: The question to answer
        top_k: Number of documents to retrieve
        
    Returns:
        dict with 'answer' and 'sources' keys
    """
    try:
        # Load vectorstore if not already loaded
        vs, chain = load_vectorstore()
        
        # Get answer and source documents
        result = chain({"query": query})
        
        # Format sources
        sources = []
        for doc in result.get("source_documents", []):
            metadata = doc.metadata
            sources.append({
                "meta": {
                    "title": metadata.get("title", "Unknown"),
                    "year": metadata.get("year", "N/A"),
                    "genre": metadata.get("genre", "N/A")
                },
                "content": doc.page_content[:200]  # First 200 chars
            })
        
        return {
            "answer": result["result"],
            "sources": sources
        }
        
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "sources": []
        }

# For testing
if __name__ == "__main__":
    result = rag_answer("What are some good action movies?")
    print("Answer:", result["answer"])
    print("\nSources:")
    for src in result["sources"]:
        meta = src["meta"]
        print(f"- {meta['title']} ({meta['year']}) — {meta['genre']}")