import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Load environment variables
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "imdb_movies")

def prepare_documents(csv_path=None):
    """Read CSV and prepare documents for embedding"""
    if csv_path is None:
        # Get the directory where this script is located
        script_dir = Path(__file__).parent
        csv_path = script_dir / "data" / "imdb_top_1000.csv"
    
    print(f"📖 Reading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    documents = []
    for idx, row in df.iterrows():
        # Create rich text content for embedding
        content = f"""
Title: {row.get('Series_Title', 'N/A')}
Year: {row.get('Released_Year', 'N/A')}
Rating: {row.get('IMDB_Rating', 'N/A')}/10
Genre: {row.get('Genre', 'N/A')}
Director: {row.get('Director', 'N/A')}
Stars: {row.get('Star1', '')}, {row.get('Star2', '')}, {row.get('Star3', '')}, {row.get('Star4', '')}
Overview: {row.get('Overview', 'N/A')}
Runtime: {row.get('Runtime', 'N/A')}
Certificate: {row.get('Certificate', 'N/A')}
Gross: {row.get('Gross', 'N/A')}
"""
        
        # Create metadata
        metadata = {
            "title": str(row.get('Series_Title', 'Unknown')),
            "year": str(row.get('Released_Year', 'N/A')),
            "rating": str(row.get('IMDB_Rating', 'N/A')),
            "genre": str(row.get('Genre', 'N/A')),
            "director": str(row.get('Director', 'N/A')),
            "stars": f"{row.get('Star1', '')}, {row.get('Star2', '')}, {row.get('Star3', '')}, {row.get('Star4', '')}",
            "overview": str(row.get('Overview', 'N/A'))[:500],  # Limit overview length
            "runtime": str(row.get('Runtime', 'N/A')),
            "certificate": str(row.get('Certificate', 'N/A')),
            "gross": str(row.get('Gross', 'N/A')),
            "meta_score": str(row.get('Meta_score', 'N/A')),
            "no_of_votes": str(row.get('No_of_Votes', 'N/A'))
        }
        
        doc = Document(
            page_content=content.strip(),
            metadata=metadata
        )
        documents.append(doc)
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1} movies...")
    
    print(f"✅ Total documents prepared: {len(documents)}")
    return documents

def create_qdrant_collection():
    """Create Qdrant collection if it doesn't exist"""
    print(f"🔧 Connecting to Qdrant at {QDRANT_URL}...")
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )
    
    # Check if collection exists
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    
    if COLLECTION_NAME in collection_names:
        print(f"⚠️  Collection '{COLLECTION_NAME}' already exists!")
        response = input("Do you want to delete and recreate it? (yes/no): ")
        if response.lower() == 'yes':
            client.delete_collection(COLLECTION_NAME)
            print(f"🗑️  Deleted collection '{COLLECTION_NAME}'")
        else:
            print("Keeping existing collection.")
            return client
    
    # Create new collection
    print(f"📦 Creating collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=1536,  # text-embedding-3-small dimension
            distance=Distance.COSINE
        )
    )
    print(f"✅ Collection '{COLLECTION_NAME}' created!")
    return client

def upload_to_qdrant(documents):
    """Upload documents to Qdrant"""
    print("🚀 Initializing embeddings...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )
    
    print(f"📤 Uploading {len(documents)} documents to Qdrant...")
    print("⏳ This may take several minutes depending on the dataset size...")
    
    vectorstore = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
        force_recreate=False
    )
    
    print("✅ Upload complete!")
    return vectorstore

def test_search(vectorstore):
    """Test the vector store with a sample query"""
    print("\n🔍 Testing search functionality...")
    test_queries = [
        "action movies with high ratings",
        "romantic comedies from the 90s",
        "Christopher Nolan movies"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        results = vectorstore.similarity_search(query, k=3)
        for i, doc in enumerate(results, 1):
            print(f"  {i}. {doc.metadata['title']} ({doc.metadata['year']}) - Rating: {doc.metadata['rating']}")

def main():
    print("=" * 60)
    print("🎬 IMDB Top 1000 to Qdrant Ingestion")
    print("=" * 60)
    
    # Step 1: Prepare documents
    documents = prepare_documents()
    
    # Step 2: Create collection
    client = create_qdrant_collection()
    
    # Step 3: Upload documents
    vectorstore = upload_to_qdrant(documents)
    
    # Step 4: Test search
    test_search(vectorstore)
    
    print("\n" + "=" * 60)
    print("✨ All done! Your IMDB dataset is ready in Qdrant!")
    print("=" * 60)

if __name__ == "__main__":
    main()