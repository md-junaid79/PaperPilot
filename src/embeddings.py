"""
Embeddings Module
Handles embedding generation and vector database storage
"""

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict
import uuid


def initialize_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Initialize sentence transformer model
    
    Args:
        model_name: HuggingFace model name
        
    Returns:
        SentenceTransformer model
    """
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"✓ Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def generate_embeddings(chunks: List[Dict], model) -> List[List[float]]:
    """
    Generate embeddings for text chunks
    
    Args:
        chunks: List of chunk dictionaries with 'text' field
        model: SentenceTransformer model
        
    Returns:
        List of embedding vectors
    """
    texts = [chunk['text'] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings.tolist()


def setup_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int):
    """
    Create or recreate Qdrant collection
    
    Args:
        client: Qdrant client
        collection_name: Name of collection
        vector_size: Dimension of embeddings
    """
    # Check if collection exists
    collections = client.get_collections().collections
    collection_names = [col.name for col in collections]
    
    if collection_name in collection_names:
        print(f"Collection '{collection_name}' already exists. Deleting...")
        client.delete_collection(collection_name)
    
    # Create new collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"✓ Created collection: {collection_name}")


def store_in_qdrant(
    chunks: List[Dict], 
    embeddings: List[List[float]], 
    client: QdrantClient,
    collection_name: str
) -> int:
    """
    Store chunks and embeddings in Qdrant
    
    Args:
        chunks: List of chunk dictionaries
        embeddings: Corresponding embeddings
        client: Qdrant client
        collection_name: Collection name
        
    Returns:
        Number of points stored
    """
    points = []
    
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                'text': chunk['text'],
                'paper_title': chunk['paper_title'],
                'author': chunk.get('author', 'Unknown'),
                'chunk_id': chunk['chunk_id'],
                'num_pages': chunk.get('num_pages', 0)
            }
        )
        points.append(point)
    
    # Upload in batches
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch
        )
    
    print(f"✓ Stored {len(points)} chunks in Qdrant")
    return len(points)


def get_collection_info(client: QdrantClient, collection_name: str) -> Dict:
    """
    Get information about collection
    
    Args:
        client: Qdrant client
        collection_name: Collection name
        
    Returns:
        Collection info dictionary
    """
    try:
        info = client.get_collection(collection_name)
        return {
            'name': collection_name,
            'vectors_count': info.vectors_count,
            'points_count': info.points_count,
            'status': info.status
        }
    except Exception as e:
        return {'error': str(e)}


def ingest_documents(
    pdf_chunks_list: List[List[Dict]],
    qdrant_url: str,
    collection_name: str,
    embedding_model_name: str = "all-MiniLM-L6-v2"
) -> Dict:
    """
    Complete ingestion pipeline
    
    Args:
        pdf_chunks_list: List of chunk lists (one per PDF)
        qdrant_url: Qdrant server URL
        collection_name: Collection name
        embedding_model_name: Embedding model name
        
    Returns:
        Ingestion statistics
    """
    # Initialize
    model = initialize_embedding_model(embedding_model_name)
    client = QdrantClient(url=qdrant_url)
    
    # Flatten all chunks
    all_chunks = [chunk for chunks in pdf_chunks_list for chunk in chunks]
    
    # Setup collection
    vector_size = model.get_sentence_embedding_dimension()
    setup_qdrant_collection(client, collection_name, vector_size)
    
    # Generate embeddings
    print(f"Generating embeddings for {len(all_chunks)} chunks...")
    embeddings = generate_embeddings(all_chunks, model)
    
    # Store in Qdrant
    num_stored = store_in_qdrant(all_chunks, embeddings, client, collection_name)
    
    return {
        'total_chunks': len(all_chunks),
        'stored_chunks': num_stored,
        'collection_name': collection_name,
        'embedding_model': embedding_model_name
    }