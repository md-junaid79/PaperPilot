"""
Retrieval Module
Handles vector search and context retrieval from Qdrant
"""

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import numpy as np


def setup_qdrant_client(url: str, api_key: str = None) -> QdrantClient:
    """
    Initialize Qdrant client
    
    Args:
        url: Qdrant server URL
        api_key: Optional API key for cloud
        
    Returns:
        QdrantClient instance
    """
    if api_key:
        client = QdrantClient(url=url, api_key=api_key)
    else:
        client = QdrantClient(url=url)
    
    print(f"✓ Connected to Qdrant at {url}")
    return client


def retrieve_relevant_chunks(
    query: str,
    client: QdrantClient,
    collection_name: str,
    embedding_model: SentenceTransformer,
    k: int = 5
) -> List[Dict]:
    """
    Retrieve top-k most relevant chunks for query
    
    Args:
        query: User query
        client: Qdrant client
        collection_name: Collection to search
        embedding_model: Model for query embedding
        k: Number of results to return
        
    Returns:
        List of retrieved chunks with scores
    """
    # Generate query embedding
    query_vector = embedding_model.encode(query).tolist()
    
    # Search in Qdrant
    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=k
    )
    
    # Format results
    retrieved_chunks = []
    for result in search_results:
        chunk = {
            'text': result.payload['text'],
            'paper_title': result.payload['paper_title'],
            'author': result.payload.get('author', 'Unknown'),
            'chunk_id': result.payload['chunk_id'],
            'score': result.score,
            'id': result.id
        }
        retrieved_chunks.append(chunk)
    
    return retrieved_chunks


def rerank_results(query: str, chunks: List[Dict], model: SentenceTransformer) -> List[Dict]:
    """
    Rerank retrieved chunks using cross-encoder or similarity
    
    Args:
        query: User query
        chunks: Retrieved chunks
        model: Embedding model
        
    Returns:
        Reranked chunks
    """
    # Simple reranking using semantic similarity
    query_embedding = model.encode(query)
    chunk_texts = [chunk['text'] for chunk in chunks]
    chunk_embeddings = model.encode(chunk_texts)
    
    # Calculate cosine similarity
    similarities = np.dot(chunk_embeddings, query_embedding) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    # Sort by similarity
    sorted_indices = np.argsort(similarities)[::-1]
    reranked = [chunks[i] for i in sorted_indices]
    
    # Update scores
    for i, chunk in enumerate(reranked):
        chunk['rerank_score'] = float(similarities[sorted_indices[i]])
    
    return reranked


def format_context_for_llm(chunks: List[Dict], max_chunks: int = 5) -> Tuple[str, List[Dict]]:
    """
    Format retrieved chunks into context string for LLM
    
    Args:
        chunks: Retrieved chunks
        max_chunks: Maximum number of chunks to include
        
    Returns:
        Tuple of (formatted_context, source_chunks)
    """
    context_parts = []
    sources = []
    
    for i, chunk in enumerate(chunks[:max_chunks], 1):
        context_parts.append(
            f"[Source {i}] Paper: {chunk['paper_title']}\n"
            f"Content: {chunk['text']}\n"
        )
        sources.append({
            'source_id': i,
            'paper_title': chunk['paper_title'],
            'author': chunk['author'],
            'score': chunk.get('score', 0),
            'text_preview': chunk['text'][:200] + '...'
        })
    
    context = "\n\n".join(context_parts)
    return context, sources


def retrieve_and_format(
    query: str,
    client: QdrantClient,
    collection_name: str,
    embedding_model: SentenceTransformer,
    k: int = 5,
    rerank: bool = True
) -> Tuple[str, List[Dict]]:
    """
    Complete retrieval pipeline: search, rerank, format
    
    Args:
        query: User query
        client: Qdrant client
        collection_name: Collection name
        embedding_model: Embedding model
        k: Number of results
        rerank: Whether to rerank results
        
    Returns:
        Tuple of (formatted_context, sources)
    """
    # Retrieve
    chunks = retrieve_relevant_chunks(query, client, collection_name, embedding_model, k)
    
    if not chunks:
        return "No relevant documents found.", []
    
    # Rerank if requested
    if rerank:
        chunks = rerank_results(query, chunks, embedding_model)
    
    # Format
    context, sources = format_context_for_llm(chunks, k)
    
    return context, sources


def list_all_papers(client: QdrantClient, collection_name: str) -> List[str]:
    """
    List all unique papers in collection
    
    Args:
        client: Qdrant client
        collection_name: Collection name
        
    Returns:
        List of unique paper titles
    """
    try:
        # Scroll through all points (limit to first 1000)
        results, _ = client.scroll(
            collection_name=collection_name,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )
        
        paper_titles = set()
        for point in results:
            if 'paper_title' in point.payload:
                paper_titles.add(point.payload['paper_title'])
        
        return sorted(list(paper_titles))
    except Exception as e:
        print(f"Error listing papers: {e}")
        return []