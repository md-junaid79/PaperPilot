"""
Agent Module
Handles LLM integration, query routing, and RAG chain
"""

from groq import Groq
from typing import Dict, List, Tuple
import re


def setup_groq_client(api_key: str) -> Groq:
    """
    Initialize Groq client
    
    Args:
        api_key: Groq API key
        
    Returns:
        Groq client instance
    """
    client = Groq(api_key=api_key)
    print("✓ Groq client initialized")
    return client


def route_query(query: str) -> str:
    """
    Classify query type based on keywords and patterns
    
    Args:
        query: User query
        
    Returns:
        Query type: 'direct_lookup', 'summarization', or 'data_extraction'
    """
    query_lower = query.lower()
    
    # Data extraction patterns
    data_patterns = [
        r'what.*(?:accuracy|precision|recall|f1|score|metric|result|performance)',
        r'(?:list|extract).*(?:numbers|values|metrics|results)',
        r'what.*reported',
        r'what.*achieve',
    ]
    
    # Summarization patterns
    summary_patterns = [
        r'summarize',
        r'explain.*methodology',
        r'what.*main.*(?:contribution|finding|result)',
        r'describe.*(?:approach|method)',
        r'key.*(?:points|findings|insights)',
    ]
    
    # Check patterns
    for pattern in data_patterns:
        if re.search(pattern, query_lower):
            return 'data_extraction'
    
    for pattern in summary_patterns:
        if re.search(pattern, query_lower):
            return 'summarization'
    
    # Default to direct lookup
    return 'direct_lookup'


def create_prompt(query: str, context: str, query_type: str) -> str:
    """
    Create specialized prompt based on query type
    
    Args:
        query: User query
        context: Retrieved context
        query_type: Type of query
        
    Returns:
        Formatted prompt
    """
    base_prompt = f"""You are a helpful research assistant analyzing academic papers.

Context from research papers:
{context}

User Question: {query}
"""
    
    if query_type == 'data_extraction':
        instruction = """
Extract specific data, metrics, or numerical results from the context. Be precise and cite the exact values mentioned.
If specific numbers are not found, clearly state that.

Answer:"""
    
    elif query_type == 'summarization':
        instruction = """
Provide a clear and concise summary of the relevant information from the context.
Focus on the key points and main ideas.

Answer:"""
    
    else:  # direct_lookup
        instruction = """
Answer the question directly based on the provided context.
Be concise and accurate. If the answer is not in the context, say so.

Answer:"""
    
    return base_prompt + instruction


def generate_response(
    client: Groq,
    prompt: str,
    model: str = "openai/gpt-oss-120b",
    temperature: float = 0.3,
    max_tokens: int = 1024,
    stream: bool = False
) -> str:
    """
    Generate response from Groq LLM
    
    Args:
        client: Groq client
        prompt: Formatted prompt
        model: Model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        stream: Whether to stream response
        
    Returns:
        Generated response text
    """
    messages = [
        {"role": "system", "content": "You are a helpful research assistant specializing in analyzing academic papers."},
        {"role": "user", "content": prompt}
    ]
    
    if stream:
        # For streaming (used in UI)
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
    else:
        # For non-streaming
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


def execute_rag_query(
    query: str,
    context: str,
    sources: List[Dict],
    groq_client: Groq,
    model: str = "openai/gpt-oss-120b"
) -> Dict:
    """
    Execute complete RAG query
    
    Args:
        query: User query
        context: Retrieved context
        sources: Source documents
        groq_client: Groq client
        model: Model name
        
    Returns:
        Dictionary with answer and metadata
    """
    # Route query
    query_type = route_query(query)
    
    # Create prompt
    prompt = create_prompt(query, context, query_type)
    
    # Generate response
    answer = generate_response(groq_client, prompt, model=model)
    
    return {
        'answer': answer,
        'query_type': query_type,
        'sources': sources,
        'num_sources': len(sources)
    }


def execute_rag_query_streaming(
    query: str,
    context: str,
    groq_client: Groq,
    model: str = "openai/gpt-oss-120b"
):
    """
    Execute RAG query with streaming response (for UI)
    
    Args:
        query: User query
        context: Retrieved context
        groq_client: Groq client
        model: Model name
        
    Yields:
        Response chunks
    """
    # Route and create prompt
    query_type = route_query(query)
    prompt = create_prompt(query, context, query_type)
    
    # Stream response
    stream = generate_response(groq_client, prompt, model=model, stream=True)
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def format_sources_display(sources: List[Dict]) -> str:
    """
    Format sources for display
    
    Args:
        sources: List of source documents
        
    Returns:
        Formatted string
    """
    if not sources:
        return "No sources used."
    
    formatted = "**Sources:**\n\n"
    for src in sources:
        formatted += f"📄 **{src['paper_title']}**\n"
        formatted += f"   Author: {src['author']}\n"
        formatted += f"   Relevance: {src['score']:.3f}\n"
        formatted += f"   Preview: {src['text_preview']}\n\n"
    
    return formatted