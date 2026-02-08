"""
Arxiv Tools Module
Handles paper search and download from Arxiv
"""

import arxiv
from typing import List, Dict
import os


def search_arxiv(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search Arxiv for papers matching query
    
    Args:
        query: Search query
        max_results: Maximum number of results
        
    Returns:
        List of paper metadata dictionaries
    """
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    papers = []
    for result in search.results():
        paper = {
            'title': result.title,
            'arxiv_id': result.entry_id.split('/')[-1],
            'authors': [author.name for author in result.authors],
            'abstract': result.summary,
            'published': result.published.strftime('%Y-%m-%d'),
            'pdf_url': result.pdf_url,
            'categories': result.categories
        }
        papers.append(paper)
    
    return papers


def download_paper(arxiv_id: str, download_dir: str = './data/uploads') -> str:
    """
    Download paper from Arxiv by ID
    
    Args:
        arxiv_id: Arxiv paper ID (e.g., '2103.00020')
        download_dir: Directory to save PDF
        
    Returns:
        Path to downloaded PDF file
    """
    # Clean arxiv_id
    arxiv_id = arxiv_id.replace('v1', '').replace('v2', '').replace('v3', '')
    
    # Create directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)
    
    # Search for the paper
    search = arxiv.Search(id_list=[arxiv_id])
    paper = next(search.results())
    
    # Download
    filename = f"{arxiv_id.replace('/', '_')}.pdf"
    filepath = os.path.join(download_dir, filename)
    
    paper.download_pdf(dirpath=download_dir, filename=filename)
    
    print(f"✓ Downloaded: {paper.title}")
    print(f"  Path: {filepath}")
    
    return filepath


def format_arxiv_results(papers: List[Dict]) -> str:
    """
    Format Arxiv search results for display
    
    Args:
        papers: List of paper dictionaries
        
    Returns:
        Formatted string
    """
    if not papers:
        return "No papers found."
    
    formatted = ""
    for i, paper in enumerate(papers, 1):
        formatted += f"**{i}. {paper['title']}**\n"
        formatted += f"   Authors: {', '.join(paper['authors'][:3])}"
        if len(paper['authors']) > 3:
            formatted += f" et al."
        formatted += f"\n   Published: {paper['published']}\n"
        formatted += f"   Arxiv ID: {paper['arxiv_id']}\n"
        formatted += f"   Abstract: {paper['abstract'][:200]}...\n\n"
    
    return formatted


def create_arxiv_search_tool():
    """
    Create a tool definition for function calling
    
    Returns:
        Tool dictionary for Groq function calling
    """
    tool = {
        "type": "function",
        "function": {
            "name": "search_arxiv",
            "description": "Search for academic papers on Arxiv based on a natural language query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for finding papers (e.g., 'transformer architecture', 'reinforcement learning')"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    }
    return tool


def execute_arxiv_search_with_llm(user_query: str, groq_client, model: str = "llama-3.1-70b-versatile") -> Dict:
    """
    Use LLM to determine if query needs Arxiv search and extract search terms
    
    Args:
        user_query: User's natural language query
        groq_client: Groq client
        model: Model name
        
    Returns:
        Dictionary with search results or None
    """
    # Check if query is about finding papers
    find_paper_keywords = ['find paper', 'search paper', 'look for paper', 'arxiv', 'research on']
    
    should_search = any(keyword in user_query.lower() for keyword in find_paper_keywords)
    
    if not should_search:
        return None
    
    # Extract search terms using LLM
    prompt = f"""Extract the key search terms for an Arxiv paper search from this query.
Return ONLY the search terms, nothing else.

Query: {user_query}

Search terms:"""
    
    messages = [{"role": "user", "content": prompt}]
    response = groq_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=100
    )
    
    search_terms = response.choices[0].message.content.strip()
    
    # Search Arxiv
    papers = search_arxiv(search_terms, max_results=5)
    
    return {
        'search_terms': search_terms,
        'papers': papers,
        'num_results': len(papers)
    }