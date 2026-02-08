"""
Document Processing Module
Handles PDF extraction, chunking, and metadata creation
"""

import pymupdf as fitz
import re
from typing import Dict, List, Tuple
from pathlib import Path


def extract_pdf_content(file_path: str) -> Dict:
    """
    Extract text and metadata from PDF
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        Dictionary with title, text, metadata, and structure
    """
    doc = fitz.open(file_path)
    
    # Extract metadata
    metadata = doc.metadata
    title = metadata.get('title', Path(file_path).stem)
    
    # Extract text with page numbers
    full_text = ""
    pages_content = []
    
    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text()
        full_text += f"\n--- Page {page_num} ---\n{page_text}"
        pages_content.append({
            'page_num': page_num,
            'text': page_text
        })
    
    doc.close()
    
    # Try to identify sections
    sections = extract_sections(full_text)
    
    return {
        'title': title,
        'full_text': full_text,
        'pages': pages_content,
        'sections': sections,
        'metadata': {
            'author': metadata.get('author', 'Unknown'),
            'subject': metadata.get('subject', ''),
            'num_pages': len(pages_content)
        }
    }


def extract_sections(text: str) -> List[Dict]:
    """
    Extract common paper sections (Abstract, Introduction, etc.)
    
    Args:
        text: Full document text
        
    Returns:
        List of sections with titles and content
    """
    sections = []
    
    # Common section headers in academic papers
    section_patterns = [
        r'\n(Abstract)\n',
        r'\n(Introduction)\n',
        r'\n(\d+\.?\s+Introduction)\n',
        r'\n(Related Work)\n',
        r'\n(Methodology)\n',
        r'\n(Method)\n',
        r'\n(Experiments?)\n',
        r'\n(Results?)\n',
        r'\n(Discussion)\n',
        r'\n(Conclusion)\n',
        r'\n(References)\n',
    ]
    
    combined_pattern = '|'.join(section_patterns)
    matches = list(re.finditer(combined_pattern, text, re.IGNORECASE))
    
    for i, match in enumerate(matches):
        section_title = match.group(0).strip()
        start_pos = match.end()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_content = text[start_pos:end_pos].strip()
        
        if section_content:
            sections.append({
                'title': section_title,
                'content': section_content[:1000]  # Preview
            })
    
    return sections


def extract_tables(file_path: str) -> List[Dict]:
    """
    Extract tables from PDF (basic implementation)
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        List of extracted tables with metadata
    """
    doc = fitz.open(file_path)
    tables = []
    
    for page_num, page in enumerate(doc, start=1):
        # Find tables by looking for grid-like structures
        text = page.get_text()
        
        # Simple heuristic: lines with multiple | or tab separators
        lines = text.split('\n')
        potential_table_lines = []
        
        for line in lines:
            if line.count('|') > 2 or line.count('\t') > 2:
                potential_table_lines.append(line)
        
        if len(potential_table_lines) > 2:
            tables.append({
                'page': page_num,
                'content': '\n'.join(potential_table_lines),
                'num_rows': len(potential_table_lines)
            })
    
    doc.close()
    return tables


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks
    
    Args:
        text: Input text
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    # Split by sentences first (basic)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Start new chunk with overlap
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def create_chunks_with_metadata(pdf_dict: Dict, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    """
    Create chunks with associated metadata
    
    Args:
        pdf_dict: Output from extract_pdf_content
        chunk_size: Target chunk size
        overlap: Overlap size
        
    Returns:
        List of chunks with metadata
    """
    chunks = chunk_text(pdf_dict['full_text'], chunk_size, overlap)
    
    chunks_with_metadata = []
    for idx, chunk in enumerate(chunks):
        chunk_dict = {
            'text': chunk,
            'chunk_id': idx,
            'paper_title': pdf_dict['title'],
            'author': pdf_dict['metadata']['author'],
            'num_pages': pdf_dict['metadata']['num_pages'],
        }
        chunks_with_metadata.append(chunk_dict)
    
    return chunks_with_metadata


def process_pdf_file(file_path: str, chunk_size: int = 500, overlap: int = 50) -> Tuple[Dict, List[Dict]]:
    """
    Complete pipeline: extract and chunk PDF
    
    Args:
        file_path: Path to PDF
        chunk_size: Chunk size
        overlap: Overlap size
        
    Returns:
        Tuple of (pdf_content_dict, chunks_list)
    """
    pdf_content = extract_pdf_content(file_path)
    chunks = create_chunks_with_metadata(pdf_content, chunk_size, overlap)
    
    print(f"✓ Processed: {pdf_content['title']}")
    print(f"  Pages: {pdf_content['metadata']['num_pages']}")
    print(f"  Chunks: {len(chunks)}")
    
    return pdf_content, chunks