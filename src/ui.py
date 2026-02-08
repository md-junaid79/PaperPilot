"""
Streamlit UI for Research Paper Q&A Agent
"""

import streamlit as st
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent))

from doc_processing import process_pdf_file
from embeddings import initialize_embedding_model, generate_embeddings, store_in_qdrant, setup_qdrant_collection
from retrieval import setup_qdrant_client, retrieve_and_format, list_all_papers
from agent import setup_groq_client, execute_rag_query_streaming, route_query, format_sources_display
from arxiv_tools import search_arxiv, download_paper, format_arxiv_results
# from evaluation import load_test_dataset, evaluate_retrieval, evaluate_answer_quality, generate_evaluation_report

# Load environment
load_dotenv()

# Page config
st.set_page_config(
    page_title="Research Paper Q&A Agent",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'papers_ingested' not in st.session_state:
    st.session_state.papers_ingested = []
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'qdrant_client' not in st.session_state:
    st.session_state.qdrant_client = None
if 'groq_client' not in st.session_state:
    st.session_state.groq_client = None


def initialize_clients():
    """Initialize all clients and models"""
    try:
        # Embedding model
        if st.session_state.embedding_model is None:
            with st.spinner("Loading embedding model..."):
                st.session_state.embedding_model = initialize_embedding_model()
        
        # Qdrant client
        if st.session_state.qdrant_client is None:
            qdrant_url = os.getenv('QDRANT_URL', 'http://localhost:6333')
            st.session_state.qdrant_client = setup_qdrant_client(qdrant_url)
        
        # Groq client
        if st.session_state.groq_client is None:
            groq_api_key = os.getenv('GROQ_API_KEY')
            if not groq_api_key:
                st.error("⚠️ GROQ_API_KEY not found in environment variables")
                return False
            st.session_state.groq_client = setup_groq_client(groq_api_key)
        
        return True
    except Exception as e:
        st.error(f"❌ Initialization error: {str(e)}")
        return False


def ingest_pdf(uploaded_file, collection_name: str):
    """Process and ingest a single PDF"""
    try:
        # Save uploaded file
        upload_dir = Path("./data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / uploaded_file.name
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        # Process PDF
        with st.spinner(f"Processing {uploaded_file.name}..."):
            pdf_content, chunks = process_pdf_file(str(file_path))
        
        # Generate embeddings
        with st.spinner("Generating embeddings..."):
            embeddings = generate_embeddings(chunks, st.session_state.embedding_model)
        
        # Check if collection exists, create if not
        try:
            st.session_state.qdrant_client.get_collection(collection_name)
        except:
            vector_size = st.session_state.embedding_model.get_sentence_embedding_dimension()
            setup_qdrant_collection(st.session_state.qdrant_client, collection_name, vector_size)
        
        # Store in Qdrant
        with st.spinner("Storing in vector database..."):
            store_in_qdrant(chunks, embeddings, st.session_state.qdrant_client, collection_name)
        
        return pdf_content['title'], len(chunks)
    
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        return None, 0


# Main UI
st.title("📄 Research Paper Q&A Agent")
st.markdown("*Powered by Groq GPT oss 120b and Qdrant*")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    

    qdrant_url = st.text_input("Qdrant URL", value=os.getenv('QDRANT_URL', 'http://localhost:6333'))
    collection_name = st.text_input("Collection Name", value="research_papers")
    
    st.divider()
    
    # Document upload
    st.header("📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload research papers in PDF format"
    )
    
    if st.button("🔄 Ingest Documents", disabled=not uploaded_files):
        if initialize_clients():
            progress_bar = st.progress(0)
            for i, uploaded_file in enumerate(uploaded_files):
                title, num_chunks = ingest_pdf(uploaded_file, collection_name)
                if title:
                    st.session_state.papers_ingested.append({
                        'title': title,
                        'chunks': num_chunks
                    })
                    st.success(f"✓ {title} ({num_chunks} chunks)")
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            st.balloons()
    
    st.divider()
    
    # Ingested papers
    st.header("📚 Ingested Papers")
    if st.session_state.papers_ingested:
        for paper in st.session_state.papers_ingested:
            st.write(f"• {paper['title']}")
            st.caption(f"  {paper['chunks']} chunks")
    else:
        st.info("No papers ingested yet")
    
    # Clear database
    if st.button("🗑️ Clear Database"):
        if st.session_state.qdrant_client:
            try:
                st.session_state.qdrant_client.delete_collection(collection_name)
                st.session_state.papers_ingested = []
                st.success("✓ Database cleared")
            except:
                st.info("Collection doesn't exist")

# Main content area
tab1, tab2, tab3 = st.tabs(["💬 Q&A", "🔍 Arxiv Search", "📊 Evaluation"])

with tab1:
    st.header("Ask Questions About Papers")
    
    # Example questions
    st.subheader("Example Questions:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📖 What is the main contribution?"):
            st.session_state.example_query = "What is the main contribution of the papers?"
    
    with col2:
        if st.button("📊 Extract metrics"):
            st.session_state.example_query = "What accuracy metrics are reported?"
    
    with col3:
        if st.button("📝 Summarize methodology"):
            st.session_state.example_query = "Summarize the methodology used"
    
    # Query input
    query = st.text_input(
        "Your Question:",
        value=st.session_state.get('example_query', ''),
        placeholder="Ask anything about the uploaded papers..."
    )
    
    if query and st.button("🔍 Search", type="primary"):
        if not initialize_clients():
            st.stop()
        
        if not st.session_state.papers_ingested:
            st.warning("⚠️ Please upload and ingest papers first")
            st.stop()
        
        # Retrieve context
        with st.spinner("Retrieving relevant context..."):
            try:
                context, sources = retrieve_and_format(
                    query,
                    st.session_state.qdrant_client,
                    collection_name,
                    st.session_state.embedding_model,
                    k=5
                )
            except Exception as e:
                st.error(f"Retrieval error: {str(e)}")
                st.stop()
        
        # Display query type
        query_type = route_query(query)
        st.info(f"🏷️ Query Type: **{query_type}**")
        
        # Generate answer with streaming
        st.subheader("💡 Answer:")
        answer_placeholder = st.empty()
        answer_text = ""
        
        try:
            for chunk in execute_rag_query_streaming(
                query,
                context,
                st.session_state.groq_client
            ):
                answer_text += chunk
                answer_placeholder.markdown(answer_text)
        except Exception as e:
            st.error(f"Generation error: {str(e)}")
        
        # Display sources
        with st.expander("📚 View Sources", expanded=False):
            st.markdown(format_sources_display(sources))

with tab2:
    st.header("🔍 Search Arxiv Papers")
    st.markdown("Find and download papers from Arxiv")
    
    arxiv_query = st.text_input("Search Query:", placeholder="e.g., transformer architecture, BERT")
    
    if st.button("🔍 Search Arxiv"):
        if arxiv_query:
            with st.spinner("Searching Arxiv..."):
                papers = search_arxiv(arxiv_query, max_results=5)
            
            if papers:
                st.success(f"Found {len(papers)} papers")
                
                for i, paper in enumerate(papers):
                    with st.expander(f"📄 {paper['title']}", expanded=(i==0)):
                        st.write(f"**Authors:** {', '.join(paper['authors'][:3])}")
                        if len(paper['authors']) > 3:
                            st.write("et al.")
                        st.write(f"**Published:** {paper['published']}")
                        st.write(f"**Arxiv ID:** {paper['arxiv_id']}")
                        st.write(f"**Abstract:** {paper['abstract'][:300]}...")
                        
                        if st.button(f"⬇️ Download & Ingest", key=f"download_{i}"):
                            if initialize_clients():
                                with st.spinner("Downloading paper..."):
                                    try:
                                        file_path = download_paper(paper['arxiv_id'])
                                        st.success(f"✓ Downloaded to {file_path}")
                                        
                                        # Auto-ingest
                                        with st.spinner("Ingesting paper..."):
                                            pdf_content, chunks = process_pdf_file(file_path)
                                            embeddings = generate_embeddings(chunks, st.session_state.embedding_model)
                                            
                                            try:
                                                st.session_state.qdrant_client.get_collection(collection_name)
                                            except:
                                                vector_size = st.session_state.embedding_model.get_sentence_embedding_dimension()
                                                setup_qdrant_collection(st.session_state.qdrant_client, collection_name, vector_size)
                                            
                                            store_in_qdrant(chunks, embeddings, st.session_state.qdrant_client, collection_name)
                                            
                                            st.session_state.papers_ingested.append({
                                                'title': pdf_content['title'],
                                                'chunks': len(chunks)
                                            })
                                            st.success(f"✓ Paper ingested successfully!")
                                    except Exception as e:
                                        st.error(f"Error: {str(e)}")
            else:
                st.warning("No papers found")

with tab3:
    st.header("📊 Evaluation")
    st.markdown("Run evaluation on test dataset")
    
    eval_file = st.file_uploader("Upload Evaluation Dataset (JSON)", type=['json'])
    
    if st.button("▶️ Run Evaluation"):
        if not eval_file:
            st.warning("Please upload an evaluation dataset")
        elif not initialize_clients():
            st.stop()
        else:
            # Save uploaded file
            eval_path = "./data/evaluation_dataset.json"
            with open(eval_path, 'wb') as f:
                f.write(eval_file.getbuffer())
            
            # Load test cases
            test_cases = load_test_dataset(eval_path)
            
            if not test_cases:
                st.error("No test cases found in dataset")
                st.stop()
            
            st.info(f"Running evaluation on {len(test_cases)} test cases...")
            
            # Run evaluation
            progress = st.progress(0)
            
            # TODO: Implement full evaluation
            st.warning("⚠️ Full evaluation implementation in progress")
            st.info("This will evaluate retrieval and answer quality metrics")
            
            progress.progress(100)

# Footer
st.divider()
st.caption("Built with Streamlit, Groq, Qdrant, and LangChain")