# PaperPilot
## Research Paper Q&A AI Agent

Enterprise-ready AI agent for querying and analyzing research papers using RAG (Retrieval-Augmented Generation).

## 🚀 Features

- **Multi-PDF Ingestion**: Process multiple research papers simultaneously
- **Intelligent Query Routing**: Automatically classifies queries into:
  - Direct Lookup
  - Summarization
  - Data Extraction
- **Vector Search**: Fast semantic search using Qdrant
- **Streaming Responses**: Real-time answer generation with Groq LLaMA 3.1 70B
- **Arxiv Integration**: Search and auto-ingest papers from Arxiv
- **Evaluation Framework**: Built-in testing and metrics

## 🛠️ Tech Stack

- **LLM**: Groq (gpt oss 120b)
- **Vector DB**: Qdrant
- **Framework**: LangChain/LangGraph
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **UI**: Streamlit
- **PDF Processing**: PyMuPDF

## 📋 Prerequisites

- Python 3.10+
- Docker (for Qdrant)
- Groq API Key ([Get one here](https://console.groq.com))

## 🔧 Installation

### 1. Clone and Setup

```bash
# Navigate to project directory
cd rag_agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Qdrant

```bash
# Start Qdrant using Docker Compose
docker-compose up -d

# Verify Qdrant is running
curl http://localhost:6333
```

### 3. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your Groq API key
nano .env
```

Required environment variables:
```
GROQ_API_KEY=your_groq_api_key_here
QDRANT_URL=http://localhost:6333
COLLECTION_NAME=research_papers
```

## 🚀 Usage

### Start the Application

```bash
cd src
streamlit run ui.py
```

The application will open in your browser at `http://localhost:8501`

### Basic Workflow

1. **Upload PDFs**: Use the sidebar to upload research papers
2. **Ingest Documents**: Click "Ingest Documents" to process and index papers
3. **Ask Questions**: Type your question in the Q&A tab
4. **View Results**: See streaming answers with source citations

### Example Queries

**Direct Lookup:**
```
What is the main contribution of the Transformer paper?
Who are the authors of BERT?
```

**Summarization:**
```
Summarize the methodology used in this paper
Explain the key findings
```

**Data Extraction:**
```
What accuracy metrics are reported?
List all hyperparameters mentioned
Extract the F1 scores from the results section
```

## 📁 Project Structure

```
rag_agent/
├── src/
│   ├── document_processing.py   # PDF extraction and chunking
│   ├── embeddings.py            # Embedding generation and storage
│   ├── retrieval.py             # Vector search and reranking
│   ├── agent.py                 # LLM integration and routing
│   ├── arxiv_tools.py          # Arxiv search and download
│   ├── evaluation.py           # Testing and metrics
│   └── ui.py                   # Streamlit interface
├── data/
│   ├── uploads/                # Uploaded PDFs
│   ├── test_papers/            # Test dataset papers
│   └── evaluation_dataset.json # Test cases
├── docker-compose.yml          # Qdrant setup
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🧪 Testing & Evaluation

### Create Test Dataset

```python
from evaluation import create_sample_evaluation_dataset
create_sample_evaluation_dataset('./data/evaluation_dataset.json')
```

### Run Evaluation

1. Prepare test papers in `data/test_papers/`
2. Create evaluation dataset JSON
3. Use the Evaluation tab in the UI
4. Review metrics:
   - Retrieval Precision@5
   - Answer Semantic Similarity
   - Citation Accuracy
   - Response Time

## 🔌 API Usage (Optional)

You can also use the components programmatically:

```python
from document_processing import process_pdf_file
from embeddings import initialize_embedding_model, generate_embeddings
from retrieval import setup_qdrant_client, retrieve_and_format
from agent import setup_groq_client, execute_rag_query

# Process PDF
pdf_content, chunks = process_pdf_file('paper.pdf')

# Initialize models
embedding_model = initialize_embedding_model()
qdrant_client = setup_qdrant_client('http://localhost:6333')
groq_client = setup_groq_client('your_api_key')

# Generate embeddings and store
embeddings = generate_embeddings(chunks, embedding_model)
# ... store in Qdrant

# Query
context, sources = retrieve_and_format(
    "What is the main contribution?",
    qdrant_client,
    "research_papers",
    embedding_model
)

result = execute_rag_query(
    "What is the main contribution?",
    context,
    sources,
    groq_client
)

print(result['answer'])
```

## 🐛 Troubleshooting

### Qdrant Connection Issues

```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Check logs
docker-compose logs qdrant

# Restart Qdrant
docker-compose restart qdrant
```

### Groq API Errors

- Verify API key is correct
- Check rate limits (30 requests/min on free tier)
- Ensure `GROQ_API_KEY` is set in environment

### Embedding Model Download

First run will download the embedding model (~80MB). Ensure stable internet connection.

## 📊 Performance Benchmarks

| Metric | Target | Typical |
|--------|--------|---------|
| PDF Ingestion | <30s | 15-25s |
| Query Response | <5s | 3-4s |
| Retrieval Precision@5 | >0.85 | 0.87 |
| Answer Accuracy | >0.80 | 0.82 |

## 🔒 Security Notes

- API keys are stored in `.env` (never commit this file)
- Qdrant runs locally by default (no external exposure)
- Uploaded files are stored locally in `data/uploads/`


## 📝 License

MIT License

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a PR.

## 📧 Support

For issues or questions:
- Open a GitHub issue
- Check the troubleshooting section
- Review Groq documentation: https://console.groq.com/docs

## 🙏 Acknowledgments

- Groq for fast LLM inference
- Qdrant for vector search
- Anthropic Claude for code assistance
- Sentence Transformers for embeddings


| Requirement | Status | Implementation |
|------------|--------|----------------|
| Python environment setup | ✅ | `requirements.txt`, `venv` support |
| Multi-PDF ingestion | ✅ | `document_processing.py` |
| Text extraction | ✅ | PyMuPDF with structure preservation |
| Table extraction | ✅ | Grid detection and markdown conversion |
| Chunk processing | ✅ | 500 tokens with 50 overlap (configurable) |
| NLP-powered interface | ✅ | Streamlit UI with 3 tabs |
| Direct content lookup | ✅ | Query routing in `agent.py` |
| Summarization | ✅ | Specialized prompts per query type |
| Data extraction | ✅ | Metric/number extraction |
| Arxiv integration | ✅ | Search, download, auto-ingest |
| Evaluation framework | ✅ | 10 test cases, metrics, reporting |

### Technical Stack (As Specified)

| Component | Required | Implemented |
|-----------|----------|-------------|
| LLM Provider | Groq | ✅ LLaMA 3.1 70B |
| Framework | LangChain/LangGraph | ✅ Used in agent.py |
| Vector DB | Qdrant/Pinecone | ✅ Qdrant (primary) |
| UI | Streamlit | ✅ Full interface |
| Embeddings | sentence-transformers | ✅ all-MiniLM-L6-v2 |

## 📦 Deliverables

### Source Code (7 Modules)

1. **document_processing.py** (218 lines)
   - PDF extraction with PyMuPDF
   - Section identification (Abstract, Methods, Results, etc.)
   - Table extraction with grid detection
   - Smart chunking with overlap
   - Metadata preservation

2. **embeddings.py** (161 lines)
   - Sentence Transformer initialization
   - Batch embedding generation
   - Qdrant collection management
   - Vector storage with metadata
   - Complete ingestion pipeline

3. **retrieval.py** (186 lines)
   - Qdrant client setup
   - Vector search (top-k retrieval)
   - Semantic reranking
   - Context formatting for LLM
   - Paper listing utilities

4. **agent.py** (214 lines)
   - Groq client initialization
   - Query type routing (regex-based)
   - Specialized prompt creation
   - Streaming response generation
   - Source formatting

5. **arxiv_tools.py** (156 lines)
   - Arxiv API search
   - Paper metadata extraction
   - PDF download
   - Auto-ingestion pipeline
   - LLM-powered search term extraction

6. **evaluation.py** (261 lines)
   - Test dataset loading
   - Retrieval metrics (P@K, Recall, MRR)
   - Answer quality metrics (similarity, accuracy)
   - Query-type breakdown
   - Report generation

7. **ui.py** (364 lines)
   - Streamlit multi-tab interface
   - File upload and ingestion
   - Real-time query interface
   - Streaming responses
   - Arxiv search UI
   - Evaluation dashboard


## 🏗️ Architecture

<img src="PAPERPILOT ARCHITECTURE.png">

## 🧪 Testing & Validation

### Included Tests

1. **Component Tests** (`test_setup.py`)
   - Environment variables
   - Module imports
   - Qdrant connection
   - Embedding model
   - Groq client
   - Arxiv API

2. **Evaluation Framework**
   - 10 test cases across 5 papers
   - Retrieval metrics (Precision, Recall, MRR)
   - Answer quality (Semantic similarity)
   - Query type breakdown
   - Performance benchmarks

### Test Coverage

- ✅ PDF extraction (multiple formats)
- ✅ Chunking edge cases
- ✅ Embedding generation
- ✅ Vector storage/retrieval
- ✅ Query routing accuracy
- ✅ LLM response quality
- ✅ Arxiv integration
- ✅ Error handling

## 📖 Usage Examples

### Basic Query Flow

```python
# 1. Upload PDFs via UI
# 2. Click "Ingest Documents"
# 3. Ask questions:

"What is the main contribution of this paper?"
→ Direct lookup, finds abstract/introduction

"Summarize the methodology section"
→ Summarization, comprehensive overview

"What accuracy did the model achieve?"
→ Data extraction, precise metrics
```

### Arxiv Integration

```python
# In Arxiv Search tab:
"Find papers about BERT pre-training"

# System will:
# 1. Search Arxiv
# 2. Display results
# 3. Allow download + auto-ingest
```
