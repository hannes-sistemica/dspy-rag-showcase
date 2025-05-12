# DSPy RAG Showcase

A demonstration of DSPy's powerful RAG (Retrieval-Augmented Generation) capabilities using local LLMs and vector search. This project showcases how to build a sophisticated question-answering system that combines semantic search with structured prompting, all running locally without cloud dependencies.

## 🎯 Key Features
- **Local-first architecture**: Runs entirely on your machine using Ollama and ChromaDB
- **Advanced DSPy patterns**: Demonstrates Chain-of-Thought reasoning, query categorization, and structured outputs  
- **Vector semantic search**: Uses embeddings to find relevant documents from a local knowledge base
- **Clean, production-ready code**: Includes proper error handling, logging suppression, and interactive UI
- **Multiple retrieval strategies**: Category filtering, similarity search, and batch processing

## 🛠️ Technologies
- **DSPy**: For structured prompting and modular AI components
- **ChromaDB**: Local vector database for semantic search
- **Ollama + Llama 3.2**: Fully local LLM inference
- **UV**: Modern Python package management

## 🚀 What It Shows
This project demonstrates DSPy's ability to:
- Create modular, reusable AI components (signatures, modules)
- Handle complex retrieval patterns with confidence scoring
- Build sophisticated RAG pipelines without external APIs
- Structure LLM outputs with specific fields (answer, confidence, sources)
- Implement different query strategies based on question type

Perfect for developers wanting to understand DSPy's RAG capabilities or build local AI applications without cloud dependencies. The clean architecture makes it easy to extend with your own documents, models, or retrieval strategies.

**Note**: All processing happens locally - despite the LLM sometimes claiming to use "Wikipedia" as a source, it only searches through 5 pre-loaded sample documents.

What's really happening:
1. **Local Document Storage**: The system stores 5 pre-defined documents in ChromaDB with vector embeddings
2. **Vector Search**: When you ask a question, it searches through these 5 documents using semantic similarity
3. **Context Retrieval**: It retrieves the most relevant documents and passes them to the LLM
4. **Answer Generation**: The LLM generates an answer based on the retrieved context
5. **Source Fabrication**: The LLM sometimes invents sources (like "Wikipedia") that aren't actually in the documents

## Setup Instructions

### Prerequisites

- Python 3.11+ 
- [uv](https://github.com/astral-sh/uv) package manager
- [Ollama](https://ollama.ai) running locally with the `llama3.2:3b` model

### Installation with UV

1. **Install UV** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Create a new project**:
```bash
mkdir dspy-rag-showcase
cd dspy-rag-showcase
uv init
```

3. **Add dependencies** to your `pyproject.toml`:
```toml
[project]
name = "dspy-rag-showcase"
version = "0.1.0"
dependencies = [
    "dspy-ai>=2.0.0",
    "chromadb>=0.4.0",
    "litellm",
    "numpy",
    "httpx",
]
```

4. **Install dependencies**:
```bash
uv pip install -e .
```

5. **Install and start Ollama**:
```bash
# macOS
brew install ollama
ollama serve

# In another terminal, pull the model
ollama pull llama3.2:3b
```

6. **Run the main script** (`main.py`) with the DSPy RAG code

7. **Run the application**:
```bash
uv run main.py
```

## System Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────┐
│   User Query    │────▶│  Query Categorizer│────▶│ Vector Search  │
└─────────────────┘     └──────────────────┘     └────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌────────────────┐
│   LLM Answer    │◀────│  Context + Query │◀────│   ChromaDB     │
└─────────────────┘     └──────────────────┘     └────────────────┘
```

## Pre-loaded Documents

The system comes with 5 pre-loaded documents:

1. **France Capital** (Geography) - Information about Paris
2. **Shakespeare's Hamlet** (Literature) - Details about the play
3. **Mona Lisa** (Art) - Information about the painting
4. **Quantum Computing Basics** (Technology) - Introduction to quantum computing
5. **Human Genome** (Biology) - Information about the human genome project

## Features

### 1. Single Question Answering
```python
result = ask_question("Who wrote Hamlet?")
```

### 2. Batch Processing
```python
questions = ["What is the capital of France?", "Tell me about the Mona Lisa"]
results = batch_questions(questions)
```

### 3. Category Filtering
```python
# Only search in Literature documents
result = filtered_query("What plays were written?", category_filter="Literature")
```

### 4. Similar Document Search
```python
# Find documents similar to a query
similar = find_similar_documents("museums and art", n_results=2)
```

### 5. Category Summarization
```python
# Summarize all documents in a category
summary = summarize_category("Technology")
```

### 6. Interactive Mode
The system includes an interactive question-answering interface:
```
🤖 RAG System Ready! Ask questions (or 'quit' to exit):
❓ Your question: What is the Louvre famous for?
```

## Output Format

Each response includes:
- 📝 **Query**: The original question
- 📊 **Category**: AI-determined query category
- ✓ **Answer**: The generated response
- 🎯 **Confidence**: High/Medium/Low confidence rating
- 📚 **Sources**: Listed sources (note: may be fabricated by the LLM)

## Configuration

### Language Model
```python
lm = dspy.LM(
    model='ollama_chat/llama3.2:3b',
    api_base='http://localhost:11434',
    api_key='',  # No API key needed for local Ollama
    verbose=False
)
```

### ChromaDB Settings
```python
chroma_settings = chromadb.Settings(
    anonymized_telemetry=False,
    allow_reset=True
)
```

### Retrieval Settings
- **k=3**: Retrieves top 3 most relevant documents
- **embedding_function**: Uses ChromaDB's default embedding function

## Advanced Usage

### Adding New Documents
```python
new_doc = Document(
    id="6",
    text="Your document text here",
    title="Document Title",
    category="Category",
    timestamp=datetime.now().isoformat(),
    metadata={"key": "value"}
)

collection.add(
    documents=[new_doc.text],
    ids=[new_doc.id],
    metadatas=[{
        "title": new_doc.title,
        "category": new_doc.category,
        **new_doc.metadata
    }]
)
```

### Custom Query Processing
```python
# With metadata
result = ask_question("Your question", return_metadata=True)

# With debug info
result = ask_question("Your question", show_debug=True)
```

## Limitations

1. **No Real Web Access**: Despite what the LLM claims, the system cannot access Wikipedia or the internet
2. **Limited Knowledge Base**: Only searches through the 5 pre-loaded documents
3. **Source Hallucination**: The LLM may invent sources that don't exist in the actual documents
4. **Local Processing**: Requires Ollama running locally with sufficient resources

## Troubleshooting

### ChromaDB Initialization Error
If you encounter "An instance of Chroma already exists" error:
```bash
rm -rf ./chroma_store
uv run main.py
```

### Ollama Connection Error
Ensure Ollama is running:
```bash
ollama serve
```

### Performance Issues
- Consider using a smaller model for faster responses
- Reduce the number of retrieved documents (k parameter)
- Disable logging for production use

## Project Structure
```
dspy-rag-showcase/
├── main.py              # Main application code
├── pyproject.toml       # Project dependencies
├── chroma_store/        # ChromaDB persistent storage
└── .venv/              # Virtual environment (created by uv)
```

## Contributing

Feel free to extend the system with:
- Additional document sources
- Web scraping capabilities
- Different embedding models
- Alternative LLMs
- Advanced ranking algorithms
- Real-time document updates

## License

This project is provided as an educational example for using DSPy with RAG systems.
