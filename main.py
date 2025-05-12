import dspy
import chromadb
import logging
import warnings
import os
from typing import List, Dict, Optional
from datetime import datetime
from chromadb.utils import embedding_functions
from dspy.retrieve.chromadb_rm import ChromadbRM
from dataclasses import dataclass

# Suppress all the noisy logging
os.environ['DSPY_VERBOSITY'] = '0'
warnings.filterwarnings('ignore')

# Disable ChromaDB telemetry
os.environ['CHROMA_TELEMETRY'] = 'false'

# Configure logging - only show errors and our custom messages
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Disable httpx logging
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

# Create a custom logger for our application
app_logger = logging.getLogger("app")
app_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
app_logger.handlers = [handler]
app_logger.propagate = False

# Configure the language model using dspy.LM
lm = dspy.LM(
    model='ollama_chat/llama3.2:3b',
    api_base='http://localhost:11434',
    api_key='',  # Leave empty if no API key is required
    verbose=False  # Disable verbose output
)
# Configure DSPy settings
dspy.configure(lm=lm, trace=[])

# Enhanced document structure
@dataclass
class Document:
    id: str
    text: str
    title: str
    category: str
    timestamp: str
    metadata: Dict[str, str]

# Sample corpus with more sophisticated documents
docs = [
    Document(
        id="1",
        text="Paris is the capital of France. It's known as the City of Light and is home to the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral.",
        title="France Capital",
        category="Geography",
        timestamp=datetime.now().isoformat(),
        metadata={"source": "Geography Encyclopedia", "confidence": "high"}
    ),
    Document(
        id="2",
        text="William Shakespeare wrote Hamlet around 1600-1601. The play is considered one of his greatest tragedies, featuring the famous soliloquy 'To be or not to be'.",
        title="Shakespeare's Hamlet",
        category="Literature",
        timestamp=datetime.now().isoformat(),
        metadata={"source": "Literary History", "year": "1600"}
    ),
    Document(
        id="3",
        text="The Mona Lisa, painted by Leonardo da Vinci, is housed in the Louvre Museum in Paris. It's one of the most valuable paintings in the world.",
        title="Mona Lisa",
        category="Art",
        timestamp=datetime.now().isoformat(),
        metadata={"artist": "Leonardo da Vinci", "location": "Louvre Museum"}
    ),
    Document(
        id="4",
        text="Quantum computing uses quantum bits or qubits, which can exist in multiple states simultaneously, unlike classical bits that are either 0 or 1.",
        title="Quantum Computing Basics",
        category="Technology",
        timestamp=datetime.now().isoformat(),
        metadata={"field": "Computer Science", "complexity": "advanced"}
    ),
    Document(
        id="5",
        text="The human genome consists of approximately 3 billion DNA base pairs. The Human Genome Project, completed in 2003, mapped the entire human genetic code.",
        title="Human Genome",
        category="Biology",
        timestamp=datetime.now().isoformat(),
        metadata={"project": "Human Genome Project", "completion_year": "2003"}
    )
]

# Set up the embedding function
embedding_function = embedding_functions.DefaultEmbeddingFunction()

# Utility function to reset ChromaDB if needed
def reset_chromadb(path: str = "./chroma_store"):
    """Reset ChromaDB instance if having initialization issues."""
    import shutil
    try:
        shutil.rmtree(path)
        app_logger.info("ChromaDB instance reset successfully")
    except Exception as e:
        app_logger.error(f"Failed to reset ChromaDB: {e}")

# Initialize the ChromaDB client with error handling
try:
    # Create settings that we'll use consistently
    chroma_settings = chromadb.Settings(
        anonymized_telemetry=False,
        allow_reset=True
    )
    
    # Try to get existing client or create new one
    try:
        # First try to connect to existing instance
        chroma_client = chromadb.PersistentClient(
            path="./chroma_store",
            settings=chroma_settings
        )
    except ValueError as e:
        if "already exists" in str(e):
            # If instance already exists, try with minimal settings
            chroma_client = chromadb.PersistentClient(path="./chroma_store")
        else:
            raise
    
    app_logger.info("ChromaDB client initialized successfully")
except Exception as e:
    app_logger.error(f"Failed to initialize ChromaDB client: {e}")
    # Optionally reset and retry
    # reset_chromadb()
    # chroma_client = chromadb.PersistentClient(path="./chroma_store", settings=chroma_settings)
    raise

# Create or get the collection with enhanced metadata
collection = chroma_client.get_or_create_collection(
    name="enhanced_rag_index",
    embedding_function=embedding_function,
    metadata={"description": "Enhanced RAG collection with multiple categories"}
)

# Add documents to the collection with metadata
collection.add(
    documents=[doc.text for doc in docs],
    ids=[doc.id for doc in docs],
    metadatas=[
        {
            "title": doc.title,
            "category": doc.category,
            "timestamp": doc.timestamp,
            **doc.metadata
        }
        for doc in docs
    ]
)

# Initialize the retriever with our existing client
retriever = ChromadbRM(
    collection_name="enhanced_rag_index",
    persist_directory="./chroma_store",
    embedding_function=embedding_function,
    client=chroma_client,  # Pass our existing client
    k=3  # Retrieve top 3 documents
)

# Enhanced task signatures
class EnhancedRAGSignature(dspy.Signature):
    """Answer using relevant context with metadata awareness."""
    context = dspy.InputField(desc="Relevant context for the question.")
    metadata = dspy.InputField(desc="Metadata about the retrieved documents.")
    question = dspy.InputField(desc="The question to be answered.")
    answer = dspy.OutputField(desc="Comprehensive answer to the question.")
    confidence = dspy.OutputField(desc="Confidence level of the answer (high/medium/low).")
    sources = dspy.OutputField(desc="List of sources used for the answer.")

class SummarySignature(dspy.Signature):
    """Summarize multiple documents."""
    documents = dspy.InputField(desc="List of documents to summarize.")
    summary = dspy.OutputField(desc="Concise summary of the documents.")
    key_points = dspy.OutputField(desc="Key points from the documents.")

# Initialize modules
rag_module = dspy.ChainOfThought(EnhancedRAGSignature)
summary_module = dspy.ChainOfThought(SummarySignature)

# Query categorizer
class QueryCategorizer(dspy.Signature):
    """Categorize the query to determine search strategy."""
    query = dspy.InputField(desc="User query to categorize.")
    category = dspy.OutputField(desc="Category of the query (e.g., factual, analytical, comparative).")
    keywords = dspy.OutputField(desc="Key terms to search for.")

categorizer_module = dspy.ChainOfThought(QueryCategorizer)

# Enhanced question-answering function
def ask_question(q: str, return_metadata: bool = False, show_debug: bool = False) -> Dict[str, any]:
    """Enhanced question answering with metadata and confidence scoring."""
    try:
        # Categorize the query first
        categorization = categorizer_module(query=q)
        if show_debug:
            app_logger.info(f"Query category: {categorization.category}")
        
        # Retrieve relevant passages
        retrieved_passages = retriever(q)
        
        # Extract text and metadata
        context = "\n".join([psg["long_text"] for psg in retrieved_passages])
        metadata = [psg.get("metadata", {}) for psg in retrieved_passages]
        metadata_str = "\n".join([f"Source: {m}" for m in metadata])
        
        # Get answer with enhanced RAG
        result = rag_module(
            question=q,
            context=context,
            metadata=metadata_str
        )
        
        response = {
            "answer": result.answer,
            "confidence": result.confidence,
            "sources": result.sources,
            "category": categorization.category
        }
        
        if return_metadata:
            response["retrieved_passages"] = retrieved_passages
            response["metadata"] = metadata
            
        return response
        
    except Exception as e:
        app_logger.error(f"Error processing question: {e}")
        return {
            "answer": "Unable to process the question.",
            "confidence": "low",
            "sources": [],
            "error": str(e)
        }

# Batch question processing
def batch_questions(questions: List[str], show_progress: bool = True) -> List[Dict[str, any]]:
    """Process multiple questions in batch."""
    results = []
    for i, question in enumerate(questions, 1):
        if show_progress:
            app_logger.info(f"Processing [{i}/{len(questions)}]: {question}")
        result = ask_question(question)
        results.append({
            "question": question,
            **result
        })
    return results

# Similarity search function
def find_similar_documents(query: str, n_results: int = 3) -> List[Dict[str, any]]:
    """Find documents similar to the query."""
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    similar_docs = []
    for i in range(len(results['ids'][0])):
        similar_docs.append({
            'id': results['ids'][0][i],
            'text': results['documents'][0][i],
            'metadata': results['metadatas'][0][i],
            'distance': results['distances'][0][i]
        })
    
    return similar_docs

# Advanced query with filtering
def filtered_query(question: str, category_filter: Optional[str] = None) -> Dict[str, any]:
    """Answer questions with optional category filtering."""
    if category_filter:
        # Use ChromaDB's filtering capability
        filtered_results = collection.query(
            query_texts=[question],
            n_results=3,
            where={"category": category_filter}
        )
        
        context = "\n".join(filtered_results['documents'][0])
        metadata_str = "\n".join([f"Source: {m}" for m in filtered_results['metadatas'][0]])
        
        result = rag_module(
            question=question,
            context=context,
            metadata=metadata_str
        )
        
        return {
            "answer": result.answer,
            "confidence": result.confidence,
            "sources": result.sources,
            "filter_applied": category_filter
        }
    else:
        return ask_question(question)

# Document summarization
def summarize_category(category: str) -> Dict[str, any]:
    """Summarize all documents in a specific category."""
    category_docs = collection.query(
        query_texts=[""],  # Empty query to get all
        n_results=100,
        where={"category": category}
    )
    
    if not category_docs['documents'][0]:
        return {"error": f"No documents found in category: {category}"}
    
    documents_text = "\n---\n".join(category_docs['documents'][0])
    
    summary_result = summary_module(documents=documents_text)
    
    return {
        "category": category,
        "summary": summary_result.summary,
        "key_points": summary_result.key_points,
        "document_count": len(category_docs['documents'][0])
    }

# Utility function to show cleaner output
def show_rag_call(question: str, response: Dict[str, any]):
    """Display RAG call and response in a clean format."""
    print(f"\n📝 Query: {question}")
    print(f"📊 Category: {response.get('category', 'Unknown')}")
    print(f"✓ Answer: {response['answer']}")
    print(f"🎯 Confidence: {response['confidence']}")
    if response.get('sources'):
        print(f"📚 Sources: {response['sources']}")
    print("-" * 50)

# Example usage with enhanced features
if __name__ == "__main__":
    app_logger.info("=== Enhanced DSPy RAG System ===")
    
    # Single question with full metadata
    app_logger.info("\n1. Single Question Example:")
    question = "Who wrote Hamlet and when?"
    result = ask_question(question, return_metadata=True)
    show_rag_call(question, result)
    
    # Batch processing - clean output
    app_logger.info("\n2. Batch Processing Example:")
    questions = [
        "What is the capital of France?",
        "Tell me about the Mona Lisa",
        "What is quantum computing?"
    ]
    batch_results = batch_questions(questions, show_progress=False)
    for r in batch_results:
        show_rag_call(r['question'], r)
    
    # Filtered query
    app_logger.info("\n3. Filtered Query Example (Literature only):")
    filtered_question = "What plays were written?"
    filtered_result = filtered_query(filtered_question, category_filter="Literature")
    show_rag_call(filtered_question, filtered_result)
    
    # Find similar documents
    app_logger.info("\n4. Find Similar Documents Example:")
    query = "museums and art"
    similar = find_similar_documents(query, n_results=2)
    print(f"\n🔍 Similar documents to: '{query}'")
    for i, doc in enumerate(similar, 1):
        print(f"\n  Document {i}:")
        print(f"  Title: {doc['metadata'].get('title', 'Unknown')}")
        print(f"  Category: {doc['metadata'].get('category', 'Unknown')}")
        print(f"  Distance: {doc['distance']:.4f}")
        print(f"  Preview: {doc['text'][:80]}...")
    
    # Category summarization
    app_logger.info("\n\n5. Category Summary Example:")
    category = "Technology"
    summary = summarize_category(category)
    print(f"\n📁 Category Summary: {category}")
    print(f"📄 Document Count: {summary.get('document_count', 0)}")
    print(f"📋 Summary: {summary.get('summary', 'No summary available')}")
    print(f"🔑 Key Points: {summary.get('key_points', 'No key points available')}")
    
    # Interactive mode example
    app_logger.info("\n\n6. Interactive Mode:")
    print("\n🤖 RAG System Ready! Ask questions (or 'quit' to exit):")
    
    while True:
        user_input = input("\n❓ Your question: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! 👋")
            break
        
        result = ask_question(user_input)
        show_rag_call(user_input, result)
