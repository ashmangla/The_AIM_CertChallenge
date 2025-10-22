"""Tools for RAG-based retrieval and web search.

This module provides two main tools:
1. `retrieve_information`: RAG-based retrieval from appliance manuals
2. `tavily_tool`: Web search using Tavily API

Call `initialize_tools(data_directory)` before using the tools.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Annotated, List

import tiktoken

# ============================================================================
# CONFIGURATION - API Keys (loaded from environment variables)
# ============================================================================
# Load API keys from environment variables (.env file)
# For local development: Copy api/.env.example to api/.env and fill in your keys
# For production: Set these as environment variables in your deployment platform
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
# ============================================================================
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict


logger = logging.getLogger(__name__)

# Global state for initialized components
_compiled_rag_graph = None
_is_initialized = False


# RAG prompt template
HUMAN_TEMPLATE = """
#CONTEXT:
{context}

QUERY:
{query}

Use the provided context to answer the provided user query. Only use the provided context to answer the query.

IMPORTANT: 
- Always cite the page numbers from the manual where you found the information
- Format page citations as: [Source: Manual, Page X] or [Source: Manual, Pages X-Y]
- If you do not know the answer, or it's not contained in the provided context respond with "I don't know"
"""


class State(TypedDict):
    """State for the RAG graph."""
    question: str
    context: List[Document]
    response: str


def tiktoken_len(text: str) -> int:
    """Calculate token length using tiktoken for gpt-4o.
    
    Args:
        text: Input text to tokenize
        
    Returns:
        Number of tokens in the text
    """
    tokens = tiktoken.encoding_for_model("gpt-4o").encode(text)
    return len(tokens)


def initialize_tools(data_directory: str = "data", force_reinit: bool = False) -> None:
    """Initialize the RAG pipeline and tools.
    
    This function loads documents from the specified directory, creates embeddings,
    builds a vector store, and compiles the RAG graph. It automatically uses the
    best configuration from the evaluation results (stored in config/retrieval_config.json).
    
    Args:
        data_directory: Path to directory containing PDF documents
        force_reinit: If True, re-initialize even if already initialized (for dynamic re-indexing)
        
    Raises:
        FileNotFoundError: If the data directory doesn't exist
        ValueError: If no documents are found in the directory
    """
    global _compiled_rag_graph, _is_initialized
    
    if _is_initialized and not force_reinit:
        logger.warning("Tools already initialized. Skipping re-initialization.")
        return
    
    # Set API keys if provided in configuration
    if OPENAI_API_KEY and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        logger.info("Using OpenAI API key from configuration")
    
    if COHERE_API_KEY and not os.getenv("COHERE_API_KEY"):
        os.environ["COHERE_API_KEY"] = COHERE_API_KEY
        logger.info("Using Cohere API key from configuration")
    
    logger.info(f"🔧 Initializing tools from directory: {data_directory}")
    
    # Load best configuration from evaluation results
    import json
    from pathlib import Path
    config_path = Path(__file__).parent.parent / "config" / "retrieval_config.json"
    
    chunking_strategy = "recursive"  # Default fallback
    use_rerank = False
    use_multi_query = False
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
            best_method = config.get("best_retriever", "")
            logger.info(f"📊 Loading best config from evaluation: {best_method}")
            logger.info(f"   Average score: {config.get('average_score', 0):.3f}")
            
            # Parse the method name to extract strategy
            if "Semantic" in best_method:
                chunking_strategy = "semantic"
            if "Cohere Rerank" in best_method:
                use_rerank = True
            if "Multi-Query" in best_method:
                use_multi_query = True
    else:
        logger.warning(f"⚠️  Config file not found at {config_path}. Using default: Recursive chunking")
    
    logger.info(f"   Chunking: {'Semantic' if chunking_strategy == 'semantic' else 'Recursive'}")
    logger.info(f"   Reranking: {'Cohere Rerank' if use_rerank else 'None'}")
    logger.info(f"   Multi-Query: {'Yes' if use_multi_query else 'No'}")
    
    # Load documents
    directory_loader = DirectoryLoader(
        data_directory,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader
    )
    appliance_manuals = directory_loader.load()
    logger.info(f"✅ Loaded {len(appliance_manuals)} documents")
    
    if not appliance_manuals:
        raise ValueError(f"No documents found in {data_directory}")
    
    # Create embeddings model
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Use chunking strategy from config
    if chunking_strategy == "semantic":
        logger.info("🔍 Using SemanticChunker (percentile breakpoint)...")
        text_splitter = SemanticChunker(
            embedding_model,
            breakpoint_threshold_type="percentile"
        )
    else:  # recursive (default)
        logger.info("📝 Using RecursiveCharacterTextSplitter...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=750,
            chunk_overlap=50,
            length_function=tiktoken_len,
        )
    
    appliance_chunks = text_splitter.split_documents(appliance_manuals)
    logger.info(f"✅ Split documents into {len(appliance_chunks)} chunks")
    
    # Create vector store
    qdrant_vectorstore = Qdrant.from_documents(
        documents=appliance_chunks,
        embedding=embedding_model,
        location=":memory:"
    )
    logger.info("✅ Created in-memory vector store")
    
    # Create retriever based on config
    if use_rerank:
        # Use Cohere Rerank for better relevance
        logger.info("🎯 Applying Cohere Rerank...")
        base_retriever = qdrant_vectorstore.as_retriever(search_kwargs={"k": 5})
        compressor = CohereRerank(model="rerank-english-v3.0", top_n=3)
        qdrant_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        logger.info("✅ Retriever ready with Cohere Rerank")
    elif use_multi_query:
        # Use Multi-Query retrieval
        logger.info("🔄 Setting up Multi-Query retrieval...")
        from langchain.retrievers.multi_query import MultiQueryRetriever
        base_retriever = qdrant_vectorstore.as_retriever(search_kwargs={"k": 5})
        multi_query_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        qdrant_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=multi_query_llm
        )
        logger.info("✅ Retriever ready with Multi-Query")
    else:
        # Use standard retriever (naive)
        logger.info("📍 Using standard vector similarity retrieval...")
        qdrant_retriever = qdrant_vectorstore.as_retriever(search_kwargs={"k": 5})
        logger.info("✅ Retriever ready")
    
    # Create prompt template and LLM
    chat_prompt = ChatPromptTemplate.from_messages([
        ("human", HUMAN_TEMPLATE)
    ])
    generator_llm = ChatOpenAI(model="gpt-4.1-nano")
    
    # Define RAG graph nodes
    def retrieve(state: State) -> dict:
        """Retrieve relevant documents based on the question."""
        retrieved_docs = qdrant_retriever.invoke(state["question"])
        return {"context": retrieved_docs}
    
    def generate(state: State) -> dict:
        """Generate response using retrieved context with page numbers."""
        # Format context with page numbers
        formatted_context = []
        for doc in state["context"]:
            page_num = doc.metadata.get('page', 'Unknown')
            # Add 1 to page number since PyMuPDF uses 0-based indexing
            display_page = page_num + 1 if isinstance(page_num, int) else page_num
            formatted_context.append(f"[Page {display_page}]: {doc.page_content}")
        
        context_str = "\n\n".join(formatted_context)
        
        generator_chain = chat_prompt | generator_llm | StrOutputParser()
        response = generator_chain.invoke({
            "query": state["question"],
            "context": context_str
        })
        return {"response": response}
    
    # Build and compile RAG graph
    rag_graph = StateGraph(State).add_sequence([retrieve, generate])
    rag_graph.add_edge(START, "retrieve")
    _compiled_rag_graph = rag_graph.compile()
    
    _is_initialized = True
    logger.info("Tools initialized successfully")


def re_index_manuals(data_directory: str = "data") -> str:
    """Re-index all manuals in the data directory with the current best chunking strategy.
    
    This function is called after downloading a new manual to dynamically update
    the vector store without restarting the server. It uses the same chunking
    strategy (from config/retrieval_config.json) to maintain consistency.
    
    Args:
        data_directory: Path to directory containing PDF documents
        
    Returns:
        Status message indicating success or failure
    """
    try:
        logger.info("🔄 Re-indexing all manuals with winning chunking strategy...")
        initialize_tools(data_directory=data_directory, force_reinit=True)
        return "✅ Re-indexing complete! All manuals (including new ones) are now searchable."
    except Exception as e:
        logger.error(f"❌ Re-indexing failed: {e}")
        return f"❌ Re-indexing failed: {str(e)}"


@tool
def retrieve_information(
    query: Annotated[str, "Query to search in the appliance manuals"]
) -> dict:
    """Use Retrieval Augmented Generation to retrieve information from appliance manuals.
    
    This tool searches through loaded appliance manual documents using RAG
    to answer questions about appliance usage and maintenance.
    
    Args:
        query: The question to answer using the manual documents
        
    Returns:
        Dictionary containing the question, context, and generated response
        
    Raises:
        RuntimeError: If tools haven't been initialized via initialize_tools()
    """
    if not _is_initialized or _compiled_rag_graph is None:
        raise RuntimeError(
            "Tools not initialized. Call initialize_tools() before using this tool."
        )
    
    logger.info(f"Retrieving information for query: {query}")
    return _compiled_rag_graph.invoke({"question": query})


@tool
def tavily_tool(
    query: Annotated[str, "Search query for finding appliance manuals or troubleshooting information"]
) -> str:
    """Search the web for appliance manuals or troubleshooting information using Tavily.
    
    This tool has two modes:
    1. **Manual Search Mode**: If query contains appliance model/company, searches for PDF manual,
       downloads it, and triggers re-indexing for future RAG queries.
    2. **Web Search Mode**: For general troubleshooting, returns relevant web results with citations.
    
    Args:
        query: Search query (e.g., "GE refrigerator GNE27JSMSS manual" or "how to fix ice maker")
        
    Returns:
        String with search results, download status, or troubleshooting information with citations
    """
    import re
    import requests
    from pathlib import Path
    
    # Check if Tavily API key is available
    api_key = TAVILY_API_KEY or os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "⚠️ Tavily search is not available. Please configure TAVILY_API_KEY."
    
    # Initialize Tavily search
    os.environ["TAVILY_API_KEY"] = api_key
    search_tool = TavilySearchResults(max_results=5)
    
    logger.info(f"🔍 Tavily search for: {query}")
    
    # Determine if this is a manual search query (contains "manual" or model patterns)
    is_manual_search = bool(
        re.search(r'\bmanual\b', query, re.IGNORECASE) or
        re.search(r'\b[A-Z]{2,}\s*[-\s]?\d+\w+', query)  # Model number pattern
    )
    
    try:
        # Perform Tavily search
        search_results = search_tool.invoke({"query": query})
        
        if not search_results:
            return "❌ No results found. Please try a different search query."
        
        # Mode 1: Manual Download (if manual-related query)
        if is_manual_search:
            logger.info("📚 Manual search mode activated")
            
            # Look for PDF links in results
            pdf_links = []
            for result in search_results:
                url = result.get('url', '')
                if url.lower().endswith('.pdf'):
                    pdf_links.append({
                        'url': url,
                        'title': result.get('content', 'Manual')[:100]
                    })
            
            if pdf_links:
                # Try to download the first PDF
                pdf_url = pdf_links[0]['url']
                logger.info(f"📥 Attempting to download manual from: {pdf_url}")
                
                try:
                    response = requests.get(pdf_url, timeout=30, stream=True)
                    response.raise_for_status()
                    
                    # Save to data directory
                    data_dir = Path(__file__).parent.parent / "data"
                    data_dir.mkdir(exist_ok=True)
                    
                    # Create filename from URL or use timestamp
                    filename = pdf_url.split('/')[-1]
                    if not filename.endswith('.pdf'):
                        filename = f"manual_{int(time.time())}.pdf"
                    
                    filepath = data_dir / filename
                    
                    # Download PDF
                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    logger.info(f"✅ Manual downloaded successfully: {filepath}")
                    
                    # Automatically re-index all manuals with the same chunking strategy
                    # This maintains consistency - all manuals use the winning strategy
                    # (Semantic Chunking + Cohere Rerank from config/retrieval_config.json)
                    logger.info("🔄 Starting automatic re-indexing...")
                    reindex_result = re_index_manuals(str(data_dir))
                    
                    return f"""✅ **Manual Downloaded & Indexed Successfully!**

📄 **File**: {filename}
📥 **Saved to**: `{filepath}`
🔗 **Source**: {pdf_url}

🔄 **Auto-Indexing**: {reindex_result}

✅ **Ready to Use!** You can now ask questions about this manual using retrieve_information.
No restart needed - the manual is immediately searchable with the same optimized chunking strategy (Semantic Chunking + Cohere Rerank) as your existing manuals.
"""
                
                except requests.exceptions.RequestException as e:
                    logger.error(f"❌ Failed to download manual: {e}")
                    return f"""⚠️ **Manual Found But Download Failed**

🔗 **Manual URL**: {pdf_url}
❌ **Error**: {str(e)}

**Please download manually from the link above and place it in the `api/data/` directory.**
"""
            
            else:
                # No PDF found, switch to Mode 2 (Web Search) to answer the question
                logger.info("📄 No PDF manual found, switching to Mode 2: Web search")
                
                result_text = "⚠️ **Manual Not Found** - I couldn't find a downloadable PDF manual for this appliance.\n\n"
                result_text += "🔍 **Searching the web for troubleshooting information instead:**\n\n"
                
                # Provide web search results with citations (Mode 2 behavior)
                for i, result in enumerate(search_results[:3], 1):
                    url = result.get('url', 'Unknown source')
                    content = result.get('content', 'No description')
                    
                    result_text += f"{i}. **Source**: [{url}]({url})\n"
                    result_text += f"   {content[:300]}...\n\n"
                
                result_text += "\n💡 **Note**: Since I don't have the official manual, these results are from general web sources. For precise model-specific instructions, please provide the exact model number so I can search for the official manual."
                
                return result_text
        
        # Mode 2: Web Search (for troubleshooting questions)
        else:
            logger.info("🌐 Web search mode activated (troubleshooting)")
            
            result_text = "🔧 **Troubleshooting Information**:\n\n"
            for i, result in enumerate(search_results[:3], 1):
                url = result.get('url', 'Unknown source')
                content = result.get('content', 'No description')
                
                result_text += f"{i}. **Source**: [{url}]({url})\n"
                result_text += f"   {content[:300]}...\n\n"
            
            result_text += "\n📌 **Note**: This information is sourced from the web. For appliance-specific guidance, please provide your model number so I can search for the official manual."
            
            return result_text
    
    except Exception as e:
        logger.error(f"❌ Tavily search error: {e}")
        return f"❌ Search failed: {str(e)}"


__all__ = ["initialize_tools", "re_index_manuals", "retrieve_information", "tavily_tool"]




