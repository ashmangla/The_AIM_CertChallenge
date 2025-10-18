"""Tools for RAG-based retrieval and web search.

This module provides two main tools:
1. `retrieve_information`: RAG-based retrieval from appliance manuals
2. `tavily_tool`: Web search using Tavily API

Call `initialize_tools(data_directory)` before using the tools.
"""

from __future__ import annotations

import logging
import os
from typing import Annotated, List

import tiktoken

# ============================================================================
# CONFIGURATION - Add your API keys here
# ============================================================================
OPENAI_API_KEY = ""  # Add your OpenAI API key here (or it will be passed from frontend)
TAVILY_API_KEY = ""  # Add your Tavily API key here
# ============================================================================
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
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

Use the provided context to answer the provided user query. Only use the provided context to answer the query. If you do not know the answer, or it's not contained in the provided context respond with "I don't know"
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


def initialize_tools(data_directory: str = "data") -> None:
    """Initialize the RAG pipeline and tools.
    
    This function loads documents from the specified directory, creates embeddings,
    builds a vector store, and compiles the RAG graph. Call this once at application
    startup before using the tools.
    
    Args:
        data_directory: Path to directory containing PDF documents
        
    Raises:
        FileNotFoundError: If the data directory doesn't exist
        ValueError: If no documents are found in the directory
    """
    global _compiled_rag_graph, _is_initialized
    
    if _is_initialized:
        logger.warning("Tools already initialized. Skipping re-initialization.")
        return
    
    # Set OpenAI API key if provided in configuration
    if OPENAI_API_KEY and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        logger.info("Using OpenAI API key from configuration")
    
    logger.info(f"Initializing tools from directory: {data_directory}")
    
    # Load documents
    directory_loader = DirectoryLoader(
        data_directory,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader
    )
    appliance_manuals = directory_loader.load()
    logger.info(f"Loaded {len(appliance_manuals)} documents")
    
    if not appliance_manuals:
        raise ValueError(f"No documents found in {data_directory}")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=0,
        length_function=tiktoken_len,
    )
    appliance_chunks = text_splitter.split_documents(appliance_manuals)
    logger.info(f"Split documents into {len(appliance_chunks)} chunks")
    
    # Create embeddings and vector store
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    qdrant_vectorstore = Qdrant.from_documents(
        documents=appliance_chunks,
        embedding=embedding_model,
        location=":memory:"
    )
    logger.info("Created in-memory vector store")
    
    qdrant_retriever = qdrant_vectorstore.as_retriever()
    
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
        """Generate response using retrieved context."""
        generator_chain = chat_prompt | generator_llm | StrOutputParser()
        response = generator_chain.invoke({
            "query": state["question"],
            "context": state["context"]
        })
        return {"response": response}
    
    # Build and compile RAG graph
    rag_graph = StateGraph(State).add_sequence([retrieve, generate])
    rag_graph.add_edge(START, "retrieve")
    _compiled_rag_graph = rag_graph.compile()
    
    _is_initialized = True
    logger.info("Tools initialized successfully")


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


# Initialize Tavily search tool with max 5 results (optional)
# Uses TAVILY_API_KEY from configuration section above or environment variable
try:
    # Check configuration variable first, then environment variable
    api_key = TAVILY_API_KEY or os.getenv("TAVILY_API_KEY")
    
    if api_key:
        # Set the environment variable so TavilySearchResults can find it
        os.environ["TAVILY_API_KEY"] = api_key
        tavily_tool = TavilySearchResults(max_results=5)
        logger.info("Tavily search tool initialized successfully")
    else:
        # Create a dummy tool if Tavily API key is not available
        @tool
        def tavily_tool(query: str) -> str:
            """Placeholder for Tavily search - API key not configured."""
            return "Tavily search is not available. Please add your API key to the TAVILY_API_KEY variable in tools/tools.py"
        logger.warning("Tavily API key not found. Tavily search tool will not be available.")
except Exception as e:
    logger.warning(f"Failed to initialize Tavily tool: {e}. Creating placeholder.")
    @tool
    def tavily_tool(query: str) -> str:
        """Placeholder for Tavily search - initialization failed."""
        return f"Tavily search is not available: {str(e)}"


__all__ = ["initialize_tools", "retrieve_information", "tavily_tool"]




