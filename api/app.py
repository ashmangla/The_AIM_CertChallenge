# Import required FastAPI components for building the API
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
# Import Pydantic for data validation and settings management
from pydantic import BaseModel, HttpUrl
# Import OpenAI client for interacting with OpenAI's API
from openai import OpenAI
import os
import sys
import tempfile
import asyncio
from pathlib import Path
from typing import Optional, List
import logging
import traceback

# Add the parent directory to sys.path to import aimakerspace
sys.path.append(str(Path(__file__).parent.parent))
from aimakerspace.text_utils import PDFLoader, CharacterTextSplitter
from aimakerspace.vectordatabase import VectorDatabase
from aimakerspace.openai_utils.embedding import EmbeddingModel
from aimakerspace.openai_utils.chatmodel import ChatOpenAI
from tools.tools import initialize_tools, retrieve_information, tavily_tool
from agents.rag_agent import create_rag_agent
from langchain_core.messages import HumanMessage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI application with a title
app = FastAPI(title="RAG-Enabled Chat API")


@app.on_event("startup")
async def startup_event():
    """Initialize tools and agent on application startup."""
    global rag_agent
    
    try:
        # Check if data directory exists
        data_dir = Path(__file__).parent / "data"
        if data_dir.exists():
            logger.info("Initializing RAG tools from data directory...")
            initialize_tools(str(data_dir))
            logger.info("Tools initialized successfully")
            
            # Create the RAG agent with tools
            logger.info("Creating RAG agent...")
            rag_agent = create_rag_agent(model_name="gpt-4o-mini", temperature=0.0)
            logger.info("RAG agent created successfully")
        else:
            logger.warning(f"Data directory not found at {data_dir}. Tools and agent will not be available.")
    except Exception as e:
        logger.error(f"Failed to initialize tools/agent: {e}")
        # Don't fail startup, just log the error


# Global variables for RAG system
vector_db: Optional[VectorDatabase] = None
document_chunks: List[str] = []
document_sources: List[str] = []  # Track sources of chunks
pdf_uploaded = False
rag_agent = None  # LangGraph agent with tools

# Get allowed origins from environment variable or use defaults
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,https://the-ai-engineer-challenge-5jt8mkcw3-ashima-manglas-projects.vercel.app"
).split(",")

# Log the allowed origins for debugging
logger.info(f"Allowed origins: {ALLOWED_ORIGINS}")

# Configure CORS (Cross-Origin Resource Sharing) middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "X-Requested-With"],
    max_age=3600,
)

# Define the data model for chat requests using Pydantic
# This ensures incoming request data is properly validated
class ChatRequest(BaseModel):
    developer_message: str  # Message from the developer/system
    user_message: str      # Message from the user
    model: Optional[str] = "gpt-4o-mini"  # Changed to a valid model
    api_key: str          # OpenAI API key for authentication

# Define the data model for document upload options
class DocumentUploadOptions(BaseModel):
    append_context: bool  # Whether to append to existing context or start fresh


# Define the data model for RAG chat requests
class RAGChatRequest(BaseModel):
    user_message: str      # Message from the user
    model: Optional[str] = "gpt-4o-mini"  # OpenAI model to use
    api_key: str          # OpenAI API key for authentication
    k: Optional[int] = 8   # Number of relevant chunks to retrieve

# PDF upload endpoint for RAG system
@app.post("/api/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...), 
    api_key: str = Form(...),
    append_context: bool = Form(False)
):
    """Upload and process a PDF file for RAG system."""
    global vector_db, document_chunks, pdf_uploaded
    
    try:
        logger.info(f"Received PDF upload: {file.filename}")
        
        # Validate file type
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Validate API key
        if not api_key:
            raise HTTPException(status_code=400, detail="API key is required")
        
        # Set the API key for the aimakerspace library
        os.environ["OPENAI_API_KEY"] = api_key
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Process PDF using aimakerspace
            logger.info("Loading PDF content...")
            pdf_loader = PDFLoader(temp_file_path)
            pdf_loader.load_file()
            
            if not pdf_loader.documents:
                raise HTTPException(status_code=400, detail="Could not extract text from PDF")
            
            # Split text into chunks
            logger.info("Splitting text into chunks...")
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            new_chunks = text_splitter.split_texts(pdf_loader.documents)
            
            if not new_chunks:
                raise HTTPException(status_code=400, detail="No text chunks could be created from PDF")
            
            # Add source information to each chunk
            source_prefix = f"[Source: PDF - {file.filename}]\n"
            new_chunks_with_source = [source_prefix + chunk for chunk in new_chunks]
            
            # Handle append vs replace
            if append_context and document_chunks:
                logger.info("Appending to existing context...")
                document_chunks.extend(new_chunks_with_source)
            else:
                logger.info("Creating new context...")
                document_chunks = new_chunks_with_source
            
            # Create vector database and embeddings
            logger.info(f"Creating embeddings for {len(document_chunks)} chunks...")
            embedding_model = EmbeddingModel()
            vector_db = VectorDatabase(embedding_model)
            vector_db = await vector_db.abuild_from_list(document_chunks)
            
            # Generate a 4-line summary using the first few chunks
            logger.info("Generating document summary...")
            chat_model = ChatOpenAI(model_name="gpt-4o-mini")
            summary_prompt = f"""Provide a concise summary of this document in no more than 100 words. Focus on the key aspects and main contributions.

            Document content:
            {' '.join(document_chunks[:5])}
            """
            summary_messages = [
                {"role": "system", "content": "You are a helpful assistant that provides concise document summaries. Always stay within the specified word limit."},
                {"role": "user", "content": summary_prompt}
            ]
            summary_response = chat_model.run(summary_messages)
            
            pdf_uploaded = True
            
            logger.info(f"Successfully processed PDF with {len(document_chunks)} chunks")
            return {
                "message": f"PDF processed successfully. Here's a summary:\n{summary_response}",
                "chunks_count": len(document_chunks),
                "filename": file.filename,
                "summary": summary_response
            }
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
            
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error processing PDF: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

# RAG-enabled chat endpoint powered by agent
@app.post("/api/rag-chat-mixed-media")
async def rag_chat(request: RAGChatRequest):
    """Chat endpoint powered by RAG agent with retrieve_information and tavily_tool."""
    global rag_agent
    
    try:
        logger.info(f"Received agent chat request: {request.user_message[:100]}...")
        
        # Check if agent is initialized
        if not rag_agent:
            raise HTTPException(
                status_code=503,
                detail="Agent not initialized. Please ensure data directory with PDFs exists and restart the server."
            )
        
        # Set the API key for OpenAI
        os.environ["OPENAI_API_KEY"] = request.api_key
        
        # Create an async generator function for streaming responses
        async def generate():
            try:
                logger.info("Invoking RAG agent...")
                
                # Import system message and handyman prompt
                from langchain_core.messages import SystemMessage
                from agents.rag_agent import HANDYMAN_SYSTEM_PROMPT
                
                # Invoke the agent with system prompt and user's message
                result = rag_agent.invoke({
                    "messages": [
                        SystemMessage(content=HANDYMAN_SYSTEM_PROMPT),
                        HumanMessage(content=request.user_message)
                    ]
                })
                
                # Extract the final response from the agent
                final_message = result["messages"][-1].content
                
                logger.info("Agent response generated successfully")
                
                # Stream the response character by character for smooth UX
                for char in final_message:
                    yield char
                    await asyncio.sleep(0.01)  # Small delay for streaming effect
                        
            except Exception as e:
                error_msg = f"Error in agent generate: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                yield f"Error: {str(e)}"

        # Return a streaming response to the client
        return StreamingResponse(generate(), media_type="text/plain")
    
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error in agent chat endpoint: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=str(e))

# Get PDF status endpoint
@app.get("/api/pdf-status")
async def pdf_status():
    """Get current PDF upload status."""
    global pdf_uploaded, document_chunks
    
    return {
        "pdf_uploaded": pdf_uploaded,
        "chunks_count": len(document_chunks) if document_chunks else 0
    }

# Define the main chat endpoint that handles POST requests
@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        logger.info(f"Received chat request with model: {request.model}")
        logger.info(f"Request headers: {request.headers if hasattr(request, 'headers') else 'No headers'}")
        
        # Initialize OpenAI client with the provided API key
        client = OpenAI(api_key=request.api_key)
        
        # Create an async generator function for streaming responses
        async def generate():
            try:
                logger.info("Starting chat completion request")
                # Create a streaming chat completion request
                stream = client.chat.completions.create(
                    model=request.model,
                    messages=[
                        {"role": "developer", "content": request.developer_message},
                        {"role": "user", "content": request.user_message}
                    ],
                    stream=True  # Enable streaming response
                )
                
                logger.info("Stream created successfully")
                # Yield each chunk of the response as it becomes available
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content
                        
            except Exception as e:
                error_msg = f"Error in generate: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                yield f"Error: {str(e)}"

        # Return a streaming response to the client
        return StreamingResponse(generate(), media_type="text/plain")
    
    except Exception as e:
        error_msg = f"Error in chat endpoint: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=str(e))

# Define a health check endpoint to verify API status
@app.get("/api/health")
async def health_check():
    try:
        logger.info("Health check endpoint called")
        return {"status": "ok", "message": "API is running"}
    except Exception as e:
        error_msg = f"Error in health check: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=str(e))

# Entry point for running the application directly
if __name__ == "__main__":
    import uvicorn
    # Start the server on all network interfaces (0.0.0.0) on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
