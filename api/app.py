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
from aimakerspace.video_utils import VideoLoader
from aimakerspace.vectordatabase import VectorDatabase
from aimakerspace.openai_utils.embedding import EmbeddingModel
from aimakerspace.openai_utils.chatmodel import ChatOpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI application with a title
app = FastAPI(title="RAG-Enabled Chat API")

# Global variables for RAG system
vector_db: Optional[VectorDatabase] = None
document_chunks: List[str] = []
document_sources: List[str] = []  # Track sources of chunks
pdf_uploaded = False

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

# Define the data model for YouTube URL upload
class YouTubeRequest(BaseModel):
    url: HttpUrl          # YouTube video URL
    api_key: str          # OpenAI API key for authentication
    options: DocumentUploadOptions  # Upload options

# Define the data model for RAG chat requests
class RAGChatRequest(BaseModel):
    user_message: str      # Message from the user
    model: Optional[str] = "gpt-4o-mini"  # OpenAI model to use
    api_key: str          # OpenAI API key for authentication
    k: Optional[int] = 3   # Number of relevant chunks to retrieve

# YouTube URL upload endpoint
@app.post("/api/upload-youtube")
async def upload_youtube(request: YouTubeRequest):
    """Process a YouTube video URL for RAG system."""
    global vector_db, document_chunks, pdf_uploaded
    
    try:
        logger.info(f"Received YouTube URL: {request.url}")
        
        # Validate API key
        if not request.api_key:
            raise HTTPException(status_code=400, detail="API key is required")
        
        # Set the API key for the aimakerspace library
        os.environ["OPENAI_API_KEY"] = request.api_key
        
        try:
            # Process YouTube video
            logger.info("Loading video content...")
            video_loader = VideoLoader(str(request.url))
            video_loader.load_url()
            
            if not video_loader.documents:
                raise HTTPException(status_code=400, detail="Could not extract text from video")
            
            # Split text into chunks
            logger.info("Splitting text into chunks...")
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            new_chunks = text_splitter.split_texts(video_loader.documents)
            
            if not new_chunks:
                raise HTTPException(status_code=400, detail="No text chunks could be created from video")
            
            # Add source information to each chunk
            source_prefix = f"[Source: YouTube - {request.url}]\n"
            new_chunks_with_source = [source_prefix + chunk for chunk in new_chunks]
            
            # Handle append vs replace
            if request.options.append_context and document_chunks:
                document_chunks.extend(new_chunks_with_source)
            else:
                document_chunks = new_chunks_with_source
            
            # Create vector database and embeddings
            logger.info(f"Creating embeddings for {len(document_chunks)} chunks...")
            embedding_model = EmbeddingModel()
            vector_db = VectorDatabase(embedding_model)
            vector_db = await vector_db.abuild_from_list(document_chunks)
            
            # Generate a summary using the video chunks only
            logger.info("Generating video summary...")
            chat_model = ChatOpenAI(model_name="gpt-4o-mini")
            summary_prompt = f"""Provide a concise summary of this video in no more than 100 words. Focus on the key points and main message.

            Video transcript:
            {' '.join(new_chunks_with_source[:5])}
            """
            summary_messages = [
                {"role": "system", "content": "You are a helpful assistant that provides concise video summaries. Always stay within the specified word limit."},
                {"role": "user", "content": summary_prompt}
            ]
            summary_response = chat_model.run(summary_messages)
            
            pdf_uploaded = True
            
            logger.info(f"Successfully processed video with {len(document_chunks)} chunks")
            return {
                "message": f"Video processed successfully. Here's a summary:\n{summary_response}",
                "chunks_count": len(document_chunks),
                "url": str(request.url),
                "summary": summary_response
            }
            
        except Exception as e:
            error_msg = f"Error processing video: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
            
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error processing video: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

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

# RAG-enabled chat endpoint
@app.post("/api/rag-chat-mixed-media")
async def rag_chat(request: RAGChatRequest):
    """Chat endpoint that uses uploaded mixed media (PDFs, YouTube videos) as context."""
    global vector_db, document_chunks, pdf_uploaded
    
    try:
        logger.info(f"Received RAG chat request: {request.user_message[:100]}...")
        
        # Check if PDF has been uploaded
        if not pdf_uploaded or not vector_db:
            raise HTTPException(
                status_code=400, 
                detail="No PDF has been uploaded. Please upload a PDF first."
            )
        
        # Set the API key for the aimakerspace library
        os.environ["OPENAI_API_KEY"] = request.api_key
        
        # Adjust search strategy based on question type
        k = request.k
        query = request.user_message.lower()
        
        # For video-specific questions
        if any(word in query for word in ["video", "vide", "youtube", "watch", "watched", "viewing", "clip"]):
            k = max(k, 10)  # Use more chunks to find video content
            # Add search terms to help find video content
            request.user_message += " [Source: YouTube video content]"
            logger.info("Video-specific question detected, expanding search context")
        
        # For document/PDF-specific questions  
        elif any(word in query for word in ["document", "pdf", "paper", "file", "uploaded file"]):
            k = max(k, 6)  # Use more chunks to find document content
            # Add search terms to help find PDF content
            request.user_message += " [PDF document content]"
            logger.info("Document-specific question detected, expanding search context")
        
        # For methodology/research methods
        elif any(word in query for word in ["methodology", "method", "approach", "technique", "procedure", "framework", "experimental", "analysis", "evaluation", "implementation"]):
            k = max(k, 8)  # Use more chunks to find methodology sections
            # Modify query to look for methodology-related terms
            request.user_message += " [methodology, methods, approach, techniques, procedures, framework, experimental design, analysis, evaluation]"
            logger.info("Methodology question detected, expanding search context")
        
        # For references/citations
        elif any(word in query for word in ["reference", "cite", "citation", "author", "publication", "work"]):
            k = max(k, 7)  # Use more chunks to find scattered references
            # Modify query to specifically look for reference patterns
            request.user_message += " [et al., references, citations, authors]"
            logger.info("Reference question detected, expanding search context")
        
        # For pros/cons, advantages/disadvantages
        elif any(word in query for word in ["pro", "con", "advantage", "disadvantage", "benefit", "limitation", "strength", "weakness"]):
            k = max(k, 5)  # Use more chunks to find scattered pros/cons
            # Modify query to look for evaluative statements
            request.user_message += " [advantages, disadvantages, benefits, limitations, performance]"
            logger.info("Pros/cons question detected, expanding search context")
        
        # For broad or vague questions
        elif any(phrase in query for phrase in [
            "what is this", "what's this", "what is the document", "what's the document",
            "give me more details", "tell me more", "can you explain", "give me details"
        ]):
            k = max(k, 5)  # Use at least 5 chunks for broad questions
            logger.info("Broad/vague question detected, expanding search context")
        
        # Retrieve relevant chunks from vector database
        logger.info(f"Searching for relevant context with k={k}...")
        relevant_chunks = vector_db.search_by_text(
            request.user_message, 
            k=k, 
            return_as_text=True
        )
        
        if not relevant_chunks:
            # Fallback: use first few chunks if no relevant chunks found
            logger.warning("No relevant chunks found, using fallback strategy with first 5 chunks")
            relevant_chunks = document_chunks[:5] if len(document_chunks) >= 5 else document_chunks
        
        # If still no chunks, try with a more general search
        if not relevant_chunks and document_chunks:
            logger.warning("Using all available chunks as last resort")
            relevant_chunks = document_chunks[:8]  # Use first 8 chunks
        
        if not relevant_chunks:
            raise HTTPException(status_code=500, detail="Could not retrieve relevant context")
        
        # Combine relevant chunks into context
        context = "\n\n".join(relevant_chunks)
        
        # Create system message with context
        # Detect if this is a broad/vague question
        is_broad_question = any(phrase in request.user_message.lower() for phrase in [
            "what is this", "what's this", "what is the document", "what's the document",
            "give me more details", "tell me more", "can you explain", "give me details"
        ])

        # Customize system message based on question type
        query = request.user_message.lower()
        
        if any(word in query for word in ["methodology", "method", "approach", "technique", "procedure", "framework", "experimental", "analysis", "evaluation", "implementation"]):
            system_message = f"""You are a helpful assistant that explains research methodology and methods in academic papers.

Context from PDF:
{context}

Instructions:
- Look for any descriptions of methods, methodologies, approaches, techniques, procedures, or frameworks
- Explain the experimental design, data collection methods, analysis techniques, and evaluation procedures
- Include details about datasets, tools, algorithms, or technologies used
- Describe how the research was conducted and what steps were taken
- If methodology information is found, provide a comprehensive explanation
- If no methodology is found in the provided context, say "I cannot find methodology information in this section of the document"
- Be specific and cite the exact text where methodology is described"""

        elif any(word in query for word in ["reference", "cite", "citation", "paper", "author", "publication", "work"]):
            system_message = f"""You are a helpful assistant that finds references and citations in academic papers. 

Context from PDF:
{context}

Instructions:
- Look for any references, citations, or mentions of other papers/authors in the context
- Include author names, paper titles, years, and any other citation details you find
- If you find references, format them clearly and explain what they are cited for
- If no references are found in the provided context, say "I cannot find any references in this section of the document"
- Be specific and cite the exact text where references are mentioned"""

        elif any(word in query for word in ["pro", "con", "advantage", "disadvantage", "benefit", "limitation", "strength", "weakness"]):
            system_message = f"""You are a helpful assistant that analyzes advantages and disadvantages in academic papers. 

Context from PDF:
{context}

Instructions:
- Look for any mentions of advantages, benefits, strengths, limitations, challenges, or drawbacks
- Organize your response into clear pros and cons if both are found
- Look for comparative statements, performance metrics, or evaluative language
- If you find partial information (only pros or only cons), provide what you found
- Be specific and cite the relevant parts of the context
- If no pros/cons are found in the context, say "I cannot find explicit advantages or disadvantages in this section"
- Focus on factual statements from the text, not interpretations"""

        else:
            system_message = f"""You are a helpful assistant that answers questions based ONLY on the provided context from uploaded content (PDFs, Word documents, and YouTube videos). 

Context from uploaded content:
{context}

Instructions:
- Answer the user's question using ONLY the information provided in the context above
- The context may include content from multiple sources (documents and videos) - each source is labeled
- For video-specific questions, focus on content marked with "[Source: YouTube -"
- For document-specific questions, focus on content marked with "[PDF:" or document names
- For broad or vague questions, provide a comprehensive overview of the relevant information from the context
- For specific questions, be precise and cite relevant parts of the context
- If the answer cannot be found in the context, say "I cannot find information about that in the provided content"
- Do not use any external knowledge beyond what's in the context
- Be direct and informative in your responses"""

        # Initialize ChatOpenAI and create streaming response
        chat_model = ChatOpenAI(model_name=request.model)
        
        # Create an async generator function for streaming responses
        async def generate():
            try:
                logger.info("Starting RAG chat completion request")
                
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": request.user_message}
                ]
                
                # Use the astream method for streaming
                async for chunk in chat_model.astream(messages):
                    yield chunk
                        
            except Exception as e:
                error_msg = f"Error in RAG generate: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                yield f"Error: {str(e)}"

        # Return a streaming response to the client
        return StreamingResponse(generate(), media_type="text/plain")
    
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error in RAG chat endpoint: {str(e)}\n{traceback.format_exc()}"
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
