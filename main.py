from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional
import logging
import os
import time
import gc
import psutil
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from query_engine import QueryEngine
import asyncio

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM-Powered Intelligent Query-Retrieval System",
    description="Process large documents and answer queries with explainable responses - Bajaj HackRx",
    version="1.0.0"
)

# Security
security = HTTPBearer()
EXPECTED_TOKEN = os.getenv("API_BEARER_TOKEN", "17eb267be35962c8f54b8a95b6797748e3a03b8ce17c2a23011d558f967098b2")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify Bearer token"""
    if credentials.credentials != EXPECTED_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Request/Response Models - Updated to match official requirements
class QueryRequest(BaseModel):
    documents: str  # Single document URL (as per official spec)
    questions: List[str]  # Natural language questions

class QueryResponse(BaseModel):
    answers: List[str]

# Initialize processors with memory monitoring
document_processor = DocumentProcessor()
query_engine = QueryEngine()

def log_memory_usage(stage: str):
    """Log current memory usage"""
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"💾 Memory usage at {stage}: {memory_mb:.1f} MB")
        return memory_mb
    except Exception as e:
        logger.warning(f"Could not log memory usage: {e}")
        return 0

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def process_query(request: QueryRequest, token: str = Depends(verify_token)):
    """
    Main endpoint to process documents and answer queries.
    Official Bajaj HackRx endpoint with Bearer token authentication.
    
    Args:
        request: Contains document URL and questions
        token: Bearer token for authentication
        
    Returns:
        JSON response with answers
    """
    start_time = time.time()
    initial_memory = log_memory_usage("start")
    
    try:
        logger.info(f"🚀 Starting request processing...")
        logger.info(f"📄 Processing document and {len(request.questions)} questions")
        logger.info(f"🔗 Document URL: {request.documents}")
        
        # Step 1: Download and process document (convert single URL to list for internal processing)
        logger.info(f"📥 Step 1: Starting document download and processing...")
        doc_start_time = time.time()
        document_urls = [request.documents] if isinstance(request.documents, str) else request.documents
        all_chunks = []
        
        for i, doc_url in enumerate(document_urls):
            logger.info(f"📄 Processing document {i+1}/{len(document_urls)}: {doc_url[:100]}...")
            try:
                doc_start = time.time()
                chunks = await document_processor.process_document(doc_url)
                doc_time = time.time() - doc_start
                logger.info(f"✅ Document {i+1} processed in {doc_time:.1f}s - {len(chunks)} chunks extracted")
                all_chunks.extend(chunks)
                
                # Memory cleanup after each document
                gc.collect()
                log_memory_usage(f"after_doc_{i+1}")
                
            except Exception as e:
                logger.error(f"❌ Error processing document {i+1}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
        
        total_doc_time = time.time() - doc_start_time
        logger.info(f"📊 Total chunks extracted: {len(all_chunks)} in {total_doc_time:.1f}s")
        
        # Step 2: Create embeddings and build vector index
        logger.info(f"🧠 Step 2: Building vector index...")
        index_start = time.time()
        
        await query_engine.build_index(all_chunks)
        index_time = time.time() - index_start
        logger.info(f"✅ Vector index built in {index_time:.1f}s")
        
        # Memory cleanup after indexing
        gc.collect()
        log_memory_usage("after_indexing")
        
        # Step 3: Process each question with focus on accuracy and explainability
        logger.info(f"❓ Step 3: Processing {len(request.questions)} questions...")
        answers = []
        for i, question in enumerate(request.questions):
            logger.info(f"🤔 Processing question {i+1}/{len(request.questions)}: {question[:100]}...")
            try:
                answer = await query_engine.answer_question(question)
                answers.append(answer)
                logger.info(f"✅ Question {i+1} answered successfully")
                
                # Memory cleanup after each question
                gc.collect()
                
            except Exception as e:
                logger.error(f"❌ Error answering question {i+1}: {str(e)}")
                # Continue with other questions instead of failing completely
                answers.append(f"Error processing question: {str(e)}")
        
        total_time = time.time() - start_time
        final_memory = log_memory_usage("end")
        memory_delta = final_memory - initial_memory
        logger.info(f"🎉 Successfully processed all {len(request.questions)} questions in {total_time:.1f}s")
        logger.info(f"💾 Memory delta: {memory_delta:+.1f} MB")
        
        # Final validation
        if len(answers) != len(request.questions):
            logger.error(f"❌ Answer count mismatch: {len(answers)} answers for {len(request.questions)} questions")
            raise HTTPException(status_code=500, detail="Answer count mismatch")
            
        return QueryResponse(answers=answers)
        
    except HTTPException:
        # Re-raise HTTP exceptions (timeouts, auth errors, etc.)
        raise
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"💥 Fatal error processing request after {total_time:.1f}s: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        # Final memory cleanup
        gc.collect()

@app.get("/health")
async def health_check():
    """Health check endpoint with memory info"""
    memory_mb = log_memory_usage("health_check")
    return {
        "status": "healthy", 
        "message": "LLM Query-Retrieval System is running",
        "memory_mb": round(memory_mb, 1)
    }

@app.get("/api/v1/health")
async def health_check_v1():
    """Health check endpoint for API v1 with memory info"""
    memory_mb = log_memory_usage("health_check_v1")
    return {
        "status": "healthy", 
        "message": "LLM Query-Retrieval System API v1 is running",
        "memory_mb": round(memory_mb, 1)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
