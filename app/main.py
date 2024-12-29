from fastapi import FastAPI, HTTPException
from app.models.schemas import DocumentProcessRequest, ProcessingResponse
from app.services.document_processor import DocumentProcessor
from app.config import settings
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import multiprocessing
import psutil
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app():
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app

app = create_app()

try:
    document_processor = DocumentProcessor()
except Exception as e:
    logger.error(f"Failed to initialize DocumentProcessor: {str(e)}", exc_info=True)
    raise RuntimeError(f"Service initialization failed: {str(e)}")

@app.post(
    f"{settings.API_V1_STR}/process",
    response_model=ProcessingResponse,
    description="Process a document from S3 and update status in DynamoDB"
)
async def process_document(request: DocumentProcessRequest):
    try:
        # Add logging at the start
        logger.info(f"Starting document processing for document_id: {request.document_id}")
        
        md_path, json_path = await document_processor.process_document(
            document_id=request.document_id,
            knowledge_base_id=request.knowledge_base_id,
            user_id=request.user_id,
            file_path=request.file_path,
            output_prefix=request.output_prefix
        )
        
        return ProcessingResponse(
            status="success",
            message="Document processed successfully",
            document_id=request.document_id,
            markdown_path=md_path,
            json_path=json_path
        )
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        
        # Ensure DynamoDB status is updated even if there's an error
        try:
            await document_processor.dynamodb_service.update_parsing_status(
                document_id=request.document_id,
                knowledge_base_id=request.knowledge_base_id,
                user_id=request.user_id,
                status="FAILED",
                error_message=str(e)
            )
        except Exception as db_error:
            logger.error(f"Failed to update DynamoDB status: {str(db_error)}")
            
        # Re-raise with more context
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Error processing document",
                "error": str(e),
                "document_id": request.document_id
            }
        )

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024  # MB
    }

if __name__ == "__main__":
    # Calculate optimal number of workers based on CPU cores
    workers = multiprocessing.cpu_count()
    
    config = uvicorn.Config(
        app="main:app",
        host="0.0.0.0",
        port=8000,
        workers=workers,
        timeout_keep_alive=2700,  # 45 minutes
        timeout_graceful_shutdown=600,  # 10 minutes
        reload=False,
        loop="uvloop",
        limit_concurrency=50,  # Limit concurrent connections
        limit_max_requests=1500,  # Restart worker after N requests
        timeout_notify=30,  # How long to wait for worker to start/stop
    )
    
    server = uvicorn.Server(config)
    server.run()