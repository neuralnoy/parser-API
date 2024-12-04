from fastapi import FastAPI, HTTPException
from app.models.schemas import DocumentProcessRequest, ProcessingResponse
from app.services.document_processor import DocumentProcessor
from app.config import settings
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

document_processor = DocumentProcessor()

@app.post(
    f"{settings.API_V1_STR}/process",
    response_model=ProcessingResponse,
    description="Process a document from S3 and update status in DynamoDB"
)
async def process_document(request: DocumentProcessRequest):
    try:
        # Start processing and wait for results
        md_path, json_path = await document_processor.process_document(
            document_id=request.document_id,
            knowledge_base_id=request.knowledge_base_id,
            user_id=request.user_id,
            file_path=request.file_path,
            output_prefix=request.output_prefix
        )
        
        # Return the actual results
        return ProcessingResponse(
            status="success",
            message="Document processed successfully",
            document_id=request.document_id,
            markdown_path=md_path,
            json_path=json_path
        )
        
    except Exception as e:
        await document_processor.dynamodb_service.update_parsing_status(
            document_id=request.document_id,
            knowledge_base_id=request.knowledge_base_id,
            user_id=request.user_id,
            status="FAILED",
            error_message=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8001,
        timeout_keep_alive=3600,  # 1 hour
        workers=2,  # Number of worker processes
        timeout_graceful_shutdown=300,  # 5 minutes grace period for shutdown
    )
