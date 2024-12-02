from fastapi import FastAPI, HTTPException
from app.models.schemas import DocumentProcessRequest, ProcessingResponse
from app.services.document_processor import DocumentProcessor
from app.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
)

document_processor = DocumentProcessor()

@app.post(
    f"{settings.API_V1_STR}/process",
    response_model=ProcessingResponse,
    description="Process a document from S3 and generate markdown and JSON outputs"
)
async def process_document(request: DocumentProcessRequest):
    try:
        md_path, json_path = await document_processor.process_document(
            request.file_path,
            request.output_prefix
        )
        
        return ProcessingResponse(
            status="success",
            message="Document processed successfully",
            markdown_path=md_path,
            json_path=json_path
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
