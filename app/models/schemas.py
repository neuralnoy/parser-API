from pydantic import BaseModel
from typing import Optional

class DocumentProcessRequest(BaseModel):
    file_path: str  # S3 key of the file to process
    output_prefix: Optional[str] = None  # Optional prefix for output files

class ProcessingResponse(BaseModel):
    status: str
    message: str
    markdown_path: Optional[str] = None
    json_path: Optional[str] = None
