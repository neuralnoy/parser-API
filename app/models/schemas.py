from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class DocumentProcessRequest(BaseModel):
    document_id: str
    knowledge_base_id: str
    user_id: str
    file_path: str
    output_prefix: Optional[str] = None

class ProcessingResponse(BaseModel):
    status: str
    message: str
    document_id: str
    markdown_path: Optional[str] = None
    json_path: Optional[str] = None
