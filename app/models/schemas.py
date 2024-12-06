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

class TokenUsageRecord(BaseModel):
    document_id: str
    knowledge_base_id: str
    user_id: str
    file_name: str
    input_tokens: dict  # Breakdown of input tokens by operation
    output_tokens: dict  # Breakdown of output tokens by operation
    total_input_tokens: int
    total_output_tokens: int
    number_of_images: int
    processed_at: datetime
