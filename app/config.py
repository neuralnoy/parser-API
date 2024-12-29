import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME = "Docling Service"
    VERSION = "1.0.0"
    API_V1_STR = "/api/v1"
    
    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # AWS
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET = os.getenv("S3_BUCKET")
    
    # Local processing
    PROCESSING_DIR = os.getenv("PROCESSING_DIR", "processing")
    
    # Zilliz Cloud
    ZILLIZ_CLOUD_URI = os.getenv("ZILLIZ_CLOUD_URI")
    ZILLIZ_CLOUD_API_KEY = os.getenv("ZILLIZ_CLOUD_API_KEY")
    
    # Add to Settings class
    DYNAMODB_EMBEDDING_TOKENS_TABLE = os.getenv("DYNAMODB_EMBEDDING_TOKENS_TABLE", "embedding_tokens")

settings = Settings()
