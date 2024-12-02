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

settings = Settings()
