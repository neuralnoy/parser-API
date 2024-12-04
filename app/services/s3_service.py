import boto3
from pathlib import Path
from app.config import settings
from typing import Optional

class S3Service:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )
        self.bucket = settings.S3_BUCKET

    def download_file(self, s3_key: str, local_path: Path | str) -> None:
        """Download file from S3 to local path"""
        # Convert string to Path if needed
        if isinstance(local_path, str):
            local_path = Path(local_path)
        
        # Create parent directories if they don't exist
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.s3_client.download_file(self.bucket, s3_key, str(local_path))

    def upload_file(self, local_path: Path, s3_key: str) -> None:
        self.s3_client.upload_file(str(local_path), self.bucket, s3_key)

    def get_output_prefix(self, input_path: str, output_prefix: Optional[str] = None) -> str:
        if output_prefix:
            return output_prefix
        
        # Create output prefix one level deeper from input
        parts = input_path.split('/')
        if len(parts) > 1:
            return '/'.join(parts[:-1]) + '/processed/' + parts[-1].split('.')[0]
        return 'processed/' + parts[-1].split('.')[0]
