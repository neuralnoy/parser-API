import boto3
from datetime import datetime
import asyncio
from app.config import settings
from concurrent.futures import ThreadPoolExecutor
import time
import logging

logger = logging.getLogger(__name__)

class DynamoDBService:
    def __init__(self):
        self.dynamodb = boto3.resource(
            'dynamodb',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )
        self.table = self.dynamodb.Table('parsed_documents')
        self.executor = ThreadPoolExecutor()
        self.token_usage_table = self.dynamodb.Table('parser_token_usage')
        self.embedding_tokens_table = self.dynamodb.Table(settings.DYNAMODB_EMBEDDING_TOKENS_TABLE)
        self.loop = asyncio.get_event_loop()

    def _update_status(self, item: dict):
        """Synchronous method to update DynamoDB"""
        return self.table.put_item(Item=item)

    async def update_parsing_status(
        self,
        document_id: str,
        knowledge_base_id: str,
        user_id: str,
        status: str,
        markdown_path: str = None,
        json_path: str = None,
        error_message: str = None
    ):
        item = {
            'id': document_id,
            'knowledge_base_id': knowledge_base_id,
            'user_id': user_id,
            'status': status,
            'updated_at': datetime.utcnow().isoformat(),
        }
        
        if markdown_path:
            item['markdown_path'] = markdown_path
        if json_path:
            item['json_path'] = json_path
        if error_message:
            item['error_message'] = error_message

        # Run DynamoDB operation in thread pool
        await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self._update_status,
            item
        ) 

    async def record_document_token_usage(
        self,
        document_id: str,
        knowledge_base_id: str,
        user_id: str,
        file_name: str,
        input_tokens: dict,
        output_tokens: dict,
        total_input_tokens: int,
        total_output_tokens: int,
        number_of_images: int
    ):
        item = {
            'id': user_id,
            'document_id': document_id,
            'knowledge_base_id': knowledge_base_id,
            'file_name': file_name,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'number_of_images': number_of_images,
            'processed_at': datetime.utcnow().isoformat()
        }

        await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: self.token_usage_table.put_item(Item=item)
        )

    async def record_embedding_token_usage(
        self,
        document_id: str,
        knowledge_base_id: str,
        user_id: str,
        total_tokens: int,
        chunk_count: int,
        file_name: str
    ) -> None:
        """Record embedding token usage in DynamoDB"""
        try:
            item = {
                'user_id': user_id,
                'document_id': document_id,
                'knowledge_base_id': knowledge_base_id,
                'file_name': file_name,
                'total_tokens': total_tokens,
                'chunk_count': chunk_count,
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat()
            }
            
            await self.loop.run_in_executor(
                self.executor,
                lambda: self.embedding_tokens_table.put_item(Item=item)
            )
            
            logger.info(f"Recorded embedding token usage for document {document_id}")
            
        except Exception as e:
            logger.error(f"Error recording embedding token usage: {str(e)}")
            raise