from typing import List, Dict
import logging
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from pymilvus import (
    CollectionSchema, 
    FieldSchema, 
    DataType, 
    connections, 
    utility, 
    Collection,
    MilvusException
)
import time
from ..config import settings
from pathlib import Path
from app.services.token_counter_service import TokenCounterService
from app.services.dynamodb_service import DynamoDBService

logger = logging.getLogger(__name__)

class VectorStoreService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
        self.token_counter = TokenCounterService()
        self.dynamodb_service = DynamoDBService()
        self.max_retries = 3
        self.retry_delay = 2  # Initial delay in seconds
        self.connected = False
        self.connect_to_milvus()

    def connect_to_milvus(self):
        """Establish connection to Milvus with retry logic"""
        retries = 0
        while retries < self.max_retries:
            try:
                # First check if connection exists and disconnect if it does
                try:
                    connections.disconnect("default")
                except:
                    pass
                
                # Create new connection
                connections.connect(
                    alias="default",
                    uri=settings.ZILLIZ_CLOUD_URI,
                    token=settings.ZILLIZ_CLOUD_API_KEY,
                    secure=True
                )
                logger.info("Successfully connected to Milvus")
                self.connected = True
                return
            except MilvusException as e:
                retries += 1
                wait_time = self.retry_delay * (2 ** (retries - 1))  # Exponential backoff
                logger.warning(f"Failed to connect to Milvus (attempt {retries}/{self.max_retries}): {str(e)}")
                logger.info(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            except Exception as e:
                logger.error(f"Unexpected error connecting to Milvus: {str(e)}")
                raise

        logger.error("Failed to connect to Milvus after maximum retries")
        self.connected = False

    def ensure_connection(self):
        """Ensure Milvus connection is active"""
        if not self.connected:
            self.connect_to_milvus()
        else:
            try:
                # Test connection by listing collections
                utility.list_collections()
            except Exception:
                logger.info("Connection lost, attempting to reconnect...")
                self.connect_to_milvus()

    def get_collection_name(self, user_id: str, knowledge_base_id: str) -> str:
        """Generate unique collection name for user"""
        # Remove any dots and @ symbols and replace with underscores
        sanitized_user_id = user_id.replace('.', '_').replace('@', '_')
        # Remove any remaining non-alphanumeric characters
        sanitized_user_id = ''.join(c if c.isalnum() else '_' for c in sanitized_user_id)
        # Remove any consecutive underscores
        while '__' in sanitized_user_id:
            sanitized_user_id = sanitized_user_id.replace('__', '_')
        # Remove any trailing underscores
        sanitized_user_id = sanitized_user_id.rstrip('_')
        
        return sanitized_user_id  # Just return the sanitized user ID without prefix

    def create_collection(self, collection_name: str) -> Collection:
        """Create new collection if it doesn't exist"""
        if utility.has_collection(collection_name):
            return Collection(name=collection_name)

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1536),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="knowledge_base_id", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="document_name", dtype=DataType.VARCHAR, max_length=255)
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="Document collection",
            enable_dynamic_field=True
        )

        collection = Collection(name=collection_name, schema=schema)
        
        # Create index
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection.create_index(field_name="vector", index_params=index_params)
        
        return collection

    def disconnect_from_milvus(self):
        """Disconnect from Milvus"""
        try:
            connections.disconnect("default")
            logger.info("Successfully disconnected from Milvus")
        except Exception as e:
            logger.error(f"Failed to disconnect from Milvus: {str(e)}")

    async def process_markdown(
        self, 
        markdown_path: Path,
        document_id: str,
        user_id: str,
        knowledge_base_id: str
    ) -> None:
        """Process markdown file and store in vector database"""
        try:
            logger.info(f"Starting vector processing for document {document_id}")
            
            # Ensure connection is established
            self.ensure_connection()
            if not self.connected:
                logger.error("Cannot process document: Unable to establish Milvus connection")
                return  # Return without raising to prevent complete service failure
            
            # Load and process the markdown directly from local path
            loader = UnstructuredMarkdownLoader(str(markdown_path))
            documents = loader.load()

            logger.info(f"Successfully loaded markdown, splitting into chunks...")
            
            # Split text into chunks
            text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)

            collection_name = self.get_collection_name(user_id, knowledge_base_id)

            # Create collection if it doesn't exist
            try:
                if not utility.has_collection(collection_name):
                    collection = self.create_collection(collection_name)
                else:
                    collection = Collection(name=collection_name)
            except MilvusException as e:
                logger.error(f"Milvus collection error: {str(e)}")
                return  # Return without raising to prevent complete service failure

            # Prepare data
            texts = [doc.page_content for doc in docs]
            chunk_count = len(texts)
            
            # Calculate total tokens using TokenCounterService
            total_tokens = sum(self.token_counter.count_tokens(text) for text in texts)
            
            metadatas = [doc.metadata for doc in docs]
            embeddings_list = self.embeddings.embed_documents(texts)

            # Get the original document name from the markdown path
            document_name = markdown_path.stem.replace('-processed', '')

            # Create entities
            entities = []
            for text, vector, metadata in zip(texts, embeddings_list, metadatas):
                entities.append({
                    "text": text,
                    "vector": vector,
                    "document_id": document_id,
                    "knowledge_base_id": knowledge_base_id,
                    "document_name": document_name
                })

            # Insert data
            collection.insert(entities)
            collection.flush()
            
            # Load the collection into memory
            collection.load()
            
            # Record token usage in DynamoDB
            await self.dynamodb_service.record_embedding_token_usage(
                document_id=document_id,
                knowledge_base_id=knowledge_base_id,
                user_id=user_id,
                total_tokens=total_tokens,
                chunk_count=chunk_count,
                file_name=markdown_path.name
            )
            
            logger.info(f"Successfully processed document {document_id} for user {user_id}")
            logger.info(f"Embedding stats - Total tokens: {total_tokens}, Chunks: {chunk_count}")

        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise
        finally:
            self.disconnect_from_milvus()

    async def delete_document_chunks(
        self, 
        document_id: str,
        user_id: str,
        knowledge_base_id: str
    ) -> None:
        """Delete all chunks associated with a document"""
        try:
            collection_name = self.get_collection_name(user_id, knowledge_base_id)
            if utility.has_collection(collection_name):
                collection = Collection(name=collection_name)
                collection.delete(f"document_id == '{document_id}'")
                logger.info(f"Successfully deleted chunks for document {document_id}")
        except Exception as e:
            logger.error(f"Error deleting document chunks: {str(e)}")
            raise

vector_store_service = VectorStoreService() 