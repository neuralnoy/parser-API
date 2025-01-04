import json
import uuid
import base64
import shutil
import logging
from pathlib import Path
from typing import Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import os
import resource
import gc
import psutil
from fastapi import HTTPException

from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.datamodel.base_models import InputFormat, ConversionStatus
from docling.datamodel.document import ConversionResult
from docling_core.types.doc import PictureItem

from app.config import settings
from app.services.s3_service import S3Service
from app.services.dynamodb_service import DynamoDBService
from app.services.token_counter_service import TokenCounterService
from app.services.vector_store_service import VectorStoreService, vector_store_service

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.s3_service = S3Service()
        self.dynamodb_service = DynamoDBService()
        self.executor = ThreadPoolExecutor()
        self.token_counter = TokenCounterService()
        self.vector_store_service = vector_store_service
        
        # Initialize document converter with options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.images_scale = 2.0
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True

        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                InputFormat.DOCX: WordFormatOption(pipeline_cls=SimplePipeline),
            },
        )

        # Add to existing initialization
        self.token_usage = {
            'input_tokens': {
                'image_description': 0,
                'other_operations': 0  # Add more operations as needed
            },
            'output_tokens': {
                'image_description': 0,
                'other_operations': 0  # Add more operations as needed
            }
        }
        self.number_of_images = 0
        self.process = psutil.Process(os.getpid())

    async def process_document(
        self, 
        document_id: str,
        knowledge_base_id: str,
        user_id: str,
        file_path: str, 
        output_prefix: str = None
    ) -> Tuple[str, str]:
        """Process a document and update status in DynamoDB"""
        
        try:
            # Get current limits
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            
            # Try to set a reasonable limit based on system memory
            total_memory = psutil.virtual_memory().total
            desired_limit = min(total_memory, 1024 * 1024 * 1024)  # Use 1GB or system total, whichever is smaller
            
            try:
                # Only set the limit if it's lower than current
                if desired_limit < soft or soft == -1:
                    resource.setrlimit(resource.RLIMIT_AS, (desired_limit, hard))
                    logger.info(f"Set memory limit to {desired_limit / (1024*1024):.2f} MB")
            except ValueError as e:
                logger.warning(f"Could not set memory limit: {e}. Continuing with system defaults.")
            
            # Clear memory before processing
            gc.collect()
            
            # Create unique processing directory
            processing_dir = Path(settings.PROCESSING_DIR) / str(uuid.uuid4())
            processing_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Update initial status
                await self.dynamodb_service.update_parsing_status(
                    document_id=document_id,
                    knowledge_base_id=knowledge_base_id,
                    user_id=user_id,
                    status="PROCESSING"
                )
                
                # Download file from S3
                input_file = processing_dir / Path(file_path).name
                self.s3_service.download_file(file_path, input_file)
                
                # Process document in thread pool to not block event loop
                conv_results = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self._convert_document,
                    input_file
                )
                
                # Export and upload results
                md_path, json_path = await self._export_and_upload_results(
                    conv_results,
                    processing_dir,
                    file_path,
                    document_id,
                    user_id,
                    knowledge_base_id,
                    output_prefix
                )
                
                # Before returning, record token usage
                total_input_tokens = sum(self.token_usage['input_tokens'].values())
                total_output_tokens = sum(self.token_usage['output_tokens'].values())

                await self.dynamodb_service.record_document_token_usage(
                    document_id=document_id,
                    knowledge_base_id=knowledge_base_id,
                    user_id=user_id,
                    file_name=Path(file_path).name,
                    input_tokens=self.token_usage['input_tokens'],
                    output_tokens=self.token_usage['output_tokens'],
                    total_input_tokens=total_input_tokens,
                    total_output_tokens=total_output_tokens,
                    number_of_images=self.number_of_images
                )

                # Reset token tracking for next document
                self.token_usage = {
                    'input_tokens': {
                        'image_description': 0,
                        'other_operations': 0
                    },
                    'output_tokens': {
                        'image_description': 0,
                        'other_operations': 0
                    }
                }
                self.number_of_images = 0

                # Update success status
                await self.dynamodb_service.update_parsing_status(
                    document_id=document_id,
                    knowledge_base_id=knowledge_base_id,
                    user_id=user_id,
                    status="COMPLETED",
                    markdown_path=md_path,
                    json_path=json_path
                )
                
                return md_path, json_path

            except Exception as e:
                logger.error(f"Error processing document: {str(e)}", exc_info=True)
                # Update error status
                await self.dynamodb_service.update_parsing_status(
                    document_id=document_id,
                    knowledge_base_id=knowledge_base_id,
                    user_id=user_id,
                    status="FAILED",
                    error_message=str(e)
                )
                raise

            finally:
                # Add a small delay to ensure all file handles are closed
                await asyncio.sleep(0.5)
                
                # Ensure cleanup happens in all cases
                if processing_dir.exists():
                    try:
                        # Walk directory tree bottom-up and remove everything
                        for root, dirs, files in os.walk(processing_dir, topdown=False):
                            root_path = Path(root)
                            
                            # Remove all files in current directory
                            for name in files:
                                file_path = root_path / name
                                try:
                                    file_path.unlink()
                                except Exception as e:
                                    logger.warning(f"Failed to remove file {file_path}: {e}")
                            
                            # Remove all subdirectories
                            for name in dirs:
                                dir_path = root_path / name
                                try:
                                    dir_path.rmdir()
                                except Exception as e:
                                    logger.warning(f"Failed to remove directory {dir_path}: {e}")
                        
                        # Finally remove the root processing directory
                        processing_dir.rmdir()
                        logger.info(f"Cleaned up directory: {processing_dir}")
                        
                    except Exception as cleanup_error:
                        logger.error(f"Error during cleanup: {cleanup_error}", exc_info=True)

        except MemoryError:
            self._emergency_cleanup()
            raise HTTPException(
                status_code=503,
                detail="Server is temporarily unable to process this document due to resource constraints"
            )
        except Exception as e:
            self._emergency_cleanup()
            raise

    def _convert_document(self, input_file: Path) -> list[ConversionResult]:
        """Convert document using Docling"""
        return list(self.doc_converter.convert_all(
            [input_file],
            raises_on_error=False,
        ))

    async def _export_and_upload_results(
        self,
        conv_results: list[ConversionResult],
        processing_dir: Path,
        input_path: str,
        document_id: str,
        user_id: str,
        knowledge_base_id: str,
        output_prefix: str = None
    ) -> Tuple[str, str]:
        """Export results and upload to S3"""
        output_dir = processing_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        s3_prefix = self.s3_service.get_output_prefix(input_path, output_prefix)
        md_path = None
        json_path = None
        local_md_file = None
        
        for conv_res in conv_results:
            if conv_res.status != ConversionStatus.SUCCESS:
                logger.warning(f"Document {conv_res.input.file} failed to convert or partially succeeded")
                continue
                
            doc_filename = conv_res.input.file.stem
            
            try:
                # Export and process JSON
                json_file = output_dir / f"{doc_filename}.json"
                with json_file.open("w") as fp:
                    json_data = conv_res.document.export_to_dict()
                    # Clean base64 data before saving
                    cleaned_json_data = self._clean_base64_from_json(json_data.copy())
                    json.dump(cleaned_json_data, fp, indent=2)
                
                # Get markdown content
                markdown_content = conv_res.document.export_to_markdown()
                
                # Process images and update content
                picture_counter = 0
                descriptions = []
                
                for element, _level in conv_res.document.iterate_items():
                    if isinstance(element, PictureItem):
                        picture_counter += 1
                        image_file = output_dir / f"{doc_filename}-picture-{picture_counter}.png"
                        
                        # Save image
                        with image_file.open("wb") as fp:
                            element.get_image(conv_res.document).save(fp, "PNG")
                        
                        # Get AI description
                        description = self._get_image_description(image_file)
                        descriptions.append((element.self_ref, description))
                        
                        # Update markdown
                        markdown_content = markdown_content.replace(
                            "<!-- image -->",
                            f"![Image or Figure here: \n **Description of the image or figure, generated by AI:**: \n\n {description} \n\n **End of the description.** ] \n\n",
                            1
                        )
                
                # Update JSON with descriptions
                if "pictures" in json_data:
                    for picture_ref, description in descriptions:
                        for picture in json_data["pictures"]:
                            if picture["self_ref"] == picture_ref:
                                if "annotations" not in picture:
                                    picture["annotations"] = []
                                picture["annotations"].append({
                                    "type": "ai_description",
                                    "content": description
                                })
                                break

                # Save JSON and markdown files
                with json_file.open("w") as fp:
                    json.dump(json_data, fp, indent=2)
                
                md_file = output_dir / f"{doc_filename}.md"
                with md_file.open("w") as fp:
                    fp.write(markdown_content)
                
                local_md_file = md_file  # Store the local markdown file path
                
                # Upload files to S3
                s3_md_path = f"{s3_prefix}/{doc_filename}.md"
                s3_json_path = f"{s3_prefix}/{doc_filename}.json"
                
                self.s3_service.upload_file(md_file, s3_md_path)
                self.s3_service.upload_file(json_file, s3_json_path)
                
                md_path = s3_md_path
                json_path = s3_json_path
                
                # Process markdown for vector storage
                try:
                    await self.vector_store_service.process_markdown(
                        markdown_path=local_md_file,  # Use the stored local file path
                        document_id=document_id,
                        user_id=user_id,
                        knowledge_base_id=knowledge_base_id
                    )
                except Exception as e:
                    logger.error(f"Error processing markdown for vector storage: {str(e)}")
                    # Don't raise the error - we still want to return the processed files
                
            except Exception as e:
                logger.error(f"Error processing document {doc_filename}: {str(e)}", exc_info=True)
                raise
        
        return md_path, json_path

    def _get_image_description(self, image_path: Path) -> str:
        """Get AI description for image"""
        try:
            base64_image = self._encode_image(image_path)
            prompt = "This is not a conversation. Describe this image in great detail. Be very specific. Be very accurate. If you can't see anything, say 'No visible content'."
            
            # Construct the complete messages block
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "low"
                            },
                        },
                    ],
                }
            ]
            
            # Count only the prompt tokens
            input_tokens = self.token_counter.count_tokens(prompt)
            # Add 200 tokens for the image
            input_tokens += 200

            self.token_usage['input_tokens']['image_description'] += input_tokens

            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=1000,
            )

            description = response.choices[0].message.content
            output_tokens = self.token_counter.count_tokens(description)
            self.token_usage['output_tokens']['image_description'] += output_tokens
            self.number_of_images += 1

            logger.info(f"Token usage - Input: {input_tokens}, Output: {output_tokens}")

            return description
            
        except Exception as e:
            logger.error(f"Error getting image description: {str(e)}", exc_info=True)
            return "Image description unavailable"

    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _emergency_cleanup(self):
        """Emergency cleanup when things go wrong"""
        try:
            gc.collect()
            self.process.memory_info()
            
            # Reset document converter
            self.doc_converter = None
            self.__init__()  # Reinitialize the processor
            
        except Exception as e:
            logger.error(f"Emergency cleanup failed: {str(e)}", exc_info=True)

    def _clean_base64_from_json(self, json_data: dict) -> dict:
        """Remove base64 representations from JSON data while preserving image metadata"""
        if isinstance(json_data, dict):
            for key, value in json_data.items():
                if key == "uri" and isinstance(value, str) and value.startswith('data:image/'):
                    json_data[key] = '[IMAGE DATA REMOVED]'
                elif isinstance(value, (dict, list)):
                    self._clean_base64_from_json(value)
        elif isinstance(json_data, list):
            for item in json_data:
                if isinstance(item, (dict, list)):
                    self._clean_base64_from_json(item)
        return json_data