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

from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.datamodel.base_models import InputFormat, ConversionStatus
from docling.datamodel.document import ConversionResult
from docling_core.types.doc import PictureItem

from app.config import settings
from app.services.s3_service import S3Service

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.s3_service = S3Service()
        self.executor = ThreadPoolExecutor()
        
        # Initialize document converter with your options
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

    async def process_document(self, file_path: str, output_prefix: str = None) -> Tuple[str, str]:
        """Process a document and return paths to generated markdown and JSON files"""
        
        # Create unique processing directory
        processing_dir = Path(settings.PROCESSING_DIR) / str(uuid.uuid4())
        processing_dir.mkdir(parents=True, exist_ok=True)
        
        try:
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
            md_path, json_path = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._export_and_upload_results,
                conv_results,
                processing_dir,
                file_path,
                output_prefix
            )
            
            return md_path, json_path
            
        finally:
            # Cleanup
            if processing_dir.exists():
                shutil.rmtree(processing_dir)

    def _convert_document(self, input_file: Path) -> list[ConversionResult]:
        """Convert document using Docling"""
        return list(self.doc_converter.convert_all(
            [input_file],
            raises_on_error=False,
        ))

    def _export_and_upload_results(
        self,
        conv_results: list[ConversionResult],
        processing_dir: Path,
        input_path: str,
        output_prefix: str = None
    ) -> Tuple[str, str]:
        """Export results and upload to S3"""
        output_dir = processing_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        s3_prefix = self.s3_service.get_output_prefix(input_path, output_prefix)
        md_path = None
        json_path = None
        
        for conv_res in conv_results:
            if conv_res.status != ConversionStatus.SUCCESS:
                logger.warning(f"Document {conv_res.input.file} failed to convert or partially succeeded")
                continue
                
            doc_filename = conv_res.input.file.stem
            
            # Export and process JSON
            json_file = output_dir / f"{doc_filename}.json"
            with json_file.open("w") as fp:
                json_data = conv_res.document.export_to_dict()
                json.dump(json_data, fp)
            
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
                        f"![Image {picture_counter}]({s3_prefix}/{doc_filename}-picture-{picture_counter}.png)\n\n**AI Description**: {description}\n",
                        1
                    )
            
            # Update JSON with descriptions
            for picture_ref, description in descriptions:
                for picture in json_data.get("pictures", []):
                    if picture["self_ref"] == picture_ref:
                        picture["annotations"].append(description)
                        break
            
            # Save updated JSON
            with json_file.open("w") as fp:
                json.dump(json_data, fp, indent=4)
            
            # Save markdown
            md_file = output_dir / f"{doc_filename}.md"
            with md_file.open("w") as fp:
                fp.write(markdown_content)
            
            # Upload files to S3
            s3_md_path = f"{s3_prefix}/{doc_filename}.md"
            s3_json_path = f"{s3_prefix}/{doc_filename}.json"
            
            self.s3_service.upload_file(md_file, s3_md_path)
            self.s3_service.upload_file(json_file, s3_json_path)
            
            md_path = s3_md_path
            json_path = s3_json_path
        
        return md_path, json_path

    def _get_image_description(self, image_path: Path) -> str:
        """Get AI description for image"""
        base64_image = self._encode_image(image_path)
        
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What is in this image?",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content

    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
