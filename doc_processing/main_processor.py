"""
AI Idea Evaluator - Phase 1: Document Processing & Extraction
Main Document Processor using Hybrid Approach (Docling + IBM Data Prep Kit)
"""

import os
import json
import logging
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

# Docling imports
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    DOCLING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️  Docling not available: {e}")
    DOCLING_AVAILABLE = False

# Chunking feature is not available in this Docling installation.
CHUNKING_AVAILABLE = False
HierarchicalChunker = None

# IBM Data Prep Kit imports - Updated for correct package structure
try:
    # Try to import from the local data_processing module first
    from data_processing.transform import AbstractBinaryTransform, TransformConfiguration
    from data_processing.utils import get_logger as dpk_get_logger
    DPK_AVAILABLE = True
    logging.info("✅ IBM Data Prep Kit (local) loaded successfully")
    get_logger = dpk_get_logger
except ImportError:
    try:
        # Fallback to individual transform modules if available
        import pdf2parquet_transform_python
        DPK_AVAILABLE = True
        logging.info("✅ IBM Data Prep Kit (transforms) loaded successfully")
        get_logger = lambda name: logging.getLogger(name)
    except ImportError:
        DPK_AVAILABLE = False
        logging.warning("⚠️  IBM Data Prep Kit not available. Using fallback processing.")
        # Fallback logger
        def get_logger(name):
            return logging.getLogger(name)

# Local imports
from .utils.file_handler import FileHandler
from .utils.enhanced_content_extractor import EnhancedContentExtractor
from .utils.dpk_preprocessor import DPKPreprocessor
from .utils.embedding_generator import GraniteEmbeddingGenerator
from .utils.vector_storage import VectorStorageManager
from .config.settings import ProcessingConfig

class DocumentProcessor:
    """
    Main document processor that combines Docling for extraction 
    with IBM Data Prep Kit for advanced preprocessing.
    """

    def __init__(self, config: ProcessingConfig):
        """Initialize the document processor with configuration."""
        self.config = config
        self.logger = get_logger(__name__)

        # Initialize Docling converter with optimized settings
        self.docling_converter = None
        self.chunker = None
        self._setup_docling_converter()

        # Initialize DPK preprocessor
        self.dpk_preprocessor = DPKPreprocessor(config)

        # Initialize file handler
        self.file_handler = FileHandler(config)

        # Initialize enhanced content extractor
        self.content_extractor = EnhancedContentExtractor(config)
        
        # Initialize embedding generator
        self.embedding_generator = GraniteEmbeddingGenerator()
        
        # Initialize vector storage (default to ChromaDB)
        try:
            self.vector_storage = VectorStorageManager(storage_type="chromadb")
        except ImportError as e:
            self.logger.warning(f"Vector storage not available: {e}")
            self.vector_storage = None

        # Processing statistics
        self.stats = {
            'total_files': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'processing_times': [],
            'start_time': None,
            'end_time': None,
            'embeddings_generated': 0,
            'vectors_stored': 0
        }

    def _setup_docling_converter(self) -> None:
        """Setup Docling document converter with optimized pipeline options."""
        
        if not DOCLING_AVAILABLE:
            self.logger.warning("⚠️  Docling not available. Document conversion will be limited.")
            self.docling_converter = None
            return

        try:
            # Configure PDF pipeline options for better extraction
            pipeline_options = PdfPipelineOptions(
                do_table_structure=True,  # Enable table structure recognition
                do_ocr=True,             # Enable OCR for scanned documents
            )

            # Use accurate TableFormer mode for better table parsing
            pipeline_options.table_structure_options.mode = "ACCURATE"
            pipeline_options.table_structure_options.do_cell_matching = True

            # Create converter with PDF format options
            format_options = {
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }

            self.docling_converter = DocumentConverter(
                format_options=format_options
            )

            # Set up chunker if available
            if CHUNKING_AVAILABLE:
                self.chunker = HierarchicalChunker()
            else:
                self.chunker = None

            self.logger.info("✅ Docling converter initialized with optimized settings")
            
            # Chunker is not available in this Docling installation.
            self.chunker = None
        except Exception as e:
            self.logger.error(f"❌ Error initializing Docling converter: {str(e)}")
            self.docling_converter = None
            self.chunker = None
    def process_submission_batch(self, submission_paths: List[str]) -> Dict[str, Any]:
        """
        Process a batch of idea submissions (PPT/PDF files).

        Args:
            submission_paths: List of file paths to process

        Returns:
            Dictionary containing processing results and metadata
        """
        self.stats['start_time'] = datetime.now()
        self.stats['total_files'] = len(submission_paths)

        batch_results = {
            'processed_files': [],
            'failed_files': [],
            'extraction_metadata': {},
            'processing_stats': {}
        }

        self.logger.info(f"🚀 Starting batch processing of {len(submission_paths)} submissions")

        for file_path in submission_paths:
            try:
                # Process individual file
                result = self.process_single_file(file_path)

                if result['success']:
                    batch_results['processed_files'].append(result)
                    self.stats['successful_extractions'] += 1
                else:
                    batch_results['failed_files'].append({
                        'file_path': file_path,
                        'error': result.get('error', 'Unknown error')
                    })
                    self.stats['failed_extractions'] += 1

                # Store extraction metadata
                batch_results['extraction_metadata'][file_path] = result.get('metadata', {})

            except Exception as e:
                self.logger.error(f"❌ Error processing {file_path}: {str(e)}")
                batch_results['failed_files'].append({
                    'file_path': file_path,
                    'error': str(e)
                })
                self.stats['failed_extractions'] += 1

        self.stats['end_time'] = datetime.now()
        batch_results['processing_stats'] = self._calculate_stats()

        # Save batch results
        self._save_batch_results(batch_results)

        self.logger.info(f"✅ Batch processing complete. Success: {self.stats['successful_extractions']}, Failed: {self.stats['failed_extractions']}")

        return batch_results

    def process_single_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single PPT/PDF file through the hybrid pipeline.

        Args:
            file_path: Path to the file to process

        Returns:
            Dictionary containing extraction results and metadata
        """
        start_time = datetime.now()

        try:
            # Step 1: Choose processing method based on file type
            file_extension = Path(file_path).suffix.lower()
            
            # Step 2: Extract content using optimal method
            if file_extension in ['.ppt', '.pptx']:
                # Direct PPT processing for better image/table extraction
                self.logger.info(f"📄 Processing PPT directly with Docling for visual content: {Path(file_path).name}")
                docling_result = self._extract_with_docling_direct(file_path)
            else:
                # Traditional PDF processing
                processed_path = self.file_handler.prepare_file_for_extraction(file_path)
                self.logger.info(f"📄 Extracting content from {Path(processed_path).name} using Docling...")
                docling_result = self._extract_with_docling(processed_path)

            if not docling_result['success']:
                return {
                    'success': False,
                    'error': f"Docling extraction failed: {docling_result['error']}",
                    'file_path': file_path
                }

            # Step 2.5: Apply IBM Data Prep Kit preprocessing
            if DPK_AVAILABLE and self.dpk_preprocessor:
                try:
                    self.logger.info("🔄 Applying IBM Data Prep Kit preprocessing...")
                    dpk_result = self.dpk_preprocessor.process_extracted_content(
                        docling_result['document'],
                        docling_result['raw_text']
                    )
                    
                    if dpk_result['success']:
                        # Update raw text with processed content
                        processed_chunks = dpk_result.get('chunks', [])
                        if processed_chunks:
                            # Combine processed chunks back into text
                            processed_text = '\n\n'.join([chunk.get('text', '') for chunk in processed_chunks])
                            docling_result['raw_text'] = processed_text
                            self.logger.info(f"✅ DPK preprocessing completed. Generated {len(processed_chunks)} chunks.")
                        
                        # Add DPK metadata
                        docling_result['metadata']['dpk_processing'] = dpk_result.get('metadata', {})
                    else:
                        self.logger.warning(f"⚠️ DPK preprocessing failed: {dpk_result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ DPK preprocessing error: {e}")
            else:
                self.logger.info("📝 Skipping DPK preprocessing (not available or disabled)")

            # Step 3: Extract and structure content using enhanced extractor
            content_result = self.content_extractor.process_document(
                docling_result['document'],
                docling_result['raw_text'],
                docling_result['metadata'],
                file_path
            )

            if not content_result['success']:
                return {
                    'success': False,
                    'error': f"Content extraction failed: {content_result['error']}",
                    'file_path': file_path
                }

            # Step 4: Generate embeddings and store in vector database
            embeddings_result = None
            if self.embedding_generator and self.vector_storage:
                try:
                    embeddings_result = self._generate_and_store_embeddings(
                        content_result,
                        file_path
                    )
                except Exception as e:
                    self.logger.warning(f"⚠️ Embedding generation failed: {e}")
                    embeddings_result = {'success': False, 'error': str(e)}

            # Step 5: Generate final output
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats['processing_times'].append(processing_time)

            result = {
                'success': True,
                'file_path': file_path,
                'original_filename': Path(file_path).name,
                'parquet_path': content_result['parquet_path'],
                'total_pages': content_result['total_pages'],
                'total_slides': content_result['total_slides'],
                'images_present': content_result['images_present'],
                'raw_text': docling_result['raw_text'],
                'document_metadata': docling_result['metadata'],
                'processing_time_seconds': processing_time,
                'extraction_timestamp': datetime.now().isoformat(),
                'embeddings_result': embeddings_result
            }

            # Step 5: Save individual file results (now in Parquet format)
            # The Parquet file is already saved by the enhanced content extractor
            self.logger.info(f"✅ Results saved to Parquet: {content_result['parquet_path']}")

            return result

        except Exception as e:
            self.logger.error(f"❌ Error processing {file_path}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'file_path': file_path,
                'processing_time_seconds': (datetime.now() - start_time).total_seconds()
            }

    def _extract_with_docling_direct(self, file_path: str) -> Dict[str, Any]:
        """
        Extract content directly from PPT files using Docling for optimal image/table preservation.
        
        Args:
            file_path: Path to the PPT file
            
        Returns:
            Dictionary containing extracted content and metadata
        """
        try:
            # Convert directly with Docling (preserves images and tables)
            result = self.docling_converter.convert(file_path)
            
            # Extract raw text
            raw_text = result.document.export_to_text()
            
            # Count visual elements
            tables = getattr(result.document, 'tables', [])
            pictures = getattr(result.document, 'pictures', [])
            
            # Build metadata
            metadata = {
                'file_path': file_path,
                'file_type': 'ppt_direct',
                'page_count': len(getattr(result.document, 'pages', {})),
                'has_images': len(pictures) > 0,
                'has_tables': len(tables) > 0,
                'images_count': len(pictures),
                'tables_count': len(tables),
                'text_length': len(raw_text),
                'processing_method': 'docling_direct'
            }
            
            self.logger.info(f"✅ Direct PPT processing: {len(tables)} tables, {len(pictures)} images, {len(raw_text)} chars")
            
            return {
                'success': True,
                'document': result.document,
                'raw_text': raw_text,
                'metadata': metadata
            }
            
        except Exception as e:
            self.logger.error(f"❌ Direct PPT processing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'document': None,
                'raw_text': "",
                'metadata': {}
            }

    def _extract_with_docling(self, file_path: str) -> Dict[str, Any]:
        """Extract content from PDF using Docling."""
        try:
            self.logger.info(f"📄 Extracting content from {file_path} using Docling...")

            # Check if Docling converter is available
            if self.docling_converter is None:
                return self._extract_fallback(file_path)

            # Convert document with Docling
            conversion_result = self.docling_converter.convert(
                source=file_path,
                max_file_size=self.config.max_file_size,
                max_num_pages=self.config.max_pages_per_document
            )

            # Extract text content
            markdown_content = conversion_result.document.export_to_markdown()
            json_content = conversion_result.document.export_to_dict()

            # Extract metadata with proper error handling
            document_pages = conversion_result.document.pages if hasattr(conversion_result.document, 'pages') else []
            
            # Safely check for tables and images
            has_tables = False
            has_images = False
            
            for page in document_pages:
                if hasattr(page, 'tables') and page.tables:
                    has_tables = True
                if hasattr(page, 'images') and page.images:
                    has_images = True
            
            metadata = {
                'page_count': len(document_pages),
                'has_tables': has_tables,
                'has_images': has_images,
                'extraction_method': 'docling',
                'pipeline_options': {
                    'table_structure': True,
                    'ocr_enabled': True,
                    'tableformer_mode': 'ACCURATE'
                }
            }

            return {
                'success': True,
                'document': conversion_result.document,
                'raw_text': markdown_content,
                'json_structure': json_content,
                'metadata': metadata,
                'total_pages': len(document_pages)
            }

        except Exception as e:
            self.logger.error(f"❌ Docling extraction failed for {file_path}: {str(e)}")
            # Don't use fallback - force Docling to work
            return {
                'success': False,
                'error': f"Docling extraction failed: {str(e)}",
                'document': None,
                'raw_text': '',
                'json_structure': {},
                'metadata': {'extraction_method': 'failed'},
                'total_pages': 0
            }

    def _calculate_stats(self) -> Dict[str, Any]:
        """Calculate processing statistics."""
        total_time = (self.stats['end_time'] - self.stats['start_time']).total_seconds()

        return {
            'total_files': self.stats['total_files'],
            'successful_extractions': self.stats['successful_extractions'],
            'failed_extractions': self.stats['failed_extractions'],
            'success_rate': self.stats['successful_extractions'] / self.stats['total_files'] if self.stats['total_files'] > 0 else 0,
            'total_processing_time_seconds': total_time,
            'average_processing_time_per_file': sum(self.stats['processing_times']) / len(self.stats['processing_times']) if self.stats['processing_times'] else 0,
            'files_per_second': self.stats['total_files'] / total_time if total_time > 0 else 0
        }

    def _save_batch_results(self, results: Dict[str, Any]) -> None:
        """Save batch processing results to JSON."""
        output_path = self.config.output_dir / f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        self.logger.info(f"💾 Batch results saved to {output_path}")

    def _generate_and_store_embeddings(self, content_result: Dict[str, Any], 
                                      file_path: str) -> Dict[str, Any]:
        """
        Generate embeddings for extracted content and store in vector database.
        
        Args:
            content_result: Result from enhanced content extractor
            file_path: Original file path
            
        Returns:
            Dictionary with embedding generation results
        """
        try:
            self.logger.info("🔄 Generating embeddings and storing in vector database...")
            
            # Load the parquet data to get slide content
            parquet_path = content_result['parquet_path']
            df = pd.read_parquet(parquet_path)
            
            # Prepare data for embedding generation
            slides_data = []
            document_id = str(uuid.uuid4())
            filename = Path(file_path).name
            
            for index, row in df.iterrows():
                slide_data = {
                    'slide_number': row.get('slide_number', index + 1),
                    'title': row.get('title', ''),
                    'content': row.get('content', ''),
                    'total_pages': row.get('total_pages', 0),
                    'images_present': row.get('images_present', False)
                }
                slides_data.append(slide_data)
            
            # Generate embeddings for slides
            slides_with_embeddings = self.embedding_generator.generate_slide_embeddings(slides_data)
            
            # Prepare data for vector storage
            embeddings_data = []
            for slide in slides_with_embeddings:
                embedding_data = {
                    'slide_number': slide['slide_number'],
                    'title': slide['title'],
                    'content': slide['content'],
                    'total_pages': slide['total_pages'],
                    'images_present': slide['images_present'],
                    'embedding': slide['embedding'],
                    'embedding_model': slide['embedding_model'],
                    'processing_timestamp': datetime.now().isoformat(),
                    'document_id': document_id,
                    'filename': filename
                }
                embeddings_data.append(embedding_data)
            
            # Store in vector database
            storage_success = self.vector_storage.store_document_embeddings(
                embeddings_data=embeddings_data,
                document_id=document_id,
                filename=filename
            )
            
            if storage_success:
                self.stats['embeddings_generated'] += len(embeddings_data)
                self.stats['vectors_stored'] += len(embeddings_data)
                
                self.logger.info(f"✅ Generated and stored {len(embeddings_data)} embeddings")
                
                return {
                    'success': True,
                    'embeddings_count': len(embeddings_data),
                    'document_id': document_id,
                    'vector_storage_type': self.vector_storage.storage_type
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to store embeddings in vector database'
                }
                
        except Exception as e:
            self.logger.error(f"❌ Error generating embeddings: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def search_similar_content(self, query_text: str, k: int = 5, 
                             criteria: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar content in the vector database.
        
        Args:
            query_text: Text query to search for
            k: Number of results to return
            criteria: Optional criteria to filter by
            
        Returns:
            List of similar content with metadata
        """
        try:
            if not self.vector_storage:
                return []
            
            # Generate embedding for query
            query_embedding = self.embedding_generator.generate_embeddings([query_text])[0]
            
            # Search in vector database
            results = self.vector_storage.search_similar_content(
                query_embedding=query_embedding,
                criteria=criteria,
                k=k
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error searching similar content: {e}")
            return []

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        stats = self.stats.copy()
        
        if self.vector_storage:
            vector_stats = self.vector_storage.get_stats()
            stats.update(vector_stats)
        
        return stats

    def _save_individual_result(self, result: Dict[str, Any]) -> None:
        """Save individual file processing result - disabled to reduce directory creation."""
        # Individual result saving disabled to keep clean directory structure
        filename = Path(result['file_path']).stem
        self.logger.debug(f"💾 Individual result for {filename} (not saved to reduce clutter)")
        pass
