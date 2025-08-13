"""
IBM Data Prep Kit Preprocessor for AI Idea Evaluator - Phase 1
Advanced preprocessing using IBM Data Prep Kit transforms
"""

import json
import logging
import subprocess
import sys
import importlib
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd
from datetime import datetime

# IBM Data Prep Kit imports - Updated for correct package structure
try:
    # Import core DPK framework components
    from data_processing.transform import (
        AbstractBinaryTransform, 
        AbstractTableTransform,
        TransformConfiguration
    )
    from data_processing.utils import get_logger as dpk_get_logger, TransformUtils
    from data_processing.data_access import DataAccessFactory
    from data_processing.runtime.pure_python import PythonTransformLauncher
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    # Create custom transforms for text processing
    class TextDeduplicationTransform(AbstractTableTransform):
        """DPK-based text deduplication transform."""
        
        def __init__(self, config: dict):
            super().__init__(config)
            self.similarity_threshold = config.get('similarity_threshold', 0.9)
            
        def transform(self, table: pa.Table) -> tuple[list[pa.Table], dict]:
            """Apply deduplication to text data."""
            try:
                df = table.to_pandas()
                
                # Simple hash-based deduplication for exact matches
                initial_count = len(df)
                df_deduped = df.drop_duplicates(subset=['text'], keep='first')
                
                # Calculate similarity-based deduplication for near-duplicates
                if len(df_deduped) > 1 and 'text' in df_deduped.columns:
                    # Simple length-based similarity check
                    lengths = df_deduped['text'].str.len()
                    mean_length = lengths.mean()
                    std_length = lengths.std()
                    
                    # Remove texts that are too short or too long (outliers)
                    df_deduped = df_deduped[
                        (lengths >= mean_length - 2 * std_length) & 
                        (lengths <= mean_length + 2 * std_length)
                    ]
                
                duplicates_removed = initial_count - len(df_deduped)
                stats = {"duplicates_removed": duplicates_removed}
                
                return [pa.Table.from_pandas(df_deduped)], stats
                
            except Exception as e:
                self.logger.error(f"Deduplication failed: {e}")
                return [table], {"duplicates_removed": 0}
    
    class TextQualityFilterTransform(AbstractTableTransform):
        """DPK-based text quality filtering transform."""
        
        def __init__(self, config: dict):
            super().__init__(config)
            self.min_length = config.get('min_length', 10)
            self.max_length = config.get('max_length', 10000)
            self.allowed_languages = config.get('languages', ['en'])
            
        def transform(self, table: pa.Table) -> tuple[list[pa.Table], dict]:
            """Apply quality filtering to text data."""
            try:
                df = table.to_pandas()
                initial_count = len(df)
                
                # Apply length filters
                if 'text' in df.columns:
                    df = df[
                        (df['text'].str.len() >= self.min_length) & 
                        (df['text'].str.len() <= self.max_length)
                    ]
                
                # Apply language filter if language column exists
                if 'language' in df.columns:
                    df = df[df['language'].isin(self.allowed_languages)]
                
                filtered_count = initial_count - len(df)
                stats = {"filtered_count": filtered_count}
                
                return [pa.Table.from_pandas(df)], stats
                
            except Exception as e:
                self.logger.error(f"Quality filtering failed: {e}")
                return [table], {"filtered_count": 0}
    
    # Set up transform classes
    DedupeTransform = TextDeduplicationTransform
    FilterTransform = TextQualityFilterTransform
    
    DPK_AVAILABLE = True
    logging.info("✅ IBM Data Prep Kit framework loaded successfully with custom transforms")
    get_logger = dpk_get_logger
    
except ImportError as e:
    # Complete fallback when DPK is not available
    DPK_AVAILABLE = False
    logging.warning(f"⚠️  IBM Data Prep Kit not available: {e}. Using fallback preprocessing.")

    DedupeTransform = None
    FilterTransform = None
    
    # Fallback logger
    def get_logger(name):
        return logging.getLogger(name)

class DPKPreprocessor:
    """
    IBM Data Prep Kit preprocessor for advanced document preprocessing.
    Provides sophisticated data cleaning, transformation, and structuring.
    """

    def __init__(self, config):
        """Initialize DPK preprocessor with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.dpk_available = DPK_AVAILABLE

        # Processing statistics
        self.processing_stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'duplicates_removed': 0,
            'language_detections': 0,
            'quality_filters_applied': 0
        }

        if self.dpk_available:
            self._setup_dpk_transforms()
        else:
            self.logger.warning("⚠️  DPK not available, using fallback methods")

    def _setup_dpk_transforms(self) -> None:
        """Setup IBM Data Prep Kit transforms."""
        try:
            if DedupeTransform is not None:
                self.dedupe_config = {
                    "similarity_threshold": 0.9
                }
                self.dedupe_transform = DedupeTransform(self.dedupe_config)
            else:
                self.dedupe_transform = None

            # Quality filter transform (if available)
            if FilterTransform is not None:
                self.filter_config = {
                    "min_length": 50,        # Minimum text length
                    "max_length": 10000,     # Maximum text length
                    "languages": ["en"]      # Only English content
                }
                self.filter_transform = FilterTransform(self.filter_config)
            else:
                self.filter_transform = None

            self.logger.info("✅ DPK transforms initialized successfully")

        except Exception as e:
            self.logger.error(f"❌ Failed to setup DPK transforms: {str(e)}")
            self.dpk_available = False

    def process_extracted_content(self, docling_document, raw_text: str) -> Dict[str, Any]:
        """
        Process extracted content using DPK transforms.

        Args:
            docling_document: Docling document object
            raw_text: Raw extracted text

        Returns:
            Dictionary containing processed results
        """
        try:
            self.logger.info("🔄 Starting DPK preprocessing...")

            # Convert content to structured format
            structured_data = self._structure_content(docling_document, raw_text)

            if self.dpk_available:
                # Apply DPK transforms
                processed_data = self._apply_dpk_transforms(structured_data)
            else:
                # Apply fallback processing
                processed_data = self._apply_fallback_processing(structured_data)

            # Generate chunks for downstream processing
            chunks = self._create_chunks(processed_data)

            # Update statistics
            self.processing_stats['documents_processed'] += 1
            self.processing_stats['chunks_created'] += len(chunks)

            result = {
                'success': True,
                'processed_data': processed_data,
                'chunks': chunks,
                'metadata': {
                    'processing_method': 'dpk' if self.dpk_available else 'fallback',
                    'total_chunks': len(chunks),
                    'original_text_length': len(raw_text),
                    'processed_text_length': sum(len(chunk.get('text', '')) for chunk in chunks),
                    'language_detected': processed_data.get('language', 'unknown'),
                    'quality_score': processed_data.get('quality_score', 0),
                    'processing_timestamp': datetime.now().isoformat()
                }
            }

            self.logger.info(f"✅ DPK preprocessing complete. Created {len(chunks)} chunks.")
            return result

        except Exception as e:
            self.logger.error(f"❌ DPK preprocessing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'metadata': {
                    'processing_method': 'failed',
                    'processing_timestamp': datetime.now().isoformat()
                }
            }

    def _structure_content(self, docling_document, raw_text: str) -> Dict[str, Any]:
        """Structure content from Docling document."""

        structured_content = {
            'text': raw_text,
            'pages': [],
            'tables': [],
            'images': [],
            'headings': [],
            'metadata': {}
        }

        if docling_document:
            # Extract page information
            for i, page in enumerate(docling_document.pages):
                page_info = {
                    'page_number': i + 1,
                    'text': page.text if hasattr(page, 'text') else '',
                    'tables_count': len(page.tables) if hasattr(page, 'tables') else 0,
                    'images_count': len(page.images) if hasattr(page, 'images') else 0
                }
                structured_content['pages'].append(page_info)

                # Extract tables
                if hasattr(page, 'tables'):
                    for table in page.tables:
                        table_info = {
                            'page_number': i + 1,
                            'table_data': table.export_to_dict() if hasattr(table, 'export_to_dict') else str(table),
                            'rows': getattr(table, 'num_rows', 0),
                            'cols': getattr(table, 'num_cols', 0)
                        }
                        structured_content['tables'].append(table_info)

                # Extract images
                if hasattr(page, 'images'):
                    for img in page.images:
                        img_info = {
                            'page_number': i + 1,
                            'image_metadata': getattr(img, 'metadata', {})
                        }
                        structured_content['images'].append(img_info)

            # Extract headings and structure
            if hasattr(docling_document, 'headings'):
                structured_content['headings'] = [
                    {
                        'text': heading.text,
                        'level': heading.level,
                        'page': getattr(heading, 'page', 0)
                    }
                    for heading in docling_document.headings
                ]

        return structured_content

    def _apply_dpk_transforms(self, structured_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply IBM Data Prep Kit transforms using the DPK framework."""

        # Convert to DataFrame for DPK processing
        df_data = []

        # Process main text
        df_data.append({
            'text': structured_data['text'],
            'content_type': 'main_text',
            'page_number': 0
        })

        # Process page texts
        for page in structured_data['pages']:
            if page['text'].strip():
                df_data.append({
                    'text': page['text'],
                    'content_type': 'page_text',
                    'page_number': page['page_number']
                })

        # Process headings
        for heading in structured_data['headings']:
            if heading['text'].strip():
                df_data.append({
                    'text': heading['text'],
                    'content_type': 'heading',
                    'page_number': heading.get('page', 0),
                    'heading_level': heading['level']
                })

        df = pd.DataFrame(df_data)
        
        if df.empty:
            return structured_data

        try:
            # Convert to PyArrow Table for DPK processing
            table = pa.Table.from_pandas(df)
            
            # Apply language identification (simple fallback)
            df_with_lang = df.copy()
            df_with_lang['language'] = 'en'  # Default to English
            table_with_lang = pa.Table.from_pandas(df_with_lang)

            # Apply deduplication using DPK transform
            if self.dedupe_transform is not None and self.config.dpk_enable_deduplication:
                try:
                    result_tables, dedup_stats = self.dedupe_transform.transform(table_with_lang)
                    if result_tables:
                        table_deduped = result_tables[0]
                        df_deduped = table_deduped.to_pandas()
                        duplicates_removed = dedup_stats.get('duplicates_removed', 0)
                        self.processing_stats['duplicates_removed'] += duplicates_removed
                        self.logger.info(f"✅ DPK deduplication removed {duplicates_removed} duplicates")
                    else:
                        df_deduped = df_with_lang.copy()
                except Exception as e:
                    self.logger.warning(f"DPK deduplication failed: {e}")
                    df_deduped = df_with_lang.copy()
            else:
                df_deduped = df_with_lang.copy()

            # Apply quality filtering using DPK transform
            if self.filter_transform is not None and self.config.dpk_enable_quality_check:
                try:
                    table_for_filtering = pa.Table.from_pandas(df_deduped)
                    result_tables, filter_stats = self.filter_transform.transform(table_for_filtering)
                    if result_tables:
                        table_filtered = result_tables[0]
                        df_filtered = table_filtered.to_pandas()
                        filtered_count = filter_stats.get('filtered_count', 0)
                        self.processing_stats['quality_filters_applied'] += 1
                        self.logger.info(f"✅ DPK quality filter removed {filtered_count} low-quality entries")
                    else:
                        df_filtered = df_deduped.copy()
                except Exception as e:
                    self.logger.warning(f"DPK quality filtering failed: {e}")
                    df_filtered = df_deduped.copy()
            else:
                df_filtered = df_deduped.copy()

            # Update structured data with processed results
            processed_data = structured_data.copy()
            processed_data['processed_text_data'] = df_filtered.to_dict('records')
            processed_data['language'] = df_filtered['language'].iloc[0] if not df_filtered.empty else 'unknown'
            processed_data['quality_score'] = self._calculate_quality_score(df_filtered)

            self.logger.info(f"✅ DPK transforms applied successfully. Processed {len(df_filtered)} text entries.")
            return processed_data

        except Exception as e:
            self.logger.error(f"❌ Error applying DPK transforms: {str(e)}")
            return structured_data

    def _apply_language_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply language detection transform."""
        try:
            # Create Arrow table for DPK processing
            import pyarrow as pa
            table = pa.Table.from_pandas(df)

            # Apply language identification transform
            result_table, metadata = self.lang_id_transform.transform(table)

            # Convert back to DataFrame
            return result_table.to_pandas()

        except Exception as e:
            self.logger.warning(f"Language detection failed: {str(e)}")
            # Fallback: assume English
            df['language'] = 'en'
            return df

    def _apply_deduplication(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply deduplication transform."""
        try:
            # Simple deduplication based on text similarity
            df_unique = df.drop_duplicates(subset=['text'], keep='first')
            return df_unique

        except Exception as e:
            self.logger.warning(f"Deduplication failed: {str(e)}")
            return df

    def _apply_quality_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply quality filtering transform."""
        try:
            # Filter based on text length and language
            filtered_df = df[
                (df['text'].str.len() >= 10) &  # Minimum length
                (df['text'].str.len() <= 10000) &  # Maximum length
                (df['language'].isin(['en', 'unknown']))  # Language filter
            ].copy()

            return filtered_df

        except Exception as e:
            self.logger.warning(f"Quality filtering failed: {str(e)}")
            return df

    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate overall quality score for the processed data."""
        if df.empty:
            return 0.0

        # Simple quality scoring based on multiple factors
        avg_length = df['text'].str.len().mean()
        length_score = min(avg_length / 100, 1.0)  # Normalize to 0-1

        # Language consistency score
        lang_consistency = (df['language'] == 'en').sum() / len(df)

        # Content diversity score (based on unique content types)
        content_diversity = df['content_type'].nunique() / df['content_type'].nunique()

        # Overall quality score (weighted average)
        quality_score = (
            length_score * 0.4 + 
            lang_consistency * 0.4 + 
            content_diversity * 0.2
        )

        return min(quality_score, 1.0)

    def _apply_fallback_processing(self, structured_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply fallback processing when DPK is not available."""

        self.logger.info("🔄 Applying fallback preprocessing (no DPK)")

        processed_data = structured_data.copy()

        # Basic text cleaning
        text = structured_data.get('text', '')
        cleaned_text = self._clean_text_fallback(text)

        # Create processed text data
        processed_text_data = [{
            'text': cleaned_text,
            'content_type': 'main_text',
            'page_number': 0,
            'language': 'en'  # Assume English
        }]

        # Add page texts
        for page in structured_data.get('pages', []):
            if page.get('text', '').strip():
                cleaned_page_text = self._clean_text_fallback(page['text'])
                processed_text_data.append({
                    'text': cleaned_page_text,
                    'content_type': 'page_text',
                    'page_number': page['page_number'],
                    'language': 'en'
                })

        processed_data['processed_text_data'] = processed_text_data
        processed_data['language'] = 'en'
        processed_data['quality_score'] = 0.7  # Default quality score

        return processed_data

    def _clean_text_fallback(self, text: str) -> str:
        """Basic text cleaning fallback method."""
        if not text:
            return ""

        import re
        
        # Start with the original text
        cleaned = text
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove special characters but keep basic punctuation
        cleaned = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', cleaned)
        
        # Remove multiple spaces again after character cleaning
        cleaned = re.sub(r'\s+', ' ', cleaned)

        return cleaned.strip()

    def _create_chunks(self, processed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create text chunks for downstream processing."""

        chunks = []
        processed_text_data = processed_data.get('processed_text_data', [])

        for item in processed_text_data:
            text = item.get('text', '')

            if len(text) <= self.config.dpk_chunk_size:
                # Text is small enough, create single chunk
                chunks.append({
                    'text': text,
                    'chunk_id': len(chunks),
                    'content_type': item.get('content_type', 'text'),
                    'page_number': item.get('page_number', 0),
                    'language': item.get('language', 'en'),
                    'start_pos': 0,
                    'end_pos': len(text)
                })
            else:
                # Split into multiple chunks
                chunk_size = self.config.dpk_chunk_size
                overlap = self.config.dpk_chunk_overlap

                for i in range(0, len(text), chunk_size - overlap):
                    chunk_text = text[i:i + chunk_size]

                    if len(chunk_text.strip()) < 20:  # Skip very short chunks
                        continue

                    chunks.append({
                        'text': chunk_text,
                        'chunk_id': len(chunks),
                        'content_type': item.get('content_type', 'text'),
                        'page_number': item.get('page_number', 0),
                        'language': item.get('language', 'en'),
                        'start_pos': i,
                        'end_pos': i + len(chunk_text),
                        'is_partial': True
                    })

        return chunks

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'dpk_available': self.dpk_available,
            'processing_stats': self.processing_stats.copy(),
            'transforms_used': [
                'language_identification',
                'deduplication', 
                'quality_filtering',
                'chunking'
            ] if self.dpk_available else ['fallback_processing', 'chunking']
        }

    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self.processing_stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'duplicates_removed': 0,
            'language_detections': 0,
            'quality_filters_applied': 0
        }
