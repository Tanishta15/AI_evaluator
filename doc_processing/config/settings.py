"""
Configuration settings for AI Idea Evaluator - Phase 1
"""

from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import os

@dataclass
class ProcessingConfig:
    """Configuration class for document processing pipeline."""

    # Input/Output directories
    input_dir: Path
    output_dir: Path
    temp_dir: Path = Path("./temp")

    # File processing limits
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    max_pages_per_document: int = 100
    supported_formats: List[str] = None

    # Docling settings
    docling_ocr_enabled: bool = True
    docling_table_extraction: bool = True
    docling_tableformer_mode: str = "ACCURATE"  # or "FAST"

    # DPK settings
    dpk_chunk_size: int = 2000
    dpk_chunk_overlap: int = 200
    dpk_enable_deduplication: bool = True
    dpk_enable_quality_check: bool = True
    dpk_language_detection: bool = True

    # Logging settings
    log_level: str = "INFO"
    log_file: str = "document_processing.log"

    # Processing settings
    parallel_processing: bool = True
    max_workers: int = 4

    def __post_init__(self):
        """Post-initialization setup."""
        if self.supported_formats is None:
            self.supported_formats = ['.pdf', '.ppt', '.pptx']

        # Create directories if they don't exist
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Temp directory removed - not needed for direct PPT processing

    @classmethod
    def from_env(cls) -> 'ProcessingConfig':
        """Create configuration from environment variables."""
        return cls(
            input_dir=Path(os.getenv('AI_EVALUATOR_INPUT_DIR', './input_submissions')),
            output_dir=Path(os.getenv('AI_EVALUATOR_OUTPUT_DIR', './processed_output')),
            temp_dir=Path(os.getenv('AI_EVALUATOR_TEMP_DIR', './temp')),
            max_file_size=int(os.getenv('AI_EVALUATOR_MAX_FILE_SIZE', 50 * 1024 * 1024)),
            max_pages_per_document=int(os.getenv('AI_EVALUATOR_MAX_PAGES', 100)),
            parallel_processing=os.getenv('AI_EVALUATOR_PARALLEL', 'true').lower() == 'true',
            max_workers=int(os.getenv('AI_EVALUATOR_MAX_WORKERS', 4)),
            log_level=os.getenv('AI_EVALUATOR_LOG_LEVEL', 'INFO')
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'input_dir': str(self.input_dir),
            'output_dir': str(self.output_dir),
            'temp_dir': str(self.temp_dir),
            'max_file_size': self.max_file_size,
            'max_pages_per_document': self.max_pages_per_document,
            'supported_formats': self.supported_formats,
            'docling_settings': {
                'ocr_enabled': self.docling_ocr_enabled,
                'table_extraction': self.docling_table_extraction,
                'tableformer_mode': self.docling_tableformer_mode
            },
            'dpk_settings': {
                'chunk_size': self.dpk_chunk_size,
                'chunk_overlap': self.dpk_chunk_overlap,
                'enable_deduplication': self.dpk_enable_deduplication,
                'enable_quality_check': self.dpk_enable_quality_check,
                'language_detection': self.dpk_language_detection
            },
            'processing_settings': {
                'parallel_processing': self.parallel_processing,
                'max_workers': self.max_workers
            }
        }

# Environment-specific configurations
class DevelopmentConfig(ProcessingConfig):
    """Development environment configuration."""

    def __init__(self):
        super().__init__(
            input_dir=Path("./dev_input"),
            output_dir=Path("./dev_output"),
            max_file_size=10 * 1024 * 1024,  # 10MB for dev
            max_pages_per_document=20,
            log_level="DEBUG",
            max_workers=2
        )

class ProductionConfig(ProcessingConfig):
    """Production environment configuration."""

    def __init__(self):
        # Use relative paths for production as well to avoid permission issues
        super().__init__(
            input_dir=Path("./prod_input"),
            output_dir=Path("./prod_output"),
            temp_dir=Path("./prod_temp"),
            max_file_size=100 * 1024 * 1024,  # 100MB for production
            max_pages_per_document=200,
            log_level="WARNING",
            max_workers=8
        )

def get_config(environment: str = "development") -> ProcessingConfig:
    """Get configuration based on environment."""

    configs = {
        "development": DevelopmentConfig,
        "production": ProductionConfig
    }

    if environment in configs:
        return configs[environment]()
    else:
        # Return default configuration from environment variables
        return ProcessingConfig.from_env()
