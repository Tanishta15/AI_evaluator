"""
AI Idea Evaluator - Document Processing Module
Phase 1: Document Processing & Extraction using Hybrid Approach
"""

from .main_processor import DocumentProcessor
from .config.settings import ProcessingConfig, get_config

__version__ = "1.0.0"
__all__ = ["DocumentProcessor", "ProcessingConfig", "get_config"]
