import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from doc_processing.main_processor import DocumentProcessor
from doc_processing.config.settings import ProcessingConfig

class AIEvaluatorPipeline:
    """Complete AI Idea Evaluator Pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the pipeline with configuration."""
        self.config = config
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize processing configuration
        self.processing_config = ProcessingConfig(
            input_dir=Path(config['input_directory']),
            output_dir=Path(config['output_directory']),
            max_file_size=config.get('max_file_size', 50 * 1024 * 1024),  # 50MB default
            max_pages_per_document=config.get('max_pages', 25),
            supported_formats=config.get('supported_formats', ['.pdf', '.ppt', '.pptx'])
        )
        
        # Initialize document processor
        self.processor = DocumentProcessor(self.processing_config)
        
        # Pipeline statistics
        self.stats = {
            'start_time': None,
            'end_time': None,
            'total_files': 0,
            'successful_files': 0,
            'failed_files': 0,
            'total_slides': 0,
            'total_embeddings': 0,
            'processing_times': [],
            'errors': []
        }
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = self.config.get('log_level', 'INFO').upper()
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Setup logging (console only)
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format,
            handlers=[
                logging.StreamHandler()
            ]
        )
    
    def discover_input_files(self) -> List[Path]:
        """Discover all supported files in the input directory."""
        input_dir = Path(self.config['input_directory'])
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        supported_extensions = self.processing_config.supported_formats
        discovered_files = []
        
        self.logger.info(f"🔍 Discovering files in: {input_dir}")
        
        for ext in supported_extensions:
            pattern = f"**/*{ext}"
            files = list(input_dir.glob(pattern))
            discovered_files.extend(files)
            self.logger.info(f"Found {len(files)} {ext} files")
        
        # Remove duplicates and sort
        discovered_files = sorted(set(discovered_files))
        
        self.logger.info(f"📁 Total files discovered: {len(discovered_files)}")
        return discovered_files
    
    def process_single_document(self, file_path: Path) -> Dict[str, Any]:
        """Process a single document through the complete pipeline."""
        start_time = time.time()
        
        try:
            self.logger.info(f"🔄 Processing: {file_path.name}")
            
            # Process the document
            result = self.processor.process_single_file(str(file_path))
            
            # Add OCR processing for technical architecture images
            if result['success'] and result.get('parquet_path'):
                try:
                    from image import extract_text_from_keyword_image
                    
                    # List of keywords to search for technical architecture slides
                    architecture_keywords = [
                        'architecture',
                        'blueprint',
                        'system diagram',
                        'technical overview',
                        'system architecture'
                    ]
                    
                    # Try each keyword until we find a match
                    for keyword in architecture_keywords:
                        self.logger.info(f"🔍 Searching for '{keyword}' in slides...")
                        ocr_text = extract_text_from_keyword_image(
                            parquet_path=result['parquet_path'],
                            keyword=keyword,
                            image_base_dir='.',  # Use project root as base directory
                            save_changes=True
                        )
                        if ocr_text:
                            self.logger.info(f"✅ Found and processed image for '{keyword}'")
                            break
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ OCR processing failed: {e}")
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats['processing_times'].append(processing_time)
            
            if result['success']:
                self.stats['successful_files'] += 1
                self.stats['total_slides'] += result.get('total_slides', 0)
                
                # Count embeddings if generated
                embeddings_result = result.get('embeddings_result')
                if embeddings_result and embeddings_result.get('success'):
                    self.stats['total_embeddings'] += embeddings_result.get('embeddings_count', 0)
                
                self.logger.info(f"Successfully processed: {file_path.name}")
                self.logger.info(f"Slides: {result.get('total_slides', 0)}")
                self.logger.info(f"Time: {processing_time:.2f}s")
                
                if embeddings_result and embeddings_result.get('success'):
                    self.logger.info(f"Embeddings: {embeddings_result.get('embeddings_count', 0)}")
            else:
                self.stats['failed_files'] += 1
                error_msg = result.get('error', 'Unknown error')
                self.stats['errors'].append({
                    'file': str(file_path),
                    'error': error_msg,
                    'timestamp': datetime.now().isoformat()
                })
                self.logger.error(f"Failed to process: {file_path.name} - {error_msg}")
            
            return result
            
        except Exception as e:
            self.stats['failed_files'] += 1
            error_msg = str(e)
            self.stats['errors'].append({
                'file': str(file_path),
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            })
            self.logger.error(f"Exception processing {file_path.name}: {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'file_path': str(file_path)
            }
    
    def process_batch(self, file_paths: List[Path]) -> List[Dict[str, Any]]:
        """Process a batch of documents."""
        results = []
        
        self.logger.info(f"Starting batch processing of {len(file_paths)} files")
        
        for i, file_path in enumerate(file_paths, 1):
            self.logger.info(f"Progress: {i}/{len(file_paths)} ({i/len(file_paths)*100:.1f}%)")
            
            result = self.process_single_document(file_path)
            results.append(result)
            
            # Add delay if specified
            if self.config.get('processing_delay', 0) > 0:
                time.sleep(self.config['processing_delay'])
        
        return results
    
    def generate_summary_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a comprehensive summary report."""
        total_time = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        # Generate summary
        summary = {
            'pipeline_info': {
                'execution_time': total_time,
                'start_time': self.stats['start_time'].isoformat(),
                'end_time': self.stats['end_time'].isoformat(),
                'config': self.config
            },
            'processing_statistics': {
                'total_files': self.stats['total_files'],
                'successful_files': self.stats['successful_files'],
                'failed_files': self.stats['failed_files'],
                'success_rate': self.stats['successful_files'] / max(self.stats['total_files'], 1) * 100,
                'total_slides': self.stats['total_slides'],
                'total_embeddings': self.stats['total_embeddings'],
                'average_processing_time': sum(self.stats['processing_times']) / max(len(self.stats['processing_times']), 1),
                'total_processing_time': sum(self.stats['processing_times'])
            },
            'errors': self.stats['errors'],
            'file_results': [
                {
                    'filename': Path(result['file_path']).name,
                    'success': result['success'],
                    'slides': result.get('total_slides', 0),
                    'embeddings': result.get('embeddings_result', {}).get('embeddings_count', 0),
                    'processing_time': result.get('processing_time_seconds', 0),
                    'error': result.get('error') if not result['success'] else None
                }
                for result in results
            ]
        }
        
        return summary
    
    def run_pipeline(self) -> Dict[str, Any]:
        """Run the complete pipeline."""
        self.stats['start_time'] = datetime.now()
        
        try:
            self.logger.info("🚀 AI IDEA EVALUATOR PIPELINE STARTING")
            self.logger.info("=" * 60)
            
            # Discover input files
            input_files = self.discover_input_files()
            if not input_files:
                self.logger.warning("No input files found")
                return {'success': False, 'error': 'No input files found'}
            
            self.stats['total_files'] = len(input_files)
            
            # Process files
            results = self.process_batch(input_files)
            
            # Generate summary
            self.stats['end_time'] = datetime.now()
            
            # Log final statistics only (no report saving)
            self.logger.info("PIPELINE COMPLETED")
            self.logger.info("=" * 60)
            self.logger.info(f"Processed: {self.stats['successful_files']}/{self.stats['total_files']} files")
            self.logger.info(f"Total slides: {self.stats['total_slides']}")
            self.logger.info(f"Total embeddings: {self.stats['total_embeddings']}")
            self.logger.info(f"Total time: {(self.stats['end_time'] - self.stats['start_time']).total_seconds():.2f}s")
            
            if self.stats['failed_files'] > 0:
                self.logger.warning(f"{self.stats['failed_files']} files failed processing")
            
            return {
                'success': True,
                'results': results
            }
            
        except Exception as e:
            self.stats['end_time'] = datetime.now()
            self.logger.error(f"❌ Pipeline failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file or use defaults."""
    default_config = {
        'input_directory': './input_submissions',
        'output_directory': './pipeline_output',
        'max_file_size': 50 * 1024 * 1024,  # 50MB
        'max_pages': 25,
        'supported_formats': ['.pdf', '.ppt', '.pptx'],
        'log_level': 'INFO',
        'processing_delay': 0,  # Seconds between files
        'vector_storage_type': 'chromadb'
    }
    
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            file_config = json.load(f)
        default_config.update(file_config)
    
    return default_config

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='AI Idea Evaluator - Complete Processing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python ai_evaluator_pipeline.py
  
  # Run with custom directories
  python ai_evaluator_pipeline.py --input ./submissions --output ./results
  
  # Run with configuration file
  python ai_evaluator_pipeline.py --config config.json
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        default='./input_submissions',
        help='Input directory containing documents to process (default: ./input_submissions)'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='./pipeline_output',
        help='Output directory for results (default: ./pipeline_output)'
    )
    
    parser.add_argument(
        '--config', '-c',
        help='Configuration file path (JSON format)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    config['input_directory'] = args.input
    config['output_directory'] = args.output
    config['log_level'] = args.log_level
    
    # Ensure directories exist
    Path(config['input_directory']).mkdir(parents=True, exist_ok=True)
    Path(config['output_directory']).mkdir(parents=True, exist_ok=True)
    
    # Initialize and run pipeline
    try:
        pipeline = AIEvaluatorPipeline(config)
        result = pipeline.run_pipeline()
        
        if result['success']:
            # Log success with simple information
            pipeline.logger.info("🎉 Pipeline completed successfully!")
            
            # Calculate simple stats from results
            successful_files = sum(1 for r in result['results'] if r['success'])
            total_files = len(result['results'])
            total_slides = sum(r.get('total_slides', 0) for r in result['results'] if r['success'])
            total_embeddings = sum(r.get('embeddings_result', {}).get('embeddings_count', 0) for r in result['results'] if r['success'])
            success_rate = (successful_files / max(total_files, 1)) * 100
            
            # Print simple success message
            print(f"\n✅ SUCCESS: {successful_files}/{total_files} files | {total_slides} slides | {total_embeddings} embeddings | {success_rate:.1f}% success rate")
        else:
            pipeline.logger.error(f"❌ Pipeline failed: {result['error']}")
            print(f"\n❌ Pipeline failed: {result['error']}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger = logging.getLogger(__name__)
        logger.warning("⏸️ Pipeline interrupted by user")
        print("\n⏸️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"❌ Pipeline crashed: {e}")
        print(f"\n❌ Pipeline crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
