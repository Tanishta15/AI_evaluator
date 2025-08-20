import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import shutil
import subprocess
import sys
import os
import json
from pathlib import Path
import time
from typing import List, Tuple, Dict, Any
import logging

# Import certificate verification functionality
try:
    from certificate_verifier import verify_from_parquet
    CERTIFICATE_VERIFICATION_AVAILABLE = True
except ImportError:
    CERTIFICATE_VERIFICATION_AVAILABLE = False
    print("Warning: Certificate verification not available. Install pytesseract for full functionality.")
import logging
 
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentEvaluatorPipeline:
    """Clean and focused pipeline for evaluating uploaded content."""
    
    def __init__(self):
        self.temp_dir = None
        self.results = {}
        
    def setup_temp_environment(self) -> str:
        """Setup temporary environment for processing and clean old data."""
        if self.temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Clean up old permanent data before starting new processing
        self.cleanup_old_data()
        
        self.temp_dir = tempfile.mkdtemp(prefix="evaluator_")
        
        # Create required directories
        (Path(self.temp_dir) / "input_submissions").mkdir(exist_ok=True)
        (Path(self.temp_dir) / "pipeline_output").mkdir(exist_ok=True)
        (Path(self.temp_dir) / "evaluation_results").mkdir(exist_ok=True)
        (Path(self.temp_dir) / "scores").mkdir(exist_ok=True)
        
        return self.temp_dir
    
    def cleanup_old_data(self):
        """Remove old pipeline output, results, and input submissions before processing new data."""
        print("🧹 Cleaning up old data before processing new uploads...")
        
        # Clean old pipeline output
        pipeline_output = Path("./pipeline_output")
        if pipeline_output.exists():
            shutil.rmtree(pipeline_output, ignore_errors=True)
            print(f"   Removed old pipeline_output directory")
        pipeline_output.mkdir(exist_ok=True)
        
        # Clean old results
        results_dir = Path("./results")
        if results_dir.exists():
            shutil.rmtree(results_dir, ignore_errors=True)
            print(f"   Removed old results directory")
        results_dir.mkdir(exist_ok=True)
        
        # Clean old input submissions (except the permanent ones if any)
        input_submissions = Path("./input_submissions")
        if input_submissions.exists():
            # Remove any subdirectories that might contain old uploads
            for item in input_submissions.iterdir():
                if item.is_dir() and item.name.startswith("Theme"):
                    shutil.rmtree(item, ignore_errors=True)
                    print(f"   Removed old input directory: {item.name}")
        
        # Clear any temp pipeline outputs that might exist
        for temp_dir in Path(".").glob("pipeline_output*"):
            if temp_dir.is_dir() and temp_dir.name != "pipeline_output":
                shutil.rmtree(temp_dir, ignore_errors=True)
                print(f"   Removed temporary pipeline directory: {temp_dir.name}")
        
        # Reset results in memory
        self.results = {}
        print("✅ Old data cleanup completed")
    
    def save_uploaded_files(self, files: List) -> List[str]:
        """Save uploaded files to temp directory."""
        if not files:
            return []
        
        saved_files = []
        input_dir = Path(self.temp_dir) / "input_submissions"
        
        for file_obj in files:
            if file_obj and hasattr(file_obj, 'name'):
                # file_obj.name contains the full path to the uploaded file
                file_path = file_obj.name
                if os.path.exists(file_path):
                    file_name = Path(file_path).name
                    dest_path = input_dir / file_name
                    shutil.copy2(file_path, dest_path)
                    saved_files.append(str(dest_path))
                    print(f"Saved file: {file_name} to {dest_path}")
        
        return saved_files
    
    def run_document_processing(self) -> Tuple[bool, str]:
        """Run the document processing pipeline (ai_evaluator_pipeline.py)."""
        try:
            original_cwd = os.getcwd()
            
            # Save directly to main pipeline_output directory (which was cleaned in setup)
            cmd = [
                sys.executable, 
                "ai_evaluator_pipeline.py",
                "--input", str(Path(self.temp_dir) / "input_submissions"),
                "--output", "./pipeline_output",  # Save directly to main directory
                "--log-level", "INFO"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes timeout
                cwd=original_cwd
            )
            
            if result.returncode == 0:
                # Check if output was actually created in main pipeline_output
                output_dir = Path("./pipeline_output")
                if output_dir.exists():
                    created_items = list(output_dir.iterdir())
                    if created_items:
                        # Run additional image processing if parquet files were created
                        success_msg = f"Document processing completed! Created {len(created_items)} items in ./pipeline_output/"
                        
                        # Try to extract text from all images in the processed files
                        image_processing_msg = self.process_images_in_parquet_files()
                        if image_processing_msg:
                            success_msg += f"\n{image_processing_msg}"
                        
                        return True, success_msg
                    else:
                        return False, f"Document processing completed but no output generated."
                else:
                    return False, f"Pipeline output directory not created."
            else:
                return False, f"Document processing failed: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return False, "Document processing timed out after 5 minutes"
        except Exception as e:
            return False, f"Document processing error: {str(e)}"
    
    def process_images_in_parquet_files(self) -> str:
        """Process images in all parquet files using OCR text extraction."""
        try:
            # Use main pipeline_output directory instead of temp
            pipeline_output = Path("./pipeline_output")
            parquet_files = list(pipeline_output.rglob("*.parquet"))
            
            if not parquet_files:
                return ""
            
            total_processed = 0
            processing_messages = []
            
            for parquet_file in parquet_files:
                try:
                    # Import the simplified image processing function
                    sys.path.insert(0, os.getcwd())
                    from image import extract_text_from_all_images
                    
                    # Get the base directory for images (parent of parquet file)
                    base_dir = str(parquet_file.parent)
                    
                    # Extract text from all images in this parquet file
                    processed_count = extract_text_from_all_images(
                        str(parquet_file), 
                        base_dir, 
                        save_changes=True
                    )
                    
                    total_processed += processed_count
                    
                    if processed_count > 0:
                        processing_messages.append(
                            f"Images processed for {parquet_file.name}: {processed_count}"
                        )
                    
                except Exception as e:
                    processing_messages.append(f"Image processing failed for {parquet_file.name}: {str(e)}")
            
            if total_processed > 0:
                summary = f"OCR Text Extraction Complete:\n"
                summary += f"   Images Processed: {total_processed}\n"
                summary += f"   Files Processed: {len(parquet_files)}"
                
                if processing_messages:
                    summary += "\n\nDetails:\n" + "\n".join(processing_messages)
                return summary
            else:
                return "No images found to process in the parquet files."
                
        except Exception as e:
            return f"Image processing error: {str(e)}"
    
    def run_certificate_verification(self) -> Tuple[bool, str]:
        """Run certificate verification on all processed submissions."""
        if not CERTIFICATE_VERIFICATION_AVAILABLE:
            return False, "Certificate verification not available. Please install pytesseract."
        
        try:
            # Check if data.csv exists for verification
            data_csv_path = Path("./data.csv")
            if not data_csv_path.exists():
                return False, "data.csv not found. Certificate verification requires a participant database."
            
            # Find all parquet files from main pipeline output
            pipeline_output = Path("./pipeline_output")
            parquet_files = list(pipeline_output.rglob("*.parquet"))
            
            if not parquet_files:
                return False, "No parquet files found for certificate verification."
            
            verification_results = []
            successful_verifications = 0
            
            for parquet_file in parquet_files:
                try:
                    # Extract submission name from parquet file
                    submission_name = parquet_file.stem
                    
                    # Run certificate verification
                    result = verify_from_parquet(
                        str(parquet_file),
                        str(data_csv_path),
                        base_dir=str(parquet_file.parent)
                    )
                    
                    # Add submission info to result
                    result['submission_name'] = submission_name
                    verification_results.append(result)
                    
                    if result.get('ok', False):
                        successful_verifications += 1
                        
                except Exception as e:
                    # Add failed verification result
                    verification_results.append({
                        'submission_name': parquet_file.stem,
                        'ok': False,
                        'error': str(e),
                        'similarity': 0.0
                    })
            
            # Save verification results
            verification_dir = Path(self.temp_dir) / "verification_results"
            verification_dir.mkdir(exist_ok=True)
            
            # Convert results to DataFrame and save
            if verification_results:
                df_results = pd.DataFrame(verification_results)
                verification_csv = verification_dir / "certificate_verification.csv"
                df_results.to_csv(verification_csv, index=False)
                
                # Also save to permanent results directory
                permanent_results_dir = Path("./results")
                permanent_results_dir.mkdir(exist_ok=True)
                permanent_csv = permanent_results_dir / "certificate_verification.csv"
                df_results.to_csv(permanent_csv, index=False)
            
            total_files = len(parquet_files)
            success_msg = f"Certificate verification completed! Verified {successful_verifications}/{total_files} submissions successfully."
            
            if successful_verifications > 0:
                success_msg += f"\nResults saved to ./results/certificate_verification.csv"
            
            return True, success_msg
            
        except Exception as e:
            return False, f"Certificate verification error: {str(e)}"
    
    def run_evaluation(self, theme="default") -> Tuple[bool, str]:
        """Run the evaluation pipeline (report_generator.py) with specified theme."""
        try:
            # Check if there are processed files first in main pipeline_output
            pipeline_output = Path("./pipeline_output")
            if not pipeline_output.exists():
                return False, "No pipeline_output directory found"
            
            # Create evaluation_results directory and copy parquet files there
            evaluation_results = Path(self.temp_dir) / "evaluation_results"
            evaluation_results.mkdir(exist_ok=True)
            
            # Look for parquet files in subdirectories and copy them to evaluation_results
            parquet_files = list(pipeline_output.rglob("*.parquet"))
            if not parquet_files:
                return False, f"No .parquet files found in {pipeline_output}"
            
            # Copy parquet files to evaluation_results directory with flat structure
            copied_files = []
            for parquet_file in parquet_files:
                # Create a unique name if there are naming conflicts
                dest_name = parquet_file.name
                dest_path = evaluation_results / dest_name
                counter = 1
                while dest_path.exists():
                    name_parts = parquet_file.stem, counter, parquet_file.suffix
                    dest_name = f"{name_parts[0]}_{name_parts[1]}{name_parts[2]}"
                    dest_path = evaluation_results / dest_name
                    counter += 1
                
                shutil.copy2(parquet_file, dest_path)
                copied_files.append(str(dest_path))
                copied_files.append(str(dest_path))
            
            original_cwd = os.getcwd()
            
            # Create permanent results directory
            results_dir = Path("./results")
            results_dir.mkdir(exist_ok=True)
            
            # Check if theme is a predefined theme or custom track
            predefined_themes = ['default', 'sustainability', 'healthcare', 'fintech', 'education', 'ai_ml']
            
            if theme in predefined_themes:
                # Use predefined theme
                cmd = [
                    sys.executable,
                    "report_generator.py", 
                    "--input_dir", str(evaluation_results),
                    "--out_prefix", str(results_dir / "evaluation"),
                    "--theme", theme
                ]
            else:
                # Custom track - load weights and pass them individually
                tracks = load_custom_tracks()
                if theme in tracks:
                    track_data = tracks[theme]
                    dimensions = track_data.get('dimensions', {})
                    section_weights = track_data.get('section_weights', {})
                    
                    # Validate that we have all required weights
                    required_dimensions = ['uniqueness', 'Completeness of the solution', 'impact on the theme chosen', 'ethical consideration']
                    required_sections = ['problem_statement', 'proposed_solution', 'technical_architecture']
                    
                    missing_dims = [dim for dim in required_dimensions if dim not in dimensions]
                    missing_sections = [sec for sec in required_sections if sec not in section_weights]
                    
                    if missing_dims or missing_sections:
                        return False, f"Custom track '{theme}' is missing required weights. Missing dimensions: {missing_dims}, Missing sections: {missing_sections}"
                    
                    cmd = [
                        sys.executable,
                        "report_generator.py", 
                        "--input_dir", str(evaluation_results),
                        "--out_prefix", str(results_dir / "evaluation"),
                        "--uniqueness_weight", str(dimensions['uniqueness']),
                        "--completeness_weight", str(dimensions['Completeness of the solution']),
                        "--impact_weight", str(dimensions['impact on the theme chosen']),
                        "--ethics_weight", str(dimensions['ethical consideration']),
                        "--problem_weight", str(section_weights['problem_statement']),
                        "--solution_weight", str(section_weights['proposed_solution']),
                        "--architecture_weight", str(section_weights['technical_architecture'])
                    ]
                else:
                    # Fallback to default theme if custom track not found
                    return False, f"Custom track '{theme}' not found in tracks configuration"
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes timeout
                cwd=original_cwd
            )
            
            if result.returncode == 0:
                # Check if evaluation results were created in the permanent results directory
                results_dir = Path("./results")
                evaluation_csv = results_dir / "evaluation.csv"
                if evaluation_csv.exists():
                    # Limit results to top 20 only
                    self.limit_results_to_top20()
                    return True, f"Evaluation completed with {theme} theme! Processed {len(copied_files)} parquet files and generated results."
                else:
                    # List what files were actually created for debugging
                    existing_files = list(results_dir.glob("*")) if results_dir.exists() else []
                    return False, f"Evaluation completed but no CSV results generated in {results_dir}. Found files: {[f.name for f in existing_files]}"
            else:
                return False, f"Evaluation failed: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return False, "Evaluation timed out after 10 minutes"
        except Exception as e:
            return False, f"Evaluation error: {str(e)}"
    
    def load_results(self) -> Dict[str, Any]:
        """Load and return all results, prioritizing top results CSV files."""
        results = {}
        # Load results from permanent directory since report_generator.py saves directly there
        results_dir = Path("./results")
        
        # First try to load top 20 results if available
        top20_csv = results_dir / "evaluation_top20.csv"
        main_csv = results_dir / "evaluation.csv"
        
        # Load main evaluation results (prefer top20, fallback to main)
        if top20_csv.exists():
            results['main_scores'] = pd.read_csv(top20_csv)
            print(f"Loaded top 20 results from {top20_csv}")
        elif main_csv.exists():
            # Load full results but limit to top 20 in frontend
            full_df = pd.read_csv(main_csv)
            results['main_scores'] = full_df.head(20) if len(full_df) > 20 else full_df
            print(f"Loaded and limited main results to top 20 from {main_csv}")
        else:
            print("No main results CSV found in results directory")
        
        # Load per-model results
        per_model_csv = results_dir / "evaluation_per_model.csv"
        if per_model_csv.exists():
            results['per_model_scores'] = pd.read_csv(per_model_csv)

        # Load certificate verification results
        cert_verification_csv = results_dir / "certificate_verification.csv"
        if cert_verification_csv.exists():
            results['certificate_verification'] = pd.read_csv(cert_verification_csv)
            print(f"Loaded certificate verification results from {cert_verification_csv}")

        self.results = results
        
        return results
    
    def limit_results_to_top20(self):
        """Post-process results to limit main CSV files to top 20 entries only."""
        results_dir = Path("./results")
        
        if not results_dir.exists():
            return
            
        # Process main evaluation CSV to limit to top 20
        main_csv = results_dir / "evaluation.csv"
        if main_csv.exists():
            try:
                df = pd.read_csv(main_csv)
                
                if 'overall_score' in df.columns and len(df) > 20:
                    # Sort by overall_score descending and take top 20
                    df_sorted = df.sort_values('overall_score', ascending=False)
                    df_top20 = df_sorted.head(20)
                    
                    # Save the limited results back to the same file
                    df_top20.to_csv(main_csv, index=False)
                    print(f"Limited main results to top 20 ({len(df)} → {len(df_top20)} rows)")
                    
            except Exception as e:
                print(f"Error limiting results to top 20: {e}")
    
    def copy_results_to_permanent_location(self):
        """Copy evaluation results to a permanent 'results' directory, limiting to top 20."""
        if not self.temp_dir:
            return
            
        # Create permanent results directory
        permanent_results_dir = Path("./results")
        permanent_results_dir.mkdir(exist_ok=True)
        
        # Process and save only top 20 results from CSV files
        scores_dir = Path(self.temp_dir) / "scores"
        if scores_dir.exists():
            for csv_file in scores_dir.glob("*.csv"):
                try:
                    # Read the CSV file
                    df = pd.read_csv(csv_file)
                    
                    # If this is the main evaluation file, limit to top 20
                    if 'overall_score' in df.columns and len(df) > 20:
                        # Sort by overall_score descending and take top 20
                        df_sorted = df.sort_values('overall_score', ascending=False)
                        df_top20 = df_sorted.head(20)
                        
                        # Save the limited results
                        dest_file = permanent_results_dir / csv_file.name
                        df_top20.to_csv(dest_file, index=False)
                        print(f"Saved top 20 results to {dest_file} ({len(df)} → {len(df_top20)} rows)")
                    else:
                        # For other CSV files (per-model, etc.), copy as is
                        dest_file = permanent_results_dir / csv_file.name
                        shutil.copy2(csv_file, dest_file)
                        print(f"Copied {csv_file.name} to {dest_file}")
                        
                except Exception as e:
                    # Fallback to copying if there's an error reading the CSV
                    dest_file = permanent_results_dir / csv_file.name
                    shutil.copy2(csv_file, dest_file)
                    print(f"Copied {csv_file.name} to {dest_file} (fallback due to: {e})")
            
            # Also copy config file
            for json_file in scores_dir.glob("*.json"):
                dest_file = permanent_results_dir / json_file.name
                shutil.copy2(json_file, dest_file)
                print(f"Copied {json_file.name} to {dest_file}")
    
    def cleanup(self):
        """Cleanup temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.temp_dir = None

# Global pipeline instance
pipeline = ContentEvaluatorPipeline()

def process_uploads(files, theme="default") -> str:
    """Process uploaded files through the complete pipeline with specified theme."""
    print(f"process_uploads called with {len(files) if files else 0} files, theme: {theme}")
    
    if not files:
        return "No files uploaded"
    
    try:
        # Setup environment (this will clean old data first)
        temp_dir = pipeline.setup_temp_environment()
        print(f"Temp directory created: {temp_dir}")
        status_msg = f"🧹 Cleaned old data and setup environment.\n\nProcessing {len(files)} NEW files with {theme} theme...\n\n"
        
        # Debug file information
        for i, file in enumerate(files):
            print(f"File {i}: {file.name if hasattr(file, 'name') else file}")
        
        # Save uploaded files
        saved_files = pipeline.save_uploaded_files(files)
        print(f"Saved {len(saved_files)} files")
        status_msg += f"💾 Saved {len(saved_files)} files to processing directory\n\n"
        
        # Step 1: Run document processing (ai_evaluator_pipeline.py)
        status_msg += "Step 1: Running document processing and image extraction...\n"
        success, msg = pipeline.run_document_processing()
        status_msg += msg + "\n\n"
        
        if not success:
            return status_msg
        
        # Verify pipeline output
        pipeline_output = Path(temp_dir) / "pipeline_output"
        if pipeline_output.exists():
            output_items = list(pipeline_output.iterdir())
            status_msg += f"Document processing created {len(output_items)} items in pipeline_output/\n"
            
            # Check for extracted text in parquet files
            parquet_files = list(pipeline_output.rglob("*.parquet"))
            if parquet_files:
                status_msg += f"Found {len(parquet_files)} parquet files for evaluation\n\n"
            else:
                status_msg += "No parquet files found\n\n"
        
        # Step 2: Run evaluation (report_generator.py) with theme
        status_msg += f"Step 2: Running evaluation with {theme} theme (report_generator.py)...\n"
        success, msg = pipeline.run_evaluation(theme)
        status_msg += msg + "\n\n"
        
        if not success:
            return status_msg
        
        # Step 3: Run certificate verification (if available)
        if CERTIFICATE_VERIFICATION_AVAILABLE:
            status_msg += "Step 3: Running certificate verification...\n"
            success, msg = pipeline.run_certificate_verification()
            status_msg += msg + "\n\n"
            # Note: Continue even if certificate verification fails
        else:
            status_msg += "Step 3: Skipping certificate verification (pytesseract not available)\n\n"
        
        # Step 4: Load results
        status_msg += "Step 4: Loading results...\n"
        results = pipeline.load_results()
        
        if 'main_scores' in results:
            status_msg += "Pipeline completed successfully! Generated results for {len(results['main_scores'])} submissions using {theme} theme configuration.\n\n"
            status_msg += "Results saved to ./results/ directory (TOP 20 ONLY):\n"
            status_msg += "   - evaluation.csv (top 20 main results)\n"
            status_msg += "   - evaluation_top20.csv (top 20 rankings)\n"
            status_msg += "   - evaluation_per_model.csv (per-model scores)\n"
            status_msg += "   - evaluation_config.json (configuration used)"
            return status_msg
        else:
            status_msg += "No evaluation results found in CSV files"
            return status_msg
            
    except Exception as e:
        error_msg = f"Pipeline failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg

# SUBMISSIONS PAGE FUNCTIONS
def get_submissions_table():
    """Generate table with PPT names as rows and criteria as columns, plus download buttons."""
    if not pipeline.results or 'main_scores' not in pipeline.results:
        return "<p>No data available. Please run evaluation first.</p>"
    
    df = pipeline.results['main_scores'].copy()
    
    # Sort by overall score descending
    df_sorted = df.sort_values('overall_score', ascending=False).reset_index(drop=True)
    
    # Create table with PPT names as rows and criteria as columns
    criteria_columns = [
        'Problem Statement', 'Proposed Solution', 'Technical Architecture', 
        'Novelty  Uniqueness/creativity', 'presentaion', 'ethical consideration',
        'completeness of design solution', 'implementing ibm dpk/rag/aws/watsonx/granite',
        'overall_score', 'Certification(yes/no)', 'Feedback', 'Missing content(anything missing)'
    ]
    available_columns = [col for col in criteria_columns if col in df_sorted.columns]
    
    if not available_columns:
        return "<p>No scoring data available in results.</p>"
    
    # Create display dataframe with PPT names as index
    display_df = df_sorted[['submission_id'] + available_columns].copy()
    display_df = display_df.set_index('submission_id')
    
    # Round numeric columns
    for col in available_columns:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(2)
    
    # Rename columns for better display (keeping the exact names you specified)
    column_mapping = {
        'Problem Statement': 'Problem Statement',
        'Proposed Solution': 'Proposed Solution', 
        'Technical Architecture': 'Technical Architecture',
        'Novelty  Uniqueness/creativity': 'Novelty  Uniqueness/creativity',
        'presentaion': 'presentaion',
        'ethical consideration': 'ethical consideration',
        'completeness of design solution': 'completeness of design solution',
        'implementing ibm dpk/rag/aws/watsonx/granite': 'implementing ibm dpk/rag/aws/watsonx/granite',
        'overall_score': 'Overall Score',
        'Certification(yes/no)': 'Certification(yes/no)',
        'Feedback': 'Feedback',
        'Missing content(anything missing)': 'Missing content(anything missing)'
    }
    
    display_df = display_df.rename(columns=column_mapping)
    
    # Create HTML table with custom formatting for feedback columns
    html_table = display_df.to_html(
        classes="evaluation-table",
        table_id="evaluation-results",
        escape=False,
        border=0
    )
    
    # Add data-feedback attributes to feedback columns for styling
    import re
    
    # Function to add horizontal scroll wrapper to long text
    def wrap_long_text(text, max_length=100):
        if len(str(text)) > max_length:
            return f'<div data-feedback style="max-width: 300px; overflow-x: auto; white-space: nowrap;">{text}</div>'
        return text
    
    # Find and replace feedback and missing content cells
    feedback_pattern = r'<td>([^<]*(?:STRENGTHS|AREAS FOR IMPROVEMENT|Missing Requirements)[^<]*)</td>'
    missing_pattern = r'<td>([^<]*(?:Missing|None)[^<]*)</td>'
    
    html_table = re.sub(feedback_pattern, lambda m: f'<td data-feedback>{wrap_long_text(m.group(1))}</td>', html_table)
    html_table = re.sub(missing_pattern, lambda m: f'<td data-feedback>{wrap_long_text(m.group(1))}</td>', html_table)
    
    # Check available CSV files for download buttons
    results_dir = Path("./results")
    download_buttons = ""
    
    if results_dir.exists():
        csv_files = [
            ("evaluation.csv", "Complete Results", "Complete evaluation results with all details"),
            ("evaluation_top20.csv", "Top 20 Results", "Top 20 submissions only"), 
            ("evaluation_per_model.csv", "Per-Model Scores", "Individual model scoring breakdown"),
            ("certificate_verification.csv", "Certificate Verification", "Certificate verification results")
        ]
        
        download_buttons = """
        <div style="
            background: rgba(40, 167, 69, 0.15);
            border: 2px solid rgba(40, 167, 69, 0.5);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        ">
            <h3 style="color: #28a745; margin: 0 0 15px 0;">📥 Download Results</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px;">
        """
        
        for filename, title, description in csv_files:
            csv_path = results_dir / filename
            if csv_path.exists():
                file_size = csv_path.stat().st_size / 1024  # KB
                download_buttons += f"""
                <div style="
                    background: white;
                    border: 2px solid #28a745;
                    border-radius: 8px;
                    padding: 15px;
                    text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                ">
                    <h4 style="color: #28a745; margin: 0 0 8px 0;">{title}</h4>
                    <p style="color: #666; font-size: 14px; margin: 0 0 10px 0;">{description}</p>
                    <p style="color: #999; font-size: 12px; margin: 0 0 15px 0;">Size: {file_size:.1f} KB</p>
                    <button onclick="window.open('./results/{filename}', '_blank')" style="
                        background: #28a745;
                        color: white;
                        border: none;
                        padding: 10px 20px;
                        border-radius: 5px;
                        cursor: pointer;
                        font-weight: bold;
                        font-size: 14px;
                    ">
                        📄 Download {filename}
                    </button>
                    <p style="color: #999; font-size: 11px; margin: 10px 0 0 0;">Location: ./results/{filename}</p>
                </div>
                """
        
        download_buttons += """
            </div>
            <div style="
                background: rgba(40, 167, 69, 0.2);
                padding: 15px;
                border-radius: 8px;
                margin-top: 15px;
                text-align: center;
            ">
            </div>
        </div>
        """
    
    # Create summary header
    total_submissions = len(df_sorted)
    summary_html = f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        text-align: center;
    ">
        <h2 style="margin: 0 0 10px 0;">Evaluation Results - {total_submissions} Submissions</h2>
    </div>
    """
    
    # Style the table
    styled_html = f"""
    <style>
        .evaluation-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-family: 'Segoe UI', sans-serif;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
            font-size: 14px;
        }}
        .evaluation-table th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 8px;
            text-align: center;
            font-weight: 600;
            border-bottom: 2px solid #5a67d8;
        }}
        .evaluation-table td {{
            padding: 10px 8px;
            text-align: center;
            border-bottom: 1px solid #e2e8f0;
            background-color: #1a1a1a;
            color: #ffffff;
        }}
        .evaluation-table th:first-child {{
            text-align: left;
            padding-left: 15px;
        }}
        .evaluation-table td:first-child {{
            text-align: left;
            font-weight: bold;
            background-color: #2a2a2a;
            min-width: 200px;
            padding-left: 15px;
        }}
        .evaluation-table tr:nth-child(even) td {{
            background-color: #2a2a2a;
        }}
        .evaluation-table tr:nth-child(even) td:first-child {{
            background-color: #3a3a3a;
        }}
        .evaluation-table tr:hover td {{
            background-color: #3a3a3a;
        }}
        .evaluation-table tr:hover td:first-child {{
            background-color: #4a4a4a;
        }}
        
        /* Feedback column styling with horizontal scroll */
        .evaluation-table td:has-text("Feedback"), 
        .evaluation-table td:has-text("Missing content"),
        .evaluation-table th:has-text("Feedback"),
        .evaluation-table th:has-text("Missing content") {{
            max-width: 300px;
            white-space: nowrap;
            overflow-x: auto;
            text-align: left;
            position: relative;
        }}
        
        /* Style for feedback and missing content columns */
        .evaluation-table td[data-feedback] {{
            max-width: 300px;
            white-space: nowrap;
            overflow-x: auto;
            text-align: left;
            padding: 8px;
            cursor: text;
        }}
        
        /* Add scrollbar styling for better visibility */
        .evaluation-table td[data-feedback]::-webkit-scrollbar {{
            height: 6px;
        }}
        
        .evaluation-table td[data-feedback]::-webkit-scrollbar-track {{
            background: #444;
            border-radius: 3px;
        }}
        
        .evaluation-table td[data-feedback]::-webkit-scrollbar-thumb {{
            background: #666;
            border-radius: 3px;
        }}
        
        .evaluation-table td[data-feedback]::-webkit-scrollbar-thumb:hover {{
            background: #777;
        }}
        
        button:hover {{
            background: #218838 !important;
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
    </style>
    
    {summary_html}
    {download_buttons}
    {html_table}
    """
    
    return styled_html

# CERTIFICATE VERIFICATION FUNCTIONS
def get_certificate_verification_results():
    """Generate certificate verification results table."""
    if not pipeline.results or 'certificate_verification' not in pipeline.results:
        if not CERTIFICATE_VERIFICATION_AVAILABLE:
            return """
            <div style="
                background: rgba(255, 193, 7, 0.15);
                border: 2px solid rgba(255, 193, 7, 0.5);
                border-radius: 12px;
                padding: 20px;
                margin: 20px 0;
                text-align: center;
            ">
                <h3 style="color: #856404; margin: 0 0 15px 0;">⚠️ Certificate Verification Not Available</h3>
                <p style="color: #856404; margin: 0;">
                    To enable certificate verification, please install pytesseract:<br>
                    <code>pip install pytesseract</code><br>
                    <em>Also ensure Tesseract OCR binary is installed on your system</em>
                </p>
            </div>
            """
        else:
            return "<p>No certificate verification data available. Please run evaluation first.</p>"
    
    df = pipeline.results['certificate_verification'].copy()
    
    if df.empty:
        return "<p>No certificate verification results found.</p>"
    
    # Sort by verification success and similarity score
    df_sorted = df.sort_values(['ok', 'similarity'], ascending=[False, False]).reset_index(drop=True)
    
    # Count statistics
    total_submissions = len(df_sorted)
    verified_count = len(df_sorted[df_sorted['ok'] == True])
    failed_count = total_submissions - verified_count
    
    # Create summary header
    summary_html = f"""
    <div style="
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        text-align: center;
    ">
        <h2 style="margin: 0 0 10px 0;">Certificate Verification Results</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-top: 15px;">
            <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px;">
                <div style="font-size: 24px; font-weight: bold; margin-bottom: 5px;">{total_submissions}</div>
                <div style="font-size: 14px; opacity: 0.9;">Total Checked</div>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px;">
                <div style="font-size: 24px; font-weight: bold; margin-bottom: 5px;">{verified_count}</div>
                <div style="font-size: 14px; opacity: 0.9;">Verified ✅</div>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px;">
                <div style="font-size: 24px; font-weight: bold; margin-bottom: 5px;">{failed_count}</div>
                <div style="font-size: 14px; opacity: 0.9;">Failed ❌</div>
            </div>
        </div>
    </div>
    """
    
    # Create results cards
    cards_html = ""
    
    for _, row in df_sorted.iterrows():
        submission_name = row.get('submission_name', 'Unknown')
        is_verified = row.get('ok', False)
        similarity = row.get('similarity', 0.0)
        
        # Status styling
        if is_verified:
            status_color = "#28a745"
            status_icon = "✅"
            status_text = "VERIFIED"
            border_color = "#28a745"
        else:
            status_color = "#dc3545"
            status_icon = "❌"
            status_text = "FAILED"
            border_color = "#dc3545"
        
        cards_html += f"""
        <div style="
            background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
            border: 3px solid {border_color};
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h3 style="color: #ffffff; margin: 0; font-size: 18px;">
                    {submission_name}
                </h3>
                <div style="
                    background: {status_color};
                    color: white;
                    padding: 8px 16px;
                    border-radius: 20px;
                    font-weight: bold;
                    font-size: 14px;
                ">
                    {status_icon} {status_text}
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
                <div style="
                    background: rgba(255,255,255,0.1);
                    padding: 15px;
                    border-radius: 8px;
                    border-left: 4px solid {status_color};
                ">
                    <div style="color: #cccccc; font-size: 12px; margin-bottom: 8px;">
                        Similarity Score
                    </div>
                    <div style="color: #ffffff; font-weight: bold; font-size: 20px;">
                        {similarity:.1%}
                    </div>
                    <div style="background: #e9ecef; height: 6px; border-radius: 3px; margin-top: 8px;">
                        <div style="background: {status_color}; height: 6px; border-radius: 3px; width: {similarity*100:.1f}%;"></div>
                    </div>
                </div>
        """
        
        # Add extracted information if available
        if 'extracted' in row and pd.notna(row['extracted']):
            try:
                extracted = json.loads(row['extracted']) if isinstance(row['extracted'], str) else row['extracted']
                if extracted:
                    cards_html += f"""
                    <div style="
                        background: rgba(102, 126, 234, 0.15);
                        padding: 15px;
                        border-radius: 8px;
                        border-left: 4px solid #667eea;
                    ">
                        <div style="color: #cccccc; font-size: 12px; margin-bottom: 8px;">
                            Extracted Information
                        </div>
                        <div style="color: #ffffff; font-size: 14px; line-height: 1.4;">
                            <strong>Name:</strong> {extracted.get('name', 'N/A')}<br>
                            <strong>Program:</strong> {extracted.get('program', 'N/A')}<br>
                            <strong>Code:</strong> {extracted.get('code', 'N/A')}
                        </div>
                    </div>
                    """
            except:
                pass
        
        # Add error information if verification failed
        if not is_verified and 'error' in row and pd.notna(row['error']):
            error_msg = str(row['error'])
            cards_html += f"""
            <div style="
                background: rgba(244, 67, 54, 0.15);
                padding: 15px;
                border-radius: 8px;
                border-left: 4px solid #f44336;
            ">
                <div style="color: #cccccc; font-size: 12px; margin-bottom: 8px;">
                    Error Details
                </div>
                <div style="color: #ffcdd2; font-size: 14px; line-height: 1.4;">
                    {error_msg}
                </div>
            </div>
            """
        
        cards_html += """
            </div>
        </div>
        """
    
    # Add download information
    download_html = f"""
    <div style="
        background: rgba(40, 167, 69, 0.15);
        border: 2px solid rgba(40, 167, 69, 0.5);
        border-radius: 12px;
        padding: 15px;
        margin-top: 20px;
    ">
        <h3 style="color: #28a745; margin: 0 0 10px 0; display: flex; align-items: center;">
            <span style="margin-right: 8px;">📄</span> Certificate Verification Report
        </h3>
        <div style="color: #e0e0e0; font-size: 14px;">
            <p><strong>Detailed report available:</strong></p>
            <ul>
                <li><strong>certificate_verification.csv</strong> - Complete verification results</li>
            </ul>
            <p><em>Location: ./results/certificate_verification.csv</em></p>
        </div>
    </div>
    """
    
    return summary_html + cards_html + download_html

def get_per_submission_model_scores():
    """Per-submission model comparison removed for simplified interface."""
    return "<p>Per-submission model comparison tables have been removed for a cleaner interface. Use the CSV downloads for detailed model scoring data.</p>"

# MODEL PAGE FUNCTIONS
def get_available_models():
    """Get list of available models."""
    if not pipeline.results or 'per_model_scores' not in pipeline.results:
        return []

    per_model_df = pipeline.results['per_model_scores']
    if 'evaluator_model' in per_model_df.columns:
        return sorted(per_model_df['evaluator_model'].unique().tolist())
    return []

def get_model_scores_table(selected_model):
    """Generate table for specific model scores across submissions."""
    if not pipeline.results or 'per_model_scores' not in pipeline.results:
        return "<p>No per-model data available.</p>"
    
    if not selected_model:
        return "<p>Please select a model to view scores.</p>"
    
    per_model_df = pipeline.results['per_model_scores']
    model_data = per_model_df[per_model_df['evaluator_model'] == selected_model]
    
    if model_data.empty:
        return f"<p>No data available for model: {selected_model}</p>"
    
    # Create pivot table: submissions vs sections
    pivot_df = model_data.pivot_table(
        index='submission_id',
        columns='section', 
        values='section_total',
        aggfunc='mean'
    ).round(3)  # Increased precision
    
    # Add rank based on average score
    pivot_df['Average'] = pivot_df.mean(axis=1).round(3)
    pivot_df = pivot_df.sort_values('Average', ascending=False)
    pivot_df.insert(0, 'Rank', range(1, len(pivot_df) + 1))
    
    html = pivot_df.to_html(classes="model-table", escape=False)
    
    styled_html = f"""
    <style>
        .model-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-family: 'Segoe UI', sans-serif;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }}
        .model-table th, .model-table td {{
            padding: 12px 15px;
            text-align: center;
            border-bottom: 1px solid #e0e0e0;
        }}
        .model-table th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
        }}
        .model-table td {{
            background-color: #1a1a1a;
            color: #ffffff;
        }}
        .model-table tr:nth-child(even) td {{
            background-color: #2a2a2a;
        }}
        .model-table tr:hover td {{
            background-color: #3a3a3a;
        }}
    </style>
    
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; margin: 20px 0 10px 0; border-radius: 6px;">
        <h3>{selected_model} - Submission Scores</h3>
    </div>
    {html}
    """
    
    return styled_html

# ANALYTICS PAGE FUNCTIONS
def generate_analytics_charts():
    """Generate comprehensive analytics charts."""
    if not pipeline.results or 'main_scores' not in pipeline.results:
        return "<div style='padding: 20px; text-align: center; color: #888;'><h3>No data available for analytics</h3><p>Please run an evaluation first to see charts and graphs.</p></div>"
    
    main_df = pipeline.results['main_scores']
    per_model_df = pipeline.results.get('per_model_scores', pd.DataFrame())
    
    charts_html = "<div style='padding: 20px;'>"
    charts_html += "<h2 style='color: #667eea; margin-bottom: 30px;'>Analytics Dashboard</h2>"
    
    try:
        # Chart 1: Overall Score Rankings (Text-based visualization)
        if len(main_df) > 0 and 'overall_score' in main_df.columns:
            charts_html += "<div style='background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0;'>"
            charts_html += "<h3 style='color: #667eea; margin-bottom: 15px;'>Overall Score Rankings</h3>"
            
            df_sorted = main_df.sort_values('overall_score', ascending=False)
            max_score = df_sorted['overall_score'].max()
            
            for idx, (_, row) in enumerate(df_sorted.iterrows()):
                score = row['overall_score']
                percentage = (score / max_score) * 100
                rank = idx + 1
                
                # Color coding for ranks
                if rank == 1:
                    rank_color = "#FFD700"
                elif rank == 2:
                    rank_color = "#C0C0C0"
                elif rank == 3:
                    rank_color = "#CD7F32"
                else:
                    rank_color = "#667eea"
                
                charts_html += f"""
                <div style="margin: 10px 0; padding: 10px; background: white; border-radius: 8px; border-left: 4px solid {rank_color};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong style="color: {rank_color};">#{rank}</strong>
                            <span style="margin-left: 10px; color: #333;">{row.get('submission_id', 'Unknown')}</span>
                        </div>
                        <div style="color: #667eea; font-weight: bold;">{score:.2f}</div>
                    </div>
                    <div style="background: #e9ecef; height: 8px; border-radius: 4px; margin-top: 8px;">
                        <div style="background: {rank_color}; height: 8px; border-radius: 4px; width: {percentage}%;"></div>
                    </div>
                </div>
                """
            charts_html += "</div>"
        
        # Chart 2: Section Performance Breakdown
        section_cols = ['problem_statement_total', 'proposed_solution_total', 'technical_architecture_total']
        available_cols = [col for col in section_cols if col in main_df.columns]
        
        if len(available_cols) >= 2:
            charts_html += "<div style='background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0;'>"
            charts_html += "<h3 style='color: #667eea; margin-bottom: 15px;'>Section Performance Comparison</h3>"
            
            # Create a comparison table
            charts_html += "<div style='overflow-x: auto;'>"
            charts_html += "<table style='width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden;'>"
            charts_html += "<thead><tr style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;'>"
            charts_html += "<th style='padding: 12px; text-align: left;'>Submission</th>"
            
            for col in available_cols:
                display_name = col.replace('_total', '').replace('_', ' ').title()
                charts_html += f"<th style='padding: 12px; text-align: center;'>{display_name}</th>"
            charts_html += "<th style='padding: 12px; text-align: center;'>Overall</th>"
            charts_html += "</tr></thead><tbody>"
            
            for idx, (_, row) in enumerate(main_df.iterrows()):
                bg_color = "#f8f9fa" if idx % 2 == 0 else "white"
                charts_html += f"<tr style='background: {bg_color};'>"
                charts_html += f"<td style='padding: 12px; font-weight: bold; color: #333;'>{row.get('submission_id', 'Unknown')}</td>"
                
                for col in available_cols:
                    value = row.get(col, 0)
                    charts_html += f"<td style='padding: 12px; text-align: center; color: #667eea; font-weight: bold;'>{value:.2f}</td>"
                
                overall = row.get('overall_score', 0)
                charts_html += f"<td style='padding: 12px; text-align: center; color: #764ba2; font-weight: bold;'>{overall:.2f}</td>"
                charts_html += "</tr>"
            
            charts_html += "</tbody></table></div></div>"
        
        # Chart 3: Model Performance Analysis
        if not per_model_df.empty and 'evaluator_model' in per_model_df.columns:
            charts_html += "<div style='background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0;'>"
            charts_html += "<h3 style='color: #667eea; margin-bottom: 15px;'>Model Performance Analysis</h3>"
            
            # Calculate average scores by model
            model_avg = per_model_df.groupby('evaluator_model')['section_total'].agg(['mean', 'count']).round(2)
            model_avg = model_avg.sort_values('mean', ascending=False)
            
            if len(model_avg) > 0:
                max_avg = model_avg['mean'].max()
                
                for model_name, stats in model_avg.iterrows():
                    avg_score = stats['mean']
                    count = stats['count']
                    percentage = (avg_score / max_avg) * 100
                    
                    charts_html += f"""
                    <div style="margin: 10px 0; padding: 15px; background: white; border-radius: 8px; border-left: 4px solid #667eea;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <div>
                                <strong style="color: #333;">{model_name}</strong>
                                <span style="margin-left: 10px; color: #666; font-size: 14px;">({count} evaluations)</span>
                            </div>
                            <div style="color: #667eea; font-weight: bold; font-size: 16px;">{avg_score:.2f}</div>
                        </div>
                        <div style="background: #e9ecef; height: 10px; border-radius: 5px;">
                            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); height: 10px; border-radius: 5px; width: {percentage}%;"></div>
                        </div>
                    </div>
                    """
            charts_html += "</div>"
        
        # Summary Statistics
        charts_html += "<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 8px; margin-top: 20px;'>"
        charts_html += "<h3 style='color: white; margin-bottom: 15px;'>Summary Statistics</h3>"
        
        # Create a grid of statistics
        charts_html += "<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;'>"
        
        # Total submissions
        charts_html += f"""
        <div style="background: white; padding: 15px; border-radius: 8px; text-align: center; border-left: 4px solid #667eea;">
            <div style="color: #667eea; font-size: 24px; font-weight: bold;">{len(main_df)}</div>
            <div style="color: #666; font-size: 14px;">Total Submissions</div>
        </div>
        """
        
        if 'overall_score' in main_df.columns:
            avg_score = main_df['overall_score'].mean()
            max_score = main_df['overall_score'].max()
            min_score = main_df['overall_score'].min()
            
            charts_html += f"""
            <div style="background: white; padding: 15px; border-radius: 8px; text-align: center; border-left: 4px solid #28a745;">
                <div style="color: #28a745; font-size: 24px; font-weight: bold;">{avg_score:.2f}</div>
                <div style="color: #666; font-size: 14px;">Average Score</div>
            </div>
            <div style="background: white; padding: 15px; border-radius: 8px; text-align: center; border-left: 4px solid #ffc107;">
                <div style="color: #ffc107; font-size: 24px; font-weight: bold;">{max_score:.2f}</div>
                <div style="color: #666; font-size: 14px;">Highest Score</div>
            </div>
            <div style="background: white; padding: 15px; border-radius: 8px; text-align: center; border-left: 4px solid #dc3545;">
                <div style="color: #dc3545; font-size: 24px; font-weight: bold;">{min_score:.2f}</div>
                <div style="color: #666; font-size: 14px;">Lowest Score</div>
            </div>
            """
        
        if not per_model_df.empty:
            total_evaluations = len(per_model_df)
            unique_models = per_model_df['evaluator_model'].nunique() if 'evaluator_model' in per_model_df.columns else 0
            
            charts_html += f"""
            <div style="background: white; padding: 15px; border-radius: 8px; text-align: center; border-left: 4px solid #6f42c1;">
                <div style="color: #6f42c1; font-size: 24px; font-weight: bold;">{total_evaluations}</div>
                <div style="color: #666; font-size: 14px;">Total Evaluations</div>
            </div>
            <div style="background: white; padding: 15px; border-radius: 8px; text-align: center; border-left: 4px solid #fd7e14;">
                <div style="color: #fd7e14; font-size: 24px; font-weight: bold;">{unique_models}</div>
                <div style="color: #666; font-size: 14px;">Models Used</div>
            </div>
            """
        
        charts_html += "</div></div>"
    
    except Exception as e:
        charts_html += f"<div style='color: red; padding: 20px; border: 1px solid red; border-radius: 5px; margin: 20px 0;'><h3>Error generating analytics</h3><p>{str(e)}</p></div>"
    
    charts_html += "</div>"
    
    return charts_html

# TRACK MANAGEMENT FUNCTIONS
def load_custom_tracks():
    """Load custom tracks from JSON file."""
    tracks_file = "custom_tracks.json"
    
    try:
        if os.path.exists(tracks_file):
            with open(tracks_file, 'r') as f:
                custom_tracks = json.load(f)
            if custom_tracks:  # If there are custom tracks, return them
                return custom_tracks
        
        # If no custom tracks exist, return empty dict (user must create tracks)
        return {}
    except Exception as e:
        print(f"Error loading custom tracks: {e}")
        return {}

def save_custom_tracks(tracks):
    """Save custom tracks to JSON file."""
    tracks_file = "custom_tracks.json"
    try:
        with open(tracks_file, 'w') as f:
            json.dump(tracks, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving custom tracks: {e}")
        return False

def get_track_choices():
    """Get track choices for dropdown."""
    tracks = load_custom_tracks()
    if not tracks:
        return [("No tracks available - Create one in the Tracks tab", "")]
    
    choices = []
    for track_id, track_data in tracks.items():
        name = track_data.get('name', track_id)
        description = track_data.get('description', '')
        choices.append((f"{name} - {description}", track_id))
    return choices

def create_track(name, description, uniqueness_weight, completeness_weight, impact_weight, ethics_weight,
                problem_weight, solution_weight, architecture_weight):
    """Create a new custom track."""
    
    # Validate weights sum to 1.0
    dim_total = uniqueness_weight + completeness_weight + impact_weight + ethics_weight
    sec_total = problem_weight + solution_weight + architecture_weight
    
    if abs(dim_total - 1.0) > 0.01:
        return False, f"Dimension weights must sum to 1.0 (currently {dim_total:.3f})"
    
    if abs(sec_total - 1.0) > 0.01:
        return False, f"Section weights must sum to 1.0 (currently {sec_total:.3f})"
    
    # Create track ID from name
    track_id = name.lower().replace(' ', '_').replace('-', '_')
    
    # Load existing tracks
    tracks = load_custom_tracks()
    
    # Create new track
    tracks[track_id] = {
        "name": name,
        "description": description,
        "dimensions": {
            "uniqueness": uniqueness_weight,
            "Completeness of the solution": completeness_weight,
            "impact on the theme chosen": impact_weight,
            "ethical consideration": ethics_weight,
        },
        "section_weights": {
            "problem_statement": problem_weight,
            "proposed_solution": solution_weight,
            "technical_architecture": architecture_weight,
        }
    }
    
    # Save tracks
    if save_custom_tracks(tracks):
        return True, f"Track '{name}' created successfully!"
    else:
        return False, "Failed to save track"

def delete_track(track_id):
    """Delete a custom track."""
    if not track_id:
        return False, "No track selected"
    
    tracks = load_custom_tracks()
    if track_id in tracks:
        del tracks[track_id]
        if save_custom_tracks(tracks):
            return True, f"Track deleted successfully!"
        else:
            return False, "Failed to save changes"
    else:
        return False, "Track not found"

def get_track_analytics():
    """Generate analytics for all tracks."""
    tracks = load_custom_tracks()
    
    if not tracks:
        return """
        <div style='padding: 20px; text-align: center;'>
            <h2 style='color: #667eea; margin-bottom: 30px;'>Track Configuration Analytics</h2>
            <div style='background: #f8f9fa; padding: 30px; border-radius: 12px; border: 2px solid #dee2e6;'>
                <h3 style='color: #6c757d;'>No Tracks Available</h3>
                <p style='color: #6c757d; margin-bottom: 20px;'>Create your first track to see analytics here.</p>
                <div style='color: #495057; font-size: 14px;'>
                    Use the Create New Track section to get started.
                </div>
            </div>
        </div>
        """
    
    html = "<div style='padding: 20px;'>"
    html += "<h2 style='color: #667eea; margin-bottom: 30px;'>Track Configuration Analytics</h2>"
    
    for track_id, track_data in tracks.items():
        name = track_data.get('name', track_id)
        description = track_data.get('description', '')
        dimensions = track_data.get('dimensions', {})
        section_weights = track_data.get('section_weights', {})
        
        # All tracks are custom now
        track_type = "Custom"
        border_color = "#28a745"
        
        html += f"""
        <div style="
            border: 2px solid {border_color};
            border-radius: 12px;
            padding: 20px;
            margin: 20px 0;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h3 style="color: #333; margin: 0;">{name}</h3>
                <span style="
                    background: {border_color};
                    color: white;
                    padding: 5px 12px;
                    border-radius: 15px;
                    font-size: 12px;
                    font-weight: bold;
                ">{track_type}</span>
            </div>
            
            <p style="color: #666; margin-bottom: 20px; font-style: italic;">{description}</p>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div>
                    <h4 style="color: #667eea; margin-bottom: 10px;">Dimension Weights</h4>
                    <div style="background: white; padding: 15px; border-radius: 8px;">
        """
        
        for dim, weight in dimensions.items():
            percentage = weight * 100
            html += f"""
                        <div style="margin: 8px 0;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                                <span style="font-size: 14px; color: #333;">{dim.replace('_', ' ').title()}</span>
                                <span style="font-weight: bold; color: #667eea;">{percentage:.1f}%</span>
                            </div>
                            <div style="background: #e9ecef; height: 6px; border-radius: 3px;">
                                <div style="background: #667eea; height: 6px; border-radius: 3px; width: {percentage}%;"></div>
                            </div>
                        </div>
            """
        
        html += """
                    </div>
                </div>
                <div>
                    <h4 style="color: #764ba2; margin-bottom: 10px;">Section Weights</h4>
                    <div style="background: white; padding: 15px; border-radius: 8px;">
        """
        
        for section, weight in section_weights.items():
            percentage = weight * 100
            html += f"""
                        <div style="margin: 8px 0;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                                <span style="font-size: 14px; color: #333;">{section.replace('_', ' ').title()}</span>
                                <span style="font-weight: bold; color: #764ba2;">{percentage:.1f}%</span>
                            </div>
                            <div style="background: #e9ecef; height: 6px; border-radius: 3px;">
                                <div style="background: #764ba2; height: 6px; border-radius: 3px; width: {percentage}%;"></div>
                            </div>
                        </div>
            """
        
        html += """
                    </div>
                </div>
            </div>
        </div>
        """
    
    html += "</div>"
    return html

def create_interface():
    """Create the comprehensive Gradio interface with navigation."""
    
    # Simple theme without invalid parameters
    custom_theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="purple",
    )

    with gr.Blocks(title="AI Content Evaluator", theme=custom_theme, css="""
        .results-table, .model-table {
            border-collapse: collapse;
            width: 100%;
            margin-top: 8px;
        }
        table, th, td {
            border: 1px solid #334155;
            padding: 8px;
        }
        th {
            background-color: #475569;
            color: #f1f5f9;
        }
        tr:hover {
            background-color: #334155;
        }
        #status-output {
            font-size: 14px !important;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace !important;
            background-color: #1a1a1a !important;
            border: 2px solid #667eea !important;
            border-radius: 12px !important;
            padding: 16px !important;
            min-height: 120px !important;
            max-height: 500px !important;
            overflow-y: auto !important;
        }
        #status-output .prose, #status-output div {
            background-color: transparent !important;
            color: inherit !important;
            font-size: 14px !important;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace !important;
            line-height: 1.5 !important;
            border: none !important;
        }
        .status-collapsed {
            min-height: auto !important;
        }
    """) as interface:
        
        # Header
        with gr.Row():
            gr.Markdown(
                """
                <div style="text-align: center; margin-top: 16px; margin-bottom: 16px;">
                    <h1 style="font-size: 1.8em; font-weight: bold; margin-bottom: 8px;">AI Content Evaluator</h1>
                    <p style="font-size: 0.9em; margin-bottom: 8px;">
                        Upload PowerPoint presentations and get <b>AI-powered evaluation scores</b>
                    </p>
                    <hr style="margin-top: 16px; margin-bottom: 0;">
                </div>
                """,
                elem_id="header"
            )

        # File Upload & Actions
        with gr.Row(equal_height=True):
            with gr.Column(scale=3):  # Made larger from scale=1
                gr.Markdown("### **Upload Your PPT Files** (All uploaded files must be from the same track)")
                file_upload = gr.File(
                    label="UPLOAD PPT FILES",
                    file_count="multiple",
                    file_types=[".ppt", ".pptx"],
                    height=350,  # Made much larger from 200
                    elem_id="custom-file-upload"
                )
                
                theme_dropdown = gr.Dropdown(
                    label="Select Evaluation Track",
                    choices=get_track_choices(),
                    value=None,  # No default value since no predefined tracks
                    info="Select a track or create a custom one in the Tracks tab"
                )
                
                # Add custom CSS for more rounded edges
                interface_css = """
                #custom-file-upload .form-control, 
                #custom-file-upload .upload-box, 
                #custom-file-upload input[type="file"] {
                    border-radius: 18px !important;
                }
                """
                interface.css += interface_css
                evaluate_btn = gr.Button(
                    "Start Evaluation",
                    variant="primary",
                    size="lg"
                )
            with gr.Column(scale=1):  # Made smaller from scale=2
                status_output = gr.HTML(
                    label="Evaluation Progress",
                    value="<em>Click 'Start Evaluation' to begin processing...</em>",
                    visible=True,
                    elem_id="status-output"
                )

        # Navigation Tabs
        with gr.Tabs() as tabs:
            
            # Submissions Tab
            with gr.Tab("Submissions", id="submissions"):
                gr.Markdown("### **Submission Rankings and Model Comparisons**")
                submissions_output = gr.HTML(
                    label="Submissions Results",
                )
                per_submission_output = gr.HTML(
                    label="Per-Submission Model Scores",
                )
            
            # Models Tab
            with gr.Tab("Models", id="models"):
                gr.Markdown("### **Individual Model Performance Analysis**")
                with gr.Row():
                    with gr.Column(scale=1):
                        model_radio = gr.Radio(
                            label="Select Model",
                            choices=[],
                            value=None
                        )
                    with gr.Column(scale=3):
                        model_output = gr.HTML(
                            label="Model Scores",
                            value="<em>Please select a model to view scores.</em>"
                        )
            
            with gr.Tab("Analytics", id="analytics"):
                gr.Markdown("### **Visual Analytics and Comparisons**")
                with gr.Row():
                    refresh_analytics_btn = gr.Button("Refresh Charts", variant="secondary")
                analytics_output = gr.HTML(
                    label="Analytics Charts",
                    value="<em>No data available for analytics.</em>"
                )
            
            # Certificate Verification Tab
            with gr.Tab("Certificates", id="certificates"):
                gr.Markdown("### **Certificate Verification Results**")
                certificate_output = gr.HTML(
                    label="Certificate Verification",
                    value="<em>No certificate verification data available.</em>"
                )
            
            with gr.Tab("Tracks", id="tracks"):
                gr.Markdown("### **Track Management & Configuration**")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### **Create New Track**")
                        
                        track_name = gr.Textbox(
                            label="Track Name",
                            placeholder="e.g., Fintech Innovation",
                            info="Give your track a descriptive name"
                        )
                        
                        track_description = gr.Textbox(
                            label="Track Description", 
                            placeholder="e.g., Focus on financial technology and innovation",
                            info="Describe the focus area of this track"
                        )
                        
                        gr.Markdown("**Dimension Weights** (must sum to 1.0)")
                        with gr.Row():
                            uniqueness_weight = gr.Number(
                                label="Uniqueness",
                                value=0.25,
                                minimum=0,
                                maximum=1,
                                step=0.01
                            )
                            completeness_weight = gr.Number(
                                label="Completeness",
                                value=0.30,
                                minimum=0,
                                maximum=1,
                                step=0.01
                            )
                        with gr.Row():
                            impact_weight = gr.Number(
                                label="Impact on Theme",
                                value=0.30,
                                minimum=0,
                                maximum=1,
                                step=0.01
                            )
                            ethics_weight = gr.Number(
                                label="Ethical Consideration",
                                value=0.15,
                                minimum=0,
                                maximum=1,
                                step=0.01
                            )
                        
                        gr.Markdown("**Section Weights** (must sum to 1.0)")
                        with gr.Row():
                            problem_weight = gr.Number(
                                label="Problem Statement",
                                value=0.30,
                                minimum=0,
                                maximum=1,
                                step=0.01
                            )
                            solution_weight = gr.Number(
                                label="Proposed Solution",
                                value=0.35,
                                minimum=0,
                                maximum=1,
                                step=0.01
                            )
                            architecture_weight = gr.Number(
                                label="Technical Architecture",
                                value=0.35,
                                minimum=0,
                                maximum=1,
                                step=0.01
                            )
                        
                        with gr.Row():
                            create_track_btn = gr.Button("Create Track", variant="primary")
                            refresh_tracks_btn = gr.Button("Refresh Tracks", variant="secondary")
                        
                        track_status = gr.HTML(
                            value="<em>Configure weights and click 'Create Track'</em>"
                        )
                        
                        gr.Markdown("#### **Delete Track**")
                        delete_track_dropdown = gr.Dropdown(
                            label="Select Track to Delete",
                            choices=get_track_choices(),
                            info="Select any track to delete"
                        )
                        delete_track_btn = gr.Button("Delete Track", variant="stop")
                    
                    with gr.Column(scale=2):
                        track_analytics_output = gr.HTML(
                            label="Track Analytics",
                            value=get_track_analytics()
                        )

        # Event Handlers
        def start_and_run_evaluation(files, theme):
            print(f"Button clicked! Files received: {files}, Theme: {theme}")
            if not files:
                print("No files provided")
                return "Please upload at least one PPT file", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            
            if not theme:
                error_html = """
                <div style="background: #f8d7da; color: #721c24; padding: 15px; border-radius: 8px; border: 1px solid #f5c6cb;">
                    <strong>❌ No Track Selected</strong><br>
                    Please create and select a track in the Tracks tab before starting evaluation.
                </div>
                """
                return error_html, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            
            print(f"Processing {len(files)} files with {theme} theme...")
            
            # Step 1: Initial setup
            setup_html = f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 8px; margin: 5px 0;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong>Starting Evaluation</strong><br>
                        <small>Track: {theme} | Files: {len(files)}</small>
                    </div>
                    <div style="width: 20px; height: 20px; border: 2px solid #fff; border-top: 2px solid transparent; border-radius: 50%; animation: spin 1s linear infinite;"></div>
                </div>
            </div>
            <style>@keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}</style>
            """
            yield setup_html, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            
            # Step 2: File processing
            try:
                temp_dir = pipeline.setup_temp_environment()
                saved_files = pipeline.save_uploaded_files(files)
                
                file_process_html = setup_html + f"""
                <div style="background: #d1ecf1; color: #0c5460; padding: 12px; border-radius: 8px; margin: 5px 0;">
                    <strong>Files Processed</strong><br>
                    <small>Saved {len(saved_files)} files to processing directory</small>
                </div>
                """
                yield file_process_html, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                
                # Step 3: Document processing
                doc_process_html = file_process_html + f"""
                <div style="background: #fff3cd; color: #856404; padding: 12px; border-radius: 8px; margin: 5px 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>Document Processing</strong><br>
                            <small>Extracting text and images...</small>
                        </div>
                        <div style="width: 16px; height: 16px; border: 2px solid #856404; border-top: 2px solid transparent; border-radius: 50%; animation: spin 1s linear infinite;"></div>
                    </div>
                </div>
                """
                yield doc_process_html, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                
                success, msg = pipeline.run_document_processing()
                if not success:
                    error_html = doc_process_html + f"""
                    <div style="background: #f8d7da; color: #721c24; padding: 12px; border-radius: 8px; margin: 5px 0;">
                        <strong>Document Processing Failed</strong><br>
                        <small>{msg}</small>
                    </div>
                    """
                    yield error_html, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                    return
                
                eval_start_html = doc_process_html + f"""
                <div style="background: #d4edda; color: #155724; padding: 12px; border-radius: 8px; margin: 5px 0;">
                    <strong>Document Processing Complete</strong><br>
                    <small>Text and images extracted successfully</small>
                </div>
                <div style="background: #e2e3e5; color: #383d41; padding: 12px; border-radius: 8px; margin: 5px 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>AI Evaluation Starting</strong><br>
                            <small>Running evaluation with {theme} track settings...</small>
                        </div>
                        <div style="width: 16px; height: 16px; border: 2px solid #383d41; border-top: 2px solid transparent; border-radius: 50%; animation: spin 1s linear infinite;"></div>
                    </div>
                </div>
                """
                yield eval_start_html, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                
                success, eval_msg = pipeline.run_evaluation(theme)
                if not success:
                    error_html = eval_start_html + f"""
                    <div style="background: #f8d7da; color: #721c24; padding: 12px; border-radius: 8px; margin: 5px 0;">
                        <strong>AI Evaluation Failed</strong><br>
                        <small>{eval_msg}</small>
                    </div>
                    """
                    yield error_html, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                    return
                
                # Step 5: Loading results
                results_load_html = eval_start_html + f"""
                <div style="background: #d4edda; color: #155724; padding: 12px; border-radius: 8px; margin: 5px 0;">
                    <strong>AI Evaluation Complete</strong><br>
                    <small>Scoring and analysis finished</small>
                </div>
                <div style="background: #cce5ff; color: #004085; padding: 12px; border-radius: 8px; margin: 5px 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>Loading Results</strong><br>
                            <small>Preparing final rankings...</small>
                        </div>
                        <div style="width: 16px; height: 16px; border: 2px solid #004085; border-top: 2px solid transparent; border-radius: 50%; animation: spin 1s linear infinite;"></div>
                    </div>
                </div>
                """
                yield results_load_html, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                
                # Load and refresh results
                results = pipeline.load_results()
                
                if 'main_scores' in results:
                    # Final success with results refresh
                    success_html = f"""
                    <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white; padding: 20px; border-radius: 12px; text-align: center;">
                        <h3>Evaluation Complete!</h3>
                        <strong>Submissions:</strong> {len(results['main_scores'])}</p>
                        <p>Results loaded successfully. Check the Submissions tab for rankings!</p>
                    </div>
                    """
                    
                    # Refresh all data
                    submissions_html = get_submissions_table()
                    per_submission_html = get_per_submission_model_scores()
                    models = get_available_models()
                    first_model = models[0] if models else None
                    model_scores_html = get_model_scores_table(first_model) if first_model else "<p>No models available</p>"
                    analytics_html = generate_analytics_charts()
                    certificate_html = get_certificate_verification_results()
                    
                    yield (
                        success_html,
                        submissions_html,
                        per_submission_html,
                        gr.update(choices=models, value=first_model),
                        model_scores_html,
                        analytics_html,
                        certificate_html
                    )
                else:
                    error_html = results_load_html + f"""
                    <div style="background: #f8d7da; color: #721c24; padding: 12px; border-radius: 8px; margin: 5px 0;">
                        <strong>No Results Found</strong><br>
                        <small>Evaluation completed but no results were generated</small>
                    </div>
                    """
                    yield error_html, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                    
            except Exception as e:
                error_html = f"""
                <div style="background: linear-gradient(135deg, #dc3545 0%, #c82333 100%); color: white; padding: 20px; border-radius: 12px;">
                    <h3>Processing Failed</h3>
                    <p><strong>Error:</strong> {str(e)}</p>
                    <small>Please try again or check the file formats</small>
                </div>
                """
                yield error_html, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        
        def refresh_submissions():
            submissions_html = get_submissions_table()
            per_submission_html = get_per_submission_model_scores()
            return submissions_html, per_submission_html
        
        def refresh_certificates():
            return get_certificate_verification_results()
        
        def refresh_models():
            models = get_available_models()
            return gr.update(choices=models, value=models[0] if models else None)
        
        def update_model_scores(selected_model):
            return get_model_scores_table(selected_model)
        
        def refresh_analytics():
            """Refresh analytics with debug information."""
            print("Debug: Refreshing analytics...")
            print(f"Debug: Pipeline results available: {bool(pipeline.results)}")
            if pipeline.results:
                print(f"Debug: Keys in results: {list(pipeline.results.keys())}")
                if 'main_scores' in pipeline.results:
                    main_df = pipeline.results['main_scores']
                    print(f"Debug: Main scores shape: {main_df.shape}")
                    print(f"Debug: Main scores columns: {list(main_df.columns)}")
                if 'per_model_scores' in pipeline.results:
                    per_model_df = pipeline.results['per_model_scores']
                    print(f"Debug: Per-model scores shape: {per_model_df.shape}")
                    print(f"Debug: Per-model scores columns: {list(per_model_df.columns)}")
            
            result = generate_analytics_charts()
            print(f"Debug: Analytics result length: {len(result)}")
            return result
        
        # Button events
        evaluate_btn.click(
            fn=start_and_run_evaluation,
            inputs=[file_upload, theme_dropdown],
            outputs=[status_output, submissions_output, per_submission_output, model_radio, model_output, analytics_output, certificate_output]
        )

        model_radio.change(
            fn=update_model_scores,
            inputs=[model_radio],
            outputs=[model_output]
        )
        
        refresh_analytics_btn.click(
            fn=refresh_analytics,
            outputs=[analytics_output]
        )
        
        # Track management event handlers
        def handle_create_track(name, description, uniqueness_w, completeness_w, impact_w, ethics_w,
                               problem_w, solution_w, architecture_w):
            """Handle track creation."""
            if not name.strip():
                return "<div style='color: red;'>Please enter a track name</div>", gr.update(), gr.update(), gr.update()
            
            success, message = create_track(
                name.strip(), description.strip(),
                uniqueness_w, completeness_w, impact_w, ethics_w,
                problem_w, solution_w, architecture_w
            )
            
            if success:
                # Refresh all track-related components
                new_choices = get_track_choices()
                
                return (
                    f"<div style='color: green; padding: 10px; background: #d4edda; border-radius: 5px;'>{message}</div>",
                    gr.update(choices=new_choices),
                    get_track_analytics(),
                    gr.update(choices=new_choices)  # Refresh main dropdown
                )
            else:
                return (
                    f"<div style='color: red; padding: 10px; background: #f8d7da; border-radius: 5px;'>{message}</div>",
                    gr.update(),
                    gr.update(),
                    gr.update()
                )
        
        def handle_delete_track(track_id):
            """Handle track deletion."""
            if not track_id:
                return "<div style='color: red;'>Please select a track to delete</div>", gr.update(), gr.update(), gr.update()
            
            success, message = delete_track(track_id)
            
            if success:
                # Refresh all track-related components
                new_choices = get_track_choices()
                
                return (
                    f"<div style='color: green; padding: 10px; background: #d4edda; border-radius: 5px;'>{message}</div>",
                    gr.update(choices=new_choices, value=None),
                    get_track_analytics(),
                    gr.update(choices=new_choices)  # Refresh main dropdown
                )
            else:
                return (
                    f"<div style='color: red; padding: 10px; background: #f8d7da; border-radius: 5px;'>{message}</div>",
                    gr.update(),
                    gr.update(),
                    gr.update()
                )
        
        def refresh_track_data():
            """Refresh all track-related data."""
            new_choices = get_track_choices()
            
            return (
                gr.update(choices=new_choices),
                get_track_analytics(),
                gr.update(choices=new_choices)  # Refresh main dropdown
            )
        
        create_track_btn.click(
            fn=handle_create_track,
            inputs=[
                track_name, track_description,
                uniqueness_weight, completeness_weight, impact_weight, ethics_weight,
                problem_weight, solution_weight, architecture_weight
            ],
            outputs=[track_status, delete_track_dropdown, track_analytics_output, theme_dropdown]
        )
        
        delete_track_btn.click(
            fn=handle_delete_track,
            inputs=[delete_track_dropdown],
            outputs=[track_status, delete_track_dropdown, track_analytics_output, theme_dropdown]
        )
        
        refresh_tracks_btn.click(
            fn=refresh_track_data,
            outputs=[delete_track_dropdown, track_analytics_output, theme_dropdown]
        )

    return interface

def main():
    """Main function to launch the application."""
    try:
        # Check if required dependencies are available
        required_files = ["ai_evaluator_pipeline.py", "report_generator.py"]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            print(f"Missing required files: {missing_files}")
            print("Please ensure you're running from the correct directory with all pipeline files.")
            return
        
        # Create and launch interface
        interface = create_interface()

        import socket
        
        def find_free_port(start_port=7860, max_port=7870):
            for port in range(start_port, max_port + 1):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    try:
                        s.bind(('localhost', port))
                        return port
                    except OSError:
                        continue
            return None
        
        free_port = find_free_port()
        if free_port is None:
            print("Could not find a free port between 7860-7870")
            print("Try stopping other Gradio applications or use a different port range")
            return
        
        print(f"Using port {free_port}")
        interface.launch(
            server_name="0.0.0.0",
            server_port=free_port,
            share=False,
            debug=True
        )
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Failed to start application: {e}")
    finally:
        # Cleanup
        pipeline.cleanup()

if __name__ == "__main__":
    main()
