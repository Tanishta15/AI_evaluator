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
 
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentEvaluatorPipeline:
    """Clean and focused pipeline for evaluating uploaded content."""
    
    def __init__(self):
        self.temp_dir = None
        self.results = {}
        
    def setup_temp_environment(self) -> str:
        """Setup temporary environment for processing."""
        if self.temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        self.temp_dir = tempfile.mkdtemp(prefix="evaluator_")
        
        # Create required directories
        (Path(self.temp_dir) / "input_submissions").mkdir(exist_ok=True)
        (Path(self.temp_dir) / "pipeline_output").mkdir(exist_ok=True)
        (Path(self.temp_dir) / "evaluation_results").mkdir(exist_ok=True)
        (Path(self.temp_dir) / "scores").mkdir(exist_ok=True)
        
        return self.temp_dir
    
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
            
            cmd = [
                sys.executable, 
                "ai_evaluator_pipeline.py",
                "--input", str(Path(self.temp_dir) / "input_submissions"),
                "--output", str(Path(self.temp_dir) / "pipeline_output"),
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
                # Check if output was actually created
                output_dir = Path(self.temp_dir) / "pipeline_output"
                if output_dir.exists():
                    created_items = list(output_dir.iterdir())
                    if created_items:
                        # Run additional image processing if parquet files were created
                        success_msg = f"Document processing completed! Created {len(created_items)} items."
                        
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
            pipeline_output = Path(self.temp_dir) / "pipeline_output"
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
    
    def run_evaluation(self, theme: str = "default") -> Tuple[bool, str]:
        """Run the evaluation pipeline (report_generator.py) with specified theme."""
        try:
            # Check if there are processed files first
            pipeline_output = Path(self.temp_dir) / "pipeline_output"
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
            
            original_cwd = os.getcwd()
            
            cmd = [
                sys.executable,
                "report_generator.py", 
                "--input_dir", str(evaluation_results),
                "--out_prefix", str(Path(self.temp_dir) / "scores" / "evaluation"),
                "--theme", theme
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes timeout
                cwd=original_cwd
            )
            
            if result.returncode == 0:
                # Check if evaluation results were created
                scores_dir = Path(self.temp_dir) / "scores"
                evaluation_csv = scores_dir / "evaluation.csv"
                if evaluation_csv.exists():
                    return True, f"Evaluation completed with {theme} theme! Processed {len(copied_files)} parquet files and generated results."
                else:
                    return False, f"Evaluation completed but no CSV results generated."
            else:
                return False, f"Evaluation failed: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return False, "Evaluation timed out after 10 minutes"
        except Exception as e:
            return False, f"Evaluation error: {str(e)}"
    
    def load_results(self) -> Dict[str, Any]:
        """Load and return all results, prioritizing top results CSV files."""
        results = {}
        scores_dir = Path(self.temp_dir) / "scores"
        
        # First try to load top 20/15 results if available
        top20_csv = scores_dir / "evaluation_top20.csv"
        top15_csv = scores_dir / "evaluation_top15.csv"
        main_csv = scores_dir / "evaluation.csv"
        
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
            print("No main results CSV found")
        
        # Load top 15 if available for comparison
        if top15_csv.exists():
            results['top15_scores'] = pd.read_csv(top15_csv)
            print(f"Loaded top 15 results from {top15_csv}")
        
        # Load per-model results
        per_model_csv = scores_dir / "evaluation_per_model.csv"
        if per_model_csv.exists():
            results['per_model_scores'] = pd.read_csv(per_model_csv)

        self.results = results
        return results
    
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
        # Setup environment
        temp_dir = pipeline.setup_temp_environment()
        print(f"Temp directory created: {temp_dir}")
        status_msg = f"Setup complete. Processing {len(files)} files with {theme} theme...\n\n"
        
        # Debug file information
        for i, file in enumerate(files):
            print(f"File {i}: {file.name if hasattr(file, 'name') else file}")
        
        # Save uploaded files
        saved_files = pipeline.save_uploaded_files(files)
        print(f"Saved {len(saved_files)} files")
        status_msg += f"Saved {len(saved_files)} files to temp directory\n\n"
        
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
        
        # Step 3: Load results
        status_msg += "Step 3: Loading results...\n"
        results = pipeline.load_results()
        
        if 'main_scores' in results:
            status_msg += f"Pipeline completed successfully! Generated results for {len(results['main_scores'])} submissions using {theme} theme configuration."
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
    """Generate submissions table showing top 15-20 results with size optimization info."""
    if not pipeline.results or 'main_scores' not in pipeline.results:
        return "<p>No data available. Please run evaluation first.</p>"
    
    df = pipeline.results['main_scores'].copy()
    
    # Debug: Print the raw data to ensure consistency
    print(f"Frontend Debug - Main scores data loaded:")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Total submissions: {len(df)}")
    print(f"  Overall score: {df.iloc[0]['overall_score'] if len(df) > 0 else 'N/A'}")
    if len(df) > 0:
        for col in ['problem_statement_total', 'proposed_solution_total', 'technical_architecture_total']:
            if col in df.columns:
                print(f"  {col}: {df.iloc[0][col]}")
    
    # Add rank column properly
    df_sorted = df.sort_values('overall_score', ascending=False).reset_index(drop=True)
    df_sorted.insert(0, 'Rank', range(1, len(df_sorted) + 1))
    
    # LIMIT TO TOP 20 RESULTS ONLY
    top_20_df = df_sorted.head(20).copy()
    
    # Round numeric columns to ensure consistent display
    numeric_cols = top_20_df.select_dtypes(include=[np.number]).columns
    top_20_df[numeric_cols] = top_20_df[numeric_cols].round(3)
    
    # Add header with summary information
    total_submissions = len(df_sorted)
    showing_count = len(top_20_df)
    
    header_html = f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        text-align: center;
    ">
        <h2 style="margin: 0 0 10px 0;">🏆 Final Rankings - Top {showing_count} Presentations</h2>
        <p style="margin: 0; opacity: 0.9;">
            Showing top {showing_count} out of {total_submissions} total submissions
        </p>
        <div style="
            background: rgba(255,255,255,0.2);
            padding: 10px;
            border-radius: 8px;
            margin-top: 15px;
            display: inline-block;
        ">
            <strong>📊 Results saved to CSV files for final review</strong>
        </div>
    </div>
    """
    
    # Add optimal size recommendations
    size_recommendations_html = f"""
    <div style="
        background: rgba(255, 193, 7, 0.15);
        border: 2px solid rgba(255, 193, 7, 0.5);
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 20px;
    ">
        <h3 style="color: #ffc107; margin: 0 0 10px 0; display: flex; align-items: center;">
            <span style="margin-right: 8px;"></span> Optimal PPT Size Guidelines
        </h3>
        <div style="color: #e0e0e0; font-size: 14px;">
            <p><strong>Recommended:</strong> PPT files under 5MB for optimal processing</p>
            <p><strong>Note:</strong> Larger files may have longer processing times and reduced accuracy</p>
            <p><strong>Top 10 optimally-sized presentations:</strong> Files under 5MB are prioritized for detailed review</p>
        </div>
    </div>
    """
    
    # Create responsive card-based layout instead of table
    cards_html = ""
    optimal_size_count = 0
    
    for _, row in top_20_df.iterrows():
        # Determine if this is an optimal size presentation using backend file size data
        file_size_mb = row.get('file_size_mb', 0)
        is_optimal_size = False
        
        if file_size_mb > 0 and file_size_mb <= 5.0:
            is_optimal_size = True
            optimal_size_count += 1
        elif optimal_size_count < 10 and file_size_mb == 0:  # Fallback if no size data
            is_optimal_size = True
            optimal_size_count += 1
        
        rank_color = "#FFD700" if row['Rank'] == 1 else "#C0C0C0" if row['Rank'] == 2 else "#CD7F32" if row['Rank'] == 3 else "#667eea"
        
        # Add special styling for optimal size presentations
        border_style = f"3px solid {rank_color}"
        if is_optimal_size and row['Rank'] <= 10:
            border_style = f"3px solid #28a745"  # Green border for optimal size
        
        cards_html += f"""
        <div style="
            background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
            border: {border_style};
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            position: relative;
        ">
        """
        
        # Add optimal size badge with actual file size
        if is_optimal_size and row['Rank'] <= 10:
            file_size = row.get('file_size_mb', 0)
            size_text = f"({file_size:.1f}MB)" if file_size > 0 else ""
            cards_html += f"""
            <div style="
                position: absolute;
                top: -10px;
                right: 15px;
                background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
                color: white;
                padding: 5px 12px;
                border-radius: 15px;
                font-size: 12px;
                font-weight: bold;
                box-shadow: 0 2px 6px rgba(0,0,0,0.3);
            ">
                OPTIMAL SIZE {size_text}
            </div>
            """
        
        cards_html += f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <div style="
                    background: {rank_color};
                    color: #1a1a1a;
                    padding: 8px 16px;
                    border-radius: 20px;
                    font-weight: bold;
                    font-size: 14px;
                ">
                    Rank #{row['Rank']}
                </div>
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 8px 16px;
                    border-radius: 20px;
                    font-weight: bold;
                    font-size: 16px;
                ">
                    {row['overall_score']:.3f}
                </div>
            </div>
            
            <h3 style="color: #ffffff; margin: 0 0 15px 0; font-size: 18px;">
                {row['submission_id'] if 'submission_id' in row else 'Unknown Submission'}
            </h3>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
        """
        
        # Add all other columns as key-value pairs (except feedback which we'll handle separately)
        for col in top_20_df.columns:
            if col not in ['Rank', 'submission_id', 'overall_score', 'feedback'] and pd.notna(row[col]):
                display_name = col.replace('_', ' ').title().replace('Total', 'Score')
                # Format the value properly with higher precision
                if isinstance(row[col], (int, float)):
                    formatted_value = f"{row[col]:.3f}"
                else:
                    formatted_value = str(row[col])
                
                cards_html += f"""
                <div style="
                    background: rgba(255,255,255,0.1);
                    padding: 12px;
                    border-radius: 8px;
                    border-left: 4px solid #667eea;
                ">
                    <div style="color: #cccccc; font-size: 12px; margin-bottom: 4px;">
                        {display_name}
                    </div>
                    <div style="color: #ffffff; font-weight: bold; font-size: 16px;">
                        {formatted_value}
                    </div>
                </div>
                """
        
        cards_html += """
            </div>
        """
        
        # Add feedback section if available
        if 'feedback' in row and pd.notna(row['feedback']) and str(row['feedback']).strip():
            feedback_text = str(row['feedback']).strip()
            cards_html += f"""
            <div style="
                background: rgba(102, 126, 234, 0.15);
                border: 1px solid rgba(102, 126, 234, 0.3);
                border-radius: 8px;
                padding: 15px;
                margin-top: 15px;
            ">
                <h4 style="color: #667eea; margin: 0 0 10px 0; font-size: 16px; display: flex; align-items: center;">
                    <span style="margin-right: 8px;">💡</span> AI Feedback
                </h4>
                <div style="
                    color: #e0e0e0;
                    font-size: 14px;
                    line-height: 1.5;
                    white-space: pre-line;
                    max-height: 200px;
                    overflow-y: auto;
                    padding: 10px;
                    background: rgba(0,0,0,0.3);
                    border-radius: 6px;
                ">
                    {feedback_text}
                </div>
            </div>
            """
        
        # Add missing requirements section if available
        if 'missing_requirements' in row and pd.notna(row['missing_requirements']) and str(row['missing_requirements']).strip():
            missing_reqs = str(row['missing_requirements']).strip()
            if missing_reqs and missing_reqs != "[]":
                cards_html += f"""
                <div style="
                    background: rgba(244, 67, 54, 0.15);
                    border: 1px solid rgba(244, 67, 54, 0.3);
                    border-radius: 8px;
                    padding: 15px;
                    margin-top: 10px;
                ">
                    <h4 style="color: #f44336; margin: 0 0 10px 0; font-size: 16px; display: flex; align-items: center;">
                        <span style="margin-right: 8px;"></span> Missing Requirements
                    </h4>
                    <div style="
                        color: #ffcdd2;
                        font-size: 14px;
                        line-height: 1.5;
                        padding: 10px;
                        background: rgba(244, 67, 54, 0.1);
                        border-radius: 6px;
                    ">
                        {missing_reqs}
                    </div>
                </div>
                """
        
        # Add track information
        if 'track' in row and pd.notna(row['track']):
            track_name = str(row['track']).replace('_', ' ').title()
            cards_html += f"""
            <div style="
                background: rgba(76, 175, 80, 0.15);
                border: 1px solid rgba(76, 175, 80, 0.3);
                border-radius: 8px;
                padding: 10px;
                margin-top: 10px;
                text-align: center;
            ">
                <span style="color: #4caf50; font-size: 14px; font-weight: bold;">
                    🏆 Track: {track_name}
                </span>
            </div>
            """
        
        cards_html += """
        </div>
        """
    
    # Add final summary section
    summary_html = f"""
    <div style="
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin-top: 30px;
        text-align: center;
    ">
        <h3 style="margin: 0 0 15px 0;">📋 Evaluation Summary</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
            <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px;">
                <div style="font-size: 24px; font-weight: bold; margin-bottom: 5px;">{total_submissions}</div>
                <div style="font-size: 14px; opacity: 0.9;">Total Submissions</div>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px;">
                <div style="font-size: 24px; font-weight: bold; margin-bottom: 5px;">{showing_count}</div>
                <div style="font-size: 14px; opacity: 0.9;">Top Results Shown</div>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px;">
                <div style="font-size: 24px; font-weight: bold; margin-bottom: 5px;">{optimal_size_count}</div>
                <div style="font-size: 14px; opacity: 0.9;">Optimal Size PPTs</div>
            </div>
        </div>
        <div style="
            background: rgba(255,255,255,0.2);
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
        ">
            <strong>📁 All results exported to CSV files for final review and decision making</strong>
        </div>
    </div>
    """
    
    # Return header + size recommendations + cards + summary
    return header_html + size_recommendations_html + cards_html + summary_html

def get_per_submission_model_scores():
    """Generate per-submission tables with all model scores."""
    if not pipeline.results or 'per_model_scores' not in pipeline.results:
        return "<p>No per-model data available.</p>"
    
    per_model_df = pipeline.results['per_model_scores']
    html = ""
    
    # Get unique submissions
    if 'submission_id' in per_model_df.columns:
        submissions = per_model_df['submission_id'].unique()
        
        for submission in submissions:
            submission_data = per_model_df[per_model_df['submission_id'] == submission]
            
            # Create pivot table for this submission
            if 'evaluator_model' in submission_data.columns and 'section_total' in submission_data.columns:
                pivot_df = submission_data.pivot_table(
                    index='section',
                    columns='evaluator_model', 
                    values='section_total',
                    aggfunc='mean'
                ).round(3)  # Increased precision for consistency
                
                html += f"""
                <div style="background: linear-gradient(135deg, #764ba2 0%, #667eea 100%); color: white; padding: 12px; margin: 20px 0 5px 0; border-radius: 6px;">
                    <h4>{submission} - Model Comparison</h4>
                </div>
                """
                html += pivot_df.to_html(classes="model-scores-table", escape=False)
    
    # Add styling
    styled_html = f"""
    <style>
        .model-scores-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-family: 'Segoe UI', sans-serif;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
            font-size: 13px;
        }}
        .model-scores-table th, .model-scores-table td {{
            padding: 10px 12px;
            text-align: center;
            border-bottom: 1px solid #e0e0e0;
        }}
        .model-scores-table th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
        }}
        .model-scores-table td {{
            background-color: #1a1a1a;
            color: #ffffff;
        }}
        .model-scores-table tr:nth-child(even) td {{
            background-color: #2a2a2a;
        }}
        .model-scores-table tr:hover td {{
            background-color: #3a3a3a;
        }}
    </style>
    {html}
    """
    
    return styled_html

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
                return "Please upload at least one PPT file", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            
            if not theme:
                error_html = """
                <div style="background: #f8d7da; color: #721c24; padding: 15px; border-radius: 8px; border: 1px solid #f5c6cb;">
                    <strong>❌ No Track Selected</strong><br>
                    Please create and select a track in the Tracks tab before starting evaluation.
                </div>
                """
                return error_html, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            
            print(f"Processing {len(files)} files with {theme} theme...")
            
            # Step 1: Initial setup
            setup_html = f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 8px; margin: 5px 0;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong>🚀 Starting Evaluation</strong><br>
                        <small>Track: {theme} | Files: {len(files)}</small>
                    </div>
                    <div style="width: 20px; height: 20px; border: 2px solid #fff; border-top: 2px solid transparent; border-radius: 50%; animation: spin 1s linear infinite;"></div>
                </div>
            </div>
            <style>@keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}</style>
            """
            yield setup_html, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            
            # Step 2: File processing
            try:
                temp_dir = pipeline.setup_temp_environment()
                saved_files = pipeline.save_uploaded_files(files)
                
                file_process_html = setup_html + f"""
                <div style="background: #d1ecf1; color: #0c5460; padding: 12px; border-radius: 8px; margin: 5px 0;">
                    <strong>📁 Files Processed</strong><br>
                    <small>Saved {len(saved_files)} files to processing directory</small>
                </div>
                """
                yield file_process_html, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                
                # Step 3: Document processing
                doc_process_html = file_process_html + f"""
                <div style="background: #fff3cd; color: #856404; padding: 12px; border-radius: 8px; margin: 5px 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>📄 Document Processing</strong><br>
                            <small>Extracting text and images...</small>
                        </div>
                        <div style="width: 16px; height: 16px; border: 2px solid #856404; border-top: 2px solid transparent; border-radius: 50%; animation: spin 1s linear infinite;"></div>
                    </div>
                </div>
                """
                yield doc_process_html, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                
                success, msg = pipeline.run_document_processing()
                if not success:
                    error_html = doc_process_html + f"""
                    <div style="background: #f8d7da; color: #721c24; padding: 12px; border-radius: 8px; margin: 5px 0;">
                        <strong>❌ Document Processing Failed</strong><br>
                        <small>{msg}</small>
                    </div>
                    """
                    yield error_html, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
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
                yield eval_start_html, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                
                success, eval_msg = pipeline.run_evaluation(theme)
                if not success:
                    error_html = eval_start_html + f"""
                    <div style="background: #f8d7da; color: #721c24; padding: 12px; border-radius: 8px; margin: 5px 0;">
                        <strong>AI Evaluation Failed</strong><br>
                        <small>{eval_msg}</small>
                    </div>
                    """
                    yield error_html, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
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
                            <strong>📊 Loading Results</strong><br>
                            <small>Preparing final rankings...</small>
                        </div>
                        <div style="width: 16px; height: 16px; border: 2px solid #004085; border-top: 2px solid transparent; border-radius: 50%; animation: spin 1s linear infinite;"></div>
                    </div>
                </div>
                """
                yield results_load_html, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                
                # Load and refresh results
                results = pipeline.load_results()
                
                if 'main_scores' in results:
                    # Final success with results refresh
                    success_html = f"""
                    <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white; padding: 20px; border-radius: 12px; text-align: center;">
                        <h3>🎉 Evaluation Complete!</h3>
                        <p><strong>Track:</strong> {theme} | <strong>Submissions:</strong> {len(results['main_scores'])}</p>
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
                    
                    yield (
                        success_html,
                        submissions_html,
                        per_submission_html,
                        gr.update(choices=models, value=first_model),
                        model_scores_html,
                        analytics_html
                    )
                else:
                    error_html = results_load_html + f"""
                    <div style="background: #f8d7da; color: #721c24; padding: 12px; border-radius: 8px; margin: 5px 0;">
                        <strong>❌ No Results Found</strong><br>
                        <small>Evaluation completed but no results were generated</small>
                    </div>
                    """
                    yield error_html, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                    
            except Exception as e:
                error_html = f"""
                <div style="background: linear-gradient(135deg, #dc3545 0%, #c82333 100%); color: white; padding: 20px; border-radius: 12px;">
                    <h3>❌ Processing Failed</h3>
                    <p><strong>Error:</strong> {str(e)}</p>
                    <small>Please try again or check the file formats</small>
                </div>
                """
                yield error_html, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        
        def refresh_submissions():
            submissions_html = get_submissions_table()
            per_submission_html = get_per_submission_model_scores()
            return submissions_html, per_submission_html
        
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
            outputs=[status_output, submissions_output, per_submission_output, model_radio, model_output, analytics_output]
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
        
        print("Starting AI Content Evaluator...")
        print("Upload PPT files and navigate between tabs to view results")
        print("Recommendation: Keep PPT files under 5MB for optimal processing")
        print("Final results: Only top 15-20 presentations will be shown with CSV export")
        
        # Try to find an available port
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
