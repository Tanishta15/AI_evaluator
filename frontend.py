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
    
    def run_evaluation(self) -> Tuple[bool, str]:
        """Run the evaluation pipeline (report_generator.py)."""
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
                "--out_prefix", str(Path(self.temp_dir) / "scores" / "evaluation")
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
                    return True, f"Evaluation completed! Processed {len(copied_files)} parquet files and generated results."
                else:
                    return False, f"Evaluation completed but no CSV results generated."
            else:
                return False, f"Evaluation failed: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return False, "Evaluation timed out after 10 minutes"
        except Exception as e:
            return False, f"Evaluation error: {str(e)}"
    
    def load_results(self) -> Dict[str, Any]:
        """Load and return all results."""
        results = {}
        scores_dir = Path(self.temp_dir) / "scores"
        
        # Load main evaluation results
        main_csv = scores_dir / "evaluation.csv"
        if main_csv.exists():
            results['main_scores'] = pd.read_csv(main_csv)
        
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

def process_uploads(files) -> str:
    """Process uploaded files through the complete pipeline."""
    print(f"process_uploads called with {len(files) if files else 0} files")
    
    if not files:
        return "No files uploaded"
    
    try:
        # Setup environment
        temp_dir = pipeline.setup_temp_environment()
        print(f"Temp directory created: {temp_dir}")
        status_msg = f"Setup complete. Processing {len(files)} files...\n\n"
        
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
        
        # Step 2: Run evaluation (report_generator.py)
        status_msg += "Step 2: Running evaluation (report_generator.py)...\n"
        success, msg = pipeline.run_evaluation()
        status_msg += msg + "\n\n"
        
        if not success:
            return status_msg
        
        # Step 3: Load results
        status_msg += "Step 3: Loading results...\n"
        results = pipeline.load_results()
        
        if 'main_scores' in results:
            status_msg += f"Pipeline completed successfully! Generated results for {len(results['main_scores'])} submissions"
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
    """Generate submissions table with all models."""
    if not pipeline.results or 'main_scores' not in pipeline.results:
        return "<p>No data available. Please run evaluation first.</p>"
    
    df = pipeline.results['main_scores'].copy()
    
    # Debug: Print the raw data to ensure consistency
    print(f"Frontend Debug - Main scores data loaded:")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Overall score: {df.iloc[0]['overall_score'] if len(df) > 0 else 'N/A'}")
    if len(df) > 0:
        for col in ['problem_statement_total', 'proposed_solution_total', 'technical_architecture_total']:
            if col in df.columns:
                print(f"  {col}: {df.iloc[0][col]}")
    
    # Add rank column properly
    df_sorted = df.sort_values('overall_score', ascending=False).reset_index(drop=True)
    df_sorted.insert(0, 'Rank', range(1, len(df_sorted) + 1))
    
    # Round numeric columns to ensure consistent display
    numeric_cols = df_sorted.select_dtypes(include=[np.number]).columns
    df_sorted[numeric_cols] = df_sorted[numeric_cols].round(3)  # Increased precision for better accuracy
    
    # Create responsive card-based layout instead of table
    cards_html = ""
    for _, row in df_sorted.iterrows():
        rank_color = "#FFD700" if row['Rank'] == 1 else "#C0C0C0" if row['Rank'] == 2 else "#CD7F32" if row['Rank'] == 3 else "#667eea"
        
        cards_html += f"""
        <div style="
            background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
            border: 2px solid {rank_color};
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        ">
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
        
                # Add all other columns as key-value pairs
        for col in df_sorted.columns:
            if col not in ['Rank', 'submission_id', 'overall_score'] and pd.notna(row[col]):
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
        </div>
        """
    
    styled_html = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; margin: 20px 0 10px 0; border-radius: 6px;">
        <h3>Submission Rankings (All Models Combined)</h3>
    </div>
    <div style="max-height: 600px; overflow-y: auto;">
        {cards_html}
    </div>
    """
    
    return styled_html

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
            with gr.Column(scale=1):
                gr.Markdown("### **Upload Your PPT Files**")
                file_upload = gr.File(
                    label="Select PPT/PPTX Files",
                    file_count="multiple",
                    file_types=[".ppt", ".pptx"],
                    height=240,
                    elem_id="custom-file-upload"
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
            with gr.Column(scale=2):
                status_output = gr.HTML(
                    label="Processing Status",
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

        # Event Handlers
        def start_and_run_evaluation(files):
            print(f"Button clicked! Files received: {files}")
            if not files:
                print("No files provided")
                return "Please upload at least one PPT file", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            
            print(f"Processing {len(files)} files...")
            
            # Show immediate processing status
            processing_html = """
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 12px;
                margin: 10px 0;
                text-align: center;
                animation: pulse 2s infinite;
            ">
                <h3>🔄 Processing Your Files...</h3>
                <p>Please wait while we analyze your presentations</p>
                <div style="
                    background: rgba(255,255,255,0.2);
                    padding: 10px;
                    border-radius: 8px;
                    margin-top: 15px;
                    font-family: monospace;
                ">
                    Starting evaluation pipeline...
                </div>
            </div>
            <style>
                @keyframes pulse {
                    0% { opacity: 1; }
                    50% { opacity: 0.7; }
                    100% { opacity: 1; }
                }
            </style>
            """
            
            # Return initial processing state
            yield processing_html, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            
            # Process uploads
            status = process_uploads(files)
            print(f"Processing completed with status: {status[:100]}...")
            
            # After processing, automatically refresh all tabs with results
            if "Pipeline completed successfully!" in status:
                print("Refreshing all tabs with new results...")
                # Refresh submissions data
                submissions_html = get_submissions_table()
                per_submission_html = get_per_submission_model_scores()
                
                # Refresh models data
                models = get_available_models()
                first_model = models[0] if models else None
                model_scores_html = get_model_scores_table(first_model) if first_model else "<p>No models available</p>"
                
                # Refresh analytics
                analytics_html = generate_analytics_charts()
                
                # Create collapsible summary
                collapsed_status = f"""
                <details>
                    <summary style="
                        cursor: pointer;
                        padding: 15px;
                        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
                        color: white;
                        border-radius: 8px;
                        margin: 10px 0;
                        display: block;
                        font-weight: bold;
                    ">
                        ✅ Processing Complete - Click to view details
                    </summary>
                    <div style="
                        padding: 15px;
                        background: #f8f9fa;
                        border-radius: 8px;
                        margin-top: 5px;
                        font-family: monospace;
                        white-space: pre-wrap;
                        max-height: 300px;
                        overflow-y: auto;
                        border: 1px solid #dee2e6;
                    ">{status}</div>
                </details>
                """
                
                yield (
                    collapsed_status,
                    submissions_html,
                    per_submission_html,
                    gr.update(choices=models, value=first_model),
                    model_scores_html,
                    analytics_html
                )
            else:
                # Show error status
                error_html = f"""
                <div style="
                    background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 12px;
                    margin: 10px 0;
                ">
                    <h3>❌ Processing Failed</h3>
                    <div style="
                        background: rgba(255,255,255,0.2);
                        padding: 15px;
                        border-radius: 8px;
                        margin-top: 15px;
                        font-family: monospace;
                        white-space: pre-wrap;
                        max-height: 200px;
                        overflow-y: auto;
                    ">{status}</div>
                </div>
                """
                yield (
                    error_html,
                    gr.update(),
                    gr.update(), 
                    gr.update(),
                    gr.update(),
                    gr.update()
                )
        
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
            inputs=[file_upload],
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
