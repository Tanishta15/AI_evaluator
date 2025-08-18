# Features of the AI Idea Evaluator

## Data Structures
The project uses structured data formats such as Parquet files to store slide-level information extracted from PowerPoint presentations. Each Parquet file contains metadata like slide titles, content, image paths, and processing timestamps. This ensures efficient storage and retrieval of structured data for downstream processing.

### Use in the Project
- *Parquet Files*: Store structured slide data, including titles, content, and image paths.
- *CSV Files*: Generated during evaluation to store aggregated scores and feedback.
- *Image Files*: Extracted from slides and saved in a dedicated folder for OCR processing.

## Models
The project employs a multi-model ensemble approach using advanced AI models for text and vision-based evaluation tasks.

### Use in the Project
- *IBM Granite Models*: Used for text evaluation, focusing on problem statements, solutions, and technical architecture.
- *Meta LLaMA Models*: Provide additional text analysis and scoring capabilities.
- *Mistral Vision Models*: Analyze images, including technical diagrams, to extract insights and provide scores.
- *Ensemble Scoring*: Combines results from multiple models to ensure robust and consistent evaluation.

## IBM Data Prep Kit
The IBM Data Prep Kit (DPK) is integrated into the pipeline for advanced text preprocessing. It handles tasks like deduplication, quality filtering, and chunking to prepare data for AI model evaluation.

### Use in the Project
- *Deduplication*: Removes redundant text from slides to ensure clean input for models.
- *Quality Filtering*: Filters out low-quality or irrelevant text, improving model accuracy.
- *Chunking*: Splits large blocks of text into manageable chunks for better processing by AI models.

## OCR Integration
OCR (Optical Character Recognition) is implemented using pytesseract to extract text from images, particularly technical diagrams embedded in slides.

### Use in the Project
- *pytesseract*: Extracts text from images identified during the document processing phase.
- *Image Identification*: The pipeline identifies images in slides that are likely to contain technical diagrams.
- *Diagram Analysis*: Extracted text is used to evaluate the technical depth and relevance of diagrams.

## Dynamic Content Extraction
The pipeline dynamically adapts to the structure of each PowerPoint presentation, ensuring no hardcoded assumptions about slide layouts.

### Use in the Project
- *Slide Count Detection*: Automatically counts the number of slides in each presentation.
- *Metadata Extraction*: Identifies and extracts key metadata such as team member names and slide titles.
- *Flexible Parsing*: Adapts to varying slide layouts and content structures without requiring manual adjustments.

## Vector Storage
The project optionally uses in-memory vector storage (ChromaDB) for embedding generation and retrieval.

### Use in the Project
- *ChromaDB*: Stores vector embeddings for slide content to enable similarity searches.
- *Embedding Generation*: Creates vector embeddings for slide content to enhance analysis.
- *Efficient Retrieval*: Enables quick access to embeddings during evaluation.

## Web Interface
A Gradio-based web interface provides an easy-to-use platform for uploading presentations and visualizing evaluation results.

### Use in the Project
- *Gradio*: Provides a user-friendly interface for file uploads and result visualization.
- *Interactive Charts*: Displays evaluation scores and insights using Plotly charts and tables.
- *Pipeline Orchestration*: Allows users to trigger document processing and evaluation directly from the interface.

## Evaluation Engine
The evaluation engine aggregates scores from multiple models and generates comprehensive reports.

### Use in the Project
- *Score Computation*: Processes Parquet files to compute scores for problem statements, solutions, and architecture.
- *Report Generation*: Outputs results in CSV and Parquet formats for further analysis.
- *Model Feedback*: Provides detailed feedback from AI models to help users understand the strengths and weaknesses of their presentations.

---

These features collectively enable the AI Idea Evaluator to provide a robust and scalable solution for analyzing and scoring PowerPoint presentations.
