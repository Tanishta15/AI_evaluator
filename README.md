# AI Idea Evaluator - Complete PPT Processing and Evaluation Pipeline

## 🚀 Overview

The AI Idea Evaluator is a comprehensive pipeline for PowerPoint presentation analysis and evaluation. It combines advanced document processing, OCR text extraction from images, and multi-model AI evaluation to provide detailed scoring across multiple criteria including technical architecture, problem statements, and proposed solutions.

## ✨ Key Features

- **🎯 PPT-Focused Processing**: Direct PowerPoint analysis with accurate slide counting and content extraction using Docling
- **🔍 Dynamic Content Extraction**: No hardcoded values - automatically adapts to any PPT structure
- **🖼️ Smart Image Extraction & OCR**: Extracts meaningful images and uses pytesseract to extract text from technical diagrams
- **🧠 AI-Powered Evaluation**: Multi-model ensemble evaluation using IBM Granite and Meta LLaMA models
- **📊 Technical Architecture Analysis**: Specialized OCR processing for architecture diagrams with automatic keyword detection
- **👥 Team Member Detection**: Automatically identifies team member names from presentation content
- **📁 Clean Directory Structure**: Creates organized output with parquet files, images, and evaluation results
- **🎨 Web Interface**: Beautiful Gradio-based frontend for easy file upload and results visualization
- **⚙️ IBM Data Prep Kit Integration**: Advanced text preprocessing with deduplication, quality filtering, and chunking

## 🆕 Recent Improvements

- ✅ **OCR Integration**: Automatic text extraction from technical architecture diagrams
- ✅ **Model Optimization**: Replaced unreliable models with consistent IBM Granite models
- ✅ **Improved Consistency**: Fixed value inconsistencies across different display areas
- ✅ **Enhanced Frontend**: Higher precision display and debugging capabilities
- ✅ **Better Error Handling**: Robust error handling for OCR and model evaluation processes
- ✅ **IBM Data Prep Kit Integration**: Full DPK framework integration with custom transforms for text deduplication and quality filtering

## 🚀 Quick Start

### Option 1: Web Interface (Recommended)
```bash
# Install dependencies
pip install -r requirements.txt

# Launch the web interface
python frontend.py
```
Then open your browser and navigate to the displayed URL (usually http://localhost:7860)

### Option 2: Command Line
```bash
# Process PowerPoint files with default settings
python ai_evaluator_pipeline.py

# Process specific input/output directories
python ai_evaluator_pipeline.py --input input_submissions --output pipeline_output

# Run evaluation on processed files
python report_generator.py --input_dir pipeline_output --out_prefix evaluation_results

# View help
python ai_evaluator_pipeline.py --help
```

## 🎯 Evaluation Criteria

The system evaluates presentations across multiple dimensions:

- **Problem Statement**: Uniqueness, completeness, impact, and ethical considerations
- **Proposed Solution**: Technical feasibility, innovation, and implementation details  
- **Technical Architecture**: System design, scalability, and technical depth (includes OCR analysis of diagrams)

Each section is scored by multiple AI models and aggregated for consistent results.

## 📁 Project Structure

```
AI_evaluator/
├── ai_evaluator_pipeline.py       # Main pipeline with OCR integration
├── report_generator.py            # Multi-model AI evaluation engine
├── frontend.py                    # Web interface (Gradio-based)
├── image.py                       # OCR text extraction from images
├── config_template.json           # Configuration template
├── requirements.txt                # Dependencies
├── doc_processing/                # Core processing modules
│   ├── main_processor.py          # Document processor with Docling integration
│   ├── config/settings.py         # Configuration management
│   └── utils/                     # Processing utilities
│       ├── enhanced_content_extractor.py  # Dynamic PPT extraction
│       ├── embedding_generator.py         # Embedding generation with criteria mapping
│       ├── vector_storage.py             # ChromaDB vector storage
│       ├── dpk_preprocessor.py           # IBM Data Prep Kit preprocessing with custom transforms
│       └── file_handler.py              # File management
├── data_processing/               # IBM Data Prep Kit framework (actively used)
│   ├── transform/                 # Core transform modules
│   ├── utils/                     # DPK utilities and configurations
│   ├── data_access/              # Data access abstractions
│   └── runtime/                   # Transform execution framework
├── input_submissions/             # Default input directory
├── pipeline_output/               # Processed files with parquet + images
├── vector_db/                     # Vector database storage
└── analysis_output/               # Evaluation results and reports
```
│   └── MindEase.pptx             # Example PPT file
└── pipeline_output/              # Clean output structure
    ├── COSMIC_CODERS/            # PPT-specific folder
    │   ├── COSMIC_CODERS.parquet # Extracted content
    │   └── images/               # Meaningful images only
    └── MindEase/                 # PPT-specific folder
        ├── MindEase.parquet      # Extracted content
        └── images/               # Meaningful images only
```

## ⚙️ Configuration

### Command Line Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--input` | `-i` | `./input_submissions` | Input directory containing PPT/PPTX files |
| `--output` | `-o` | `./pipeline_output` | Output directory for PPT-specific results |
| `--config` | `-c` | None | JSON configuration file path |
| `--log-level` | | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

## 🔧 Processing Pipeline

### 1. PPT File Discovery
- **Format Support**: PowerPoint files (PPT/PPTX) only
- **Dynamic Detection**: Automatically finds all PPT files in input directory
- **No Hardcoding**: Adapts to any PPT structure and content

### 2. Direct PPT Processing
- **Python-pptx Integration**: Direct slide-by-slide analysis without conversion
- **Docling OCR**: Advanced table and image recognition for visual content
- **IBM Data Prep Kit Processing**: Advanced text preprocessing with deduplication, quality filtering, and chunking
- **Dynamic Extraction**: Automatically detects slide count, titles, content, and team members
- **Smart Image Filtering**: Extracts only meaningful images based on size and quality criteria

### 3. Content Structuring
- **Slide-Based Organization**: Each slide becomes a separate record with complete metadata
- **Team Member Detection**: Automatically identifies team member names from content
- **Clean Text Processing**: Separates titles from content, removes formatting artifacts
- **Image Management**: Saves filtered images with descriptive filenames

### 4. Output Generation
- **PPT-Specific Folders**: Each PPT gets its own directory (e.g., `COSMIC_CODERS/`, `MindEase/`)
- **Parquet Files**: Structured slide data with all extracted information
- **Images Subfolder**: Only meaningful images saved with proper naming
- **Simple Success Reporting**: Console output with processing statistics only

## 🛠️ Dependencies

```bash
# Core PPT processing
python-pptx>=0.6.21
docling>=2.3.1

# Data processing and storage  
pandas>=1.5.0
pyarrow>=10.0.0
numpy>=1.21.0

# Embedding generation
sentence-transformers
transformers
torch

# Vector storage (in-memory)
chromadb

# Standard libraries
pathlib
logging
json
Pillow  # For image processing
```

## 🚨 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Install missing dependencies
pip install -r requirements.txt

# For specific PPT processing
pip install python-pptx docling
```

**2. PPT Processing Failures** 
- Ensure PPT files are not corrupted or password-protected
- Check file permissions on input/output directories
- Verify files are actual PPT/PPTX format (not renamed files)
- Review console output for specific slide-level errors

**3. Image Extraction Issues**
- Check if PPT contains embedded images (some may be linked externally)
- Verify image sizes meet minimum criteria (100x100 pixels)
- Ensure sufficient disk space for image extraction

**4. Memory Issues**
- Process smaller batches of PPT files
- Monitor system memory usage during large PPT processing
- Consider reducing image quality settings if needed

## 🔄 Recent Optimizations

### Simplified Pipeline
- ✅ **Removed Unnecessary Directories**: No more temp/, reports/, vector_db/, individual_results/, or parquet_results/
- ✅ **No Report Generation**: Streamlined to essential output only (parquet + images)
- ✅ **Direct PPT Processing**: No PPT-to-PDF conversion, processes slides directly
- ✅ **Dynamic Extraction**: Completely removed hardcoded values for slide counts and content
- ✅ **Smart Image Filtering**: Only saves meaningful images, reducing clutter

### Code Quality
- ✅ **No Hardcoded Paths or Values**: Everything is dynamically determined from PPT structure
- ✅ **Clean Directory Structure**: Each PPT gets its own folder with parquet + images only
- ✅ **Efficient Processing**: Direct slide analysis without intermediate file creation
- ✅ **Professional Logging**: Console-only output with clear success/failure indicators

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PPT Files     │───▶│  Direct PPT      │───▶│ IBM Data Prep   │
│  (.ppt/.pptx)   │    │  Processing      │    │ Kit Processing  │
└─────────────────┘    │  (python-pptx +  │    │ (Deduplication  │
                       │   Docling)       │    │ & Quality       │
                       └──────────────────┘    │ Filtering)      │
                                               └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  In-Memory      │◀───│  Embedding      │◀───│ Enhanced        │
│  Vector Store   │    │  Generator      │    │ Content         │
└─────────────────┘    │  (Optional)     │    │ Extractor       │
                       └─────────────────┘    │ (Dynamic)       │
                                │             └─────────────────┘
                                ▼
┌─────────────────┐    ┌──────────────────┐
│  PPT Folder     │    │  Images Folder   │
│  name.parquet   │    │  (Filtered)      │
│ (Slide Data)    │    │  slide_XX_XX.jpg │
└─────────────────┘    └──────────────────┘
```

## 📊 Sample Output

After processing `COSMIC_CODERS.pptx`, you'll get:

```
pipeline_output/
└── COSMIC_CODERS/
    ├── COSMIC_CODERS.parquet    # 12 slides with complete content
    └── images/                  # Only meaningful images
        ├── slide_05_diagram_01.png
        └── slide_06_chart_01.jpg
```

**Parquet Content**: Each row contains slide_number, title, content, total_pages, images_present, extracted_images, image_count, and processing_timestamp.

---

**AI Idea Evaluator** - Simplified PowerPoint processing pipeline with dynamic extraction and clean output structure.

*Built with ❤️ using python-pptx, Docling, and intelligent content filtering*
