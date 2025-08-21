# hackathon_tri_section_evaluator.py
import os, re, json, base64, glob, argparse, pathlib
from typing import Dict, List, Any, Optional
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).with_suffix('.env'), override=False)  # loads ./hackathon_tri_section_evaluator.env
load_dotenv(override=False)

# watsonx.ai SDK
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models import ModelInference
 
# ==========================
# CONFIG (edit as needed)
# ==========================

MODEL_SCORE_ROWS: List[Dict[str, Any]] = []

TARGET_SECTIONS = ["problem_statement", "proposed_solution", "technical_architecture"]

# Theme-specific scoring configurations with detailed criteria
THEME_CONFIGS = {
    "default": {
        "name": "General Hackathon",
        "dimensions": {
            "Problem Statement": 0.10,
            "Proposed Solution": 0.15,
            "Technical Architecture": 0.15,
            "Novelty  Uniqueness/creativity": 0.15,
            "presentaion": 0.10,
            "ethical consideration": 0.10,
            "completeness of design solution": 0.15,
            "implementing ibm dpk/rag/aws/watsonx/granite": 0.10,
        },
        "section_weights": {
            "problem_statement": 0.30,
            "proposed_solution": 0.35,
            "technical_architecture": 0.35,
        },
        "detailed_criteria": {
            "Problem Statement": [
                "Is the problem statement clearly defined and realistic?",
                "Is the problem well-articulated and meaningful?"
            ],
            "Proposed Solution": [
                "Is the proposed solution comprehensive and well-designed?",
                "Does the solution effectively address the problem statement?"
            ],
            "Technical Architecture": [
                "Is the technical architecture clearly presented?",
                "Are system components and their interactions explained?",
                "Is the technology stack appropriate for the solution?",
                "Are technical diagrams present and clear?"
            ],
            "Novelty  Uniqueness/creativity": [
                "Is the technology suggested feasible to implement?",
                "Is there a working prototype?",
                "How have you tested your solution for any bias?",
                "What is unique/original/creative in this solution?"
            ],
            "presentaion": [
                "Is the presentation clear, concise and structured?",
                "Is 'Why', 'What', and 'How' explained?",
                "Are services used clearly mentioned?",
                "Is the LLM used specified?",
                "Is training content used described?",
                "Is the architecture explained?"
            ],
            "ethical consideration": [
                "Is it ethical to implement the idea?",
                "Have you considered factors like gender neutrality?",
                "Are racial or religious biases being avoided?",
                "What bias testing methodology was used?",
                "How does the solution consider ethical implications?"
            ],
            "completeness of design solution": [
                "How is the solution making things better?",
                "What sort of deviation (positive/negative) can be seen by adopting your solution?",
                "Will it be scalable? Where else can this be used to make an impact?",
                "What can your solution do that just a simple Google search does not do?",
                "How would end-users use it?",
                "What is the potential positive impact on the chosen theme?"
            ],
            "implementing ibm dpk/rag/aws/watsonx/granite": [
                "Have you used one of the must-haves in your solution?",
                "What are the must-have techs that have been used in your solution?",
                "If using Traditional AI, is DPK usage present (mandatory)?",
                "How different is the solution compared to other LLMs being used?",
                "How is this solution better than using LLMs/Gen AI?",
                "Is IBM or AWS or both mentioned and used?",
                "Is RAG/Agentic usage clearly demonstrated?",
                "Are DPK/IBM Granite models used (RAG/Agentic Usage)?"
            ]
        }
    },
    "sustainability": {
        "name": "Sustainability & Climate",
        "dimensions": {
            "uniqueness": 0.20,
            "Completeness of the solution": 0.25,
            "impact on the theme chosen": 0.40,
            "ethical consideration": 0.15,
        },
        "section_weights": {
            "problem_statement": 0.35,
            "proposed_solution": 0.30,
            "technical_architecture": 0.35,
        },
        "detailed_criteria": {
            "impact_focus": [
                "Environmental impact reduction potential",
                "Carbon footprint considerations",
                "Sustainable development goals alignment",
                "Climate change mitigation strategies",
                "Resource optimization and waste reduction"
            ]
        }
    },
    "healthcare": {
        "name": "Healthcare & Life Sciences",
        "dimensions": {
            "uniqueness": 0.20,
            "Completeness of the solution": 0.35,
            "impact on the theme chosen": 0.30,
            "ethical consideration": 0.15,
        },
        "section_weights": {
            "problem_statement": 0.30,
            "proposed_solution": 0.40,
            "technical_architecture": 0.30,
        },
        "detailed_criteria": {
            "healthcare_focus": [
                "Patient safety and care improvement",
                "Medical data privacy compliance",
                "Healthcare accessibility enhancement",
                "Clinical validation and testing",
                "Regulatory compliance considerations"
            ]
        }
    },
    "fintech": {
        "name": "Financial Technology",
        "dimensions": {
            "uniqueness": 0.30,
            "Completeness of the solution": 0.35,
            "impact on the theme chosen": 0.25,
            "ethical consideration": 0.10,
        },
        "section_weights": {
            "problem_statement": 0.25,
            "proposed_solution": 0.35,
            "technical_architecture": 0.40,
        },
        "detailed_criteria": {
            "fintech_focus": [
                "Financial security and fraud prevention",
                "Regulatory compliance (KYC/AML)",
                "Financial inclusion and accessibility",
                "Transaction efficiency and cost reduction",
                "Risk management and assessment"
            ]
        }
    },
    "education": {
        "name": "Education & Learning",
        "dimensions": {
            "uniqueness": 0.25,
            "Completeness of the solution": 0.30,
            "impact on the theme chosen": 0.35,
            "ethical consideration": 0.10,
        },
        "section_weights": {
            "problem_statement": 0.30,
            "proposed_solution": 0.35,
            "technical_architecture": 0.35,
        },
        "detailed_criteria": {
            "education_focus": [
                "Learning outcome improvement",
                "Educational accessibility and inclusion",
                "Personalized learning capabilities",
                "Teacher and student engagement",
                "Educational equity considerations"
            ]
        }
    },
    "ai_ml": {
        "name": "AI & Machine Learning Innovation",
        "dimensions": {
            "uniqueness": 0.35,
            "Completeness of the solution": 0.30,
            "impact on the theme chosen": 0.25,
            "ethical consideration": 0.10,
        },
        "section_weights": {
            "problem_statement": 0.25,
            "proposed_solution": 0.30,
            "technical_architecture": 0.45,
        },
        "detailed_criteria": {
            "ai_focus": [
                "Novel AI/ML algorithm implementation",
                "Model performance and accuracy",
                "AI bias detection and mitigation",
                "Computational efficiency",
                "Real-world AI application impact"
            ]
        }
    }
}

# Global variables to store current configuration
CURRENT_THEME = "default"
DIMENSIONS = THEME_CONFIGS[CURRENT_THEME]["dimensions"]
SECTION_WEIGHTS = THEME_CONFIGS[CURRENT_THEME]["section_weights"]

# Regex for mapping slide titles → sections (when files are slide-wise)
TITLE_PATTERNS = {
    "problem_statement":  [r"\bproblem\b", r"\bpain\s*point\b", r"\bchallenge\b"],
    "proposed_solution":  [r"\bsolution\b", r"\bproposal\b", r"\bapproach\b"],
    "technical_architecture": [r"\barchitecture\b", r"\bblueprint\b", r"\bsystem\s*design\b", r"\btech\s*stack\b"]
}

# Text LLMs to ensemble (change to the model IDs you have)
TEXT_MODEL_IDS = [
    "ibm/granite-3-2-8b-instruct", 
    "ibm/granite-3-2b-instruct", 
    "meta-llama/llama-3-3-70b-instruct",
    "mistralai/mistral-large",  # Replaced granite with Mistral model
]

# Vision LLMs for images (NOT USED - we only use extracted text)
# VISION_MODEL_IDS = [
#     "meta-llama/llama-3-2-90b-vision-instruct",
#     "ibm/granite-vision-3-2-2b",
# ]

GEN_PARAMS = {
    GenParams.MAX_NEW_TOKENS: 1024,  # Increased from 256 to allow complete JSON responses
    GenParams.TEMPERATURE: 0.2,
    GenParams.TOP_P: 0.9,
    GenParams.REPETITION_PENALTY: 1.1,
}

def get_detailed_prompt(section_name: str, section_text: str, theme: str = "default") -> str:
    """Generate detailed evaluation prompt based on theme and section."""
    theme_config = THEME_CONFIGS.get(theme, THEME_CONFIGS["default"])
    base_criteria = theme_config.get("detailed_criteria", {})
    
    # Get criteria for all dimensions
    certification_criteria = base_criteria.get("certification", [
        "Are proper certifications or credentials present?",
        "Is the certification valid and relevant to the solution?"
    ])
    
    novelty_uniqueness_criteria = base_criteria.get("novelty_uniqueness", [
        "Is the problem statement clearly defined and realistic?",
        "Is the technology suggested feasible to implement?",
        "Is there a working prototype?",
        "How have you tested your solution for any bias?",
        "What is unique/original/creative in this solution?"
    ])
    
    presentation_quality_criteria = base_criteria.get("presentation_quality", [
        "Is the presentation clear, concise and structured?",
        "Is 'Why', 'What', and 'How' explained?",
        "Are services used clearly mentioned?",
        "Is the LLM used specified?",
        "Is training content used described?",
        "Is the architecture explained?"
    ])
    
    technical_architecture_criteria = base_criteria.get("technical_architecture_quality", [
        "Is the technical architecture clearly presented?",
        "Are system components and their interactions explained?",
        "Is the technology stack appropriate for the solution?",
        "Are technical diagrams present and clear?"
    ])
    
    ethical_considerations_criteria = base_criteria.get("ethical_considerations", [
        "Is it ethical to implement the idea?",
        "Have you considered factors like gender neutrality?",
        "Are racial or religious biases being avoided?",
        "What bias testing methodology was used?",
        "How does the solution consider ethical implications?"
    ])
    
    impact_scalability_criteria = base_criteria.get("impact_scalability", [
        "How is the solution making things better?",
        "What sort of deviation (positive/negative) can be seen by adopting your solution?",
        "Will it be scalable? Where else can this be used to make an impact?",
        "What can your solution do that just a simple Google search does not do?",
        "How would end-users use it?",
        "What is the potential positive impact on the chosen theme?"
    ])
    
    completeness_implementation_criteria = base_criteria.get("completeness_implementation", [
        "Have you used one of the must-haves in your solution?",
        "What are the must-have techs that have been used in your solution?",
        "If using Traditional AI, is DPK usage present (mandatory)?",
        "How different is the solution compared to other LLMs being used?",
        "How is this solution better than using LLMs/Gen AI?",
        "Is IBM or AWS or both mentioned and used?",
        "Is RAG/Agentic usage clearly demonstrated?",
        "Are DPK/IBM Granite models used (RAG/Agentic Usage)?"
    ])
    
    # Add theme-specific criteria
    theme_specific = ""
    for key, criteria_list in base_criteria.items():
        if key not in ["certification", "novelty_uniqueness", "presentation_quality", "technical_architecture_quality", "ethical_considerations", "impact_scalability", "completeness_implementation"]:
            theme_specific += f"\nTheme-specific considerations ({key}):\n"
            theme_specific += "\n".join(f"- {criterion}" for criterion in criteria_list)
    
    return f"""You are a strict hackathon evaluator for {theme_config['name']} track. 
Rate the SECTION_CONTENT for "{section_name}" based on the following comprehensive criteria. 

SCORE EACH DIMENSION from 0-10 (where 10 is excellent, 0 is missing/poor):

CERTIFICATION (5% weight):
{chr(10).join(f"- {criterion}" for criterion in certification_criteria)}

NOVELTY & UNIQUENESS (15% weight):
{chr(10).join(f"- {criterion}" for criterion in novelty_uniqueness_criteria)}

PRESENTATION QUALITY (15% weight):
{chr(10).join(f"- {criterion}" for criterion in presentation_quality_criteria)}

TECHNICAL ARCHITECTURE QUALITY (20% weight):
{chr(10).join(f"- {criterion}" for criterion in technical_architecture_criteria)}

ETHICAL CONSIDERATIONS (10% weight):
{chr(10).join(f"- {criterion}" for criterion in ethical_considerations_criteria)}

IMPACT & SCALABILITY (15% weight):
{chr(10).join(f"- {criterion}" for criterion in impact_scalability_criteria)}

COMPLETENESS & IMPLEMENTATION (20% weight):
{chr(10).join(f"- {criterion}" for criterion in completeness_implementation_criteria)}

{theme_specific}

CRITICAL REQUIREMENTS TO CHECK:
- YouTube video link (Private with "anybody with the link" access)
- Must mention IBM or AWS or both
- Must-have technologies clearly identified
- Working prototype evidence
- Bias testing methodology
- DPK usage (if Traditional AI)
- RAG/Agentic implementation details

SCORING GUIDELINES:
- 0-2: Missing or severely inadequate
- 3-4: Poor, major issues
- 5-6: Average, some issues
- 7-8: Good, minor issues
- 9-10: Excellent, comprehensive

Rules:
- Judge ONLY what is provided. If missing/irrelevant content => low score.
- Provide specific feedback on what's good and what can be improved.
- Check for required elements and deduct points if missing.
- Respond as pure JSON, exactly:
{{
  "section": "{section_name}",
  "scores": {{
    "Problem Statement": <float 0-10>,
    "Proposed Solution": <float 0-10>,
    "Technical Architecture": <float 0-10>,
    "Novelty  Uniqueness/creativity": <float 0-10>,
    "presentaion": <float 0-10>,
    "ethical consideration": <float 0-10>,
    "completeness of design solution": <float 0-10>,
    "implementing ibm dpk/rag/aws/watsonx/granite": <float 0-10>
  }},
  "notes": "<1-2 line justification>",
  "feedback": {{
    "strengths": "<what's good about this section>",
    "improvements": "<specific areas for improvement, mention missing required elements>"
  }},
  "missing_requirements": [<list of missing required elements>],
  "certification": "<yes/no based on evidence of certification>",
  "overall_feedback": "<comprehensive feedback summary>",
  "missing_content": "<detailed list of anything missing from evaluation criteria>"
}}

SECTION_CONTENT:
\"\"\"{section_text}\"\"\""""

# Vision query (NOT USED - we only use extracted text)
# VISION_USER_QUERY = (
#     "Evaluate this technical architecture image for a hackathon PPT. "
#     "Rate (0-10) the same criteria: uniqueness, Completeness of the solution, impact on the theme chosen, ethical consideration. "
#     "Provide specific feedback on what's good about the diagram and what can be improved. "
#     "Return JSON ONLY with the exact schema used before, including 'feedback' with 'strengths' and 'improvements'. Be concise in 'notes'."
# )

# Column hints when data already normalized
COL_HINTS = {
    "problem_statement": ["problem_statement", "problem", "problem slide"],
    "proposed_solution": ["proposed_solution", "solution"],
    "technical_architecture": ["technical_architecture", "architecture", "tech_arch"],
    "images_present": ["images_present", "contains_image"],
    "tech_arch_image_path": ["tech_arch_image_path", "architecture_image_path", "image_path"],
    "title": ["title", "slide_title"],
    "content": ["content", "text"],
}

class LLMReplyError(Exception): pass

def load_theme_config(theme_name: str = None, custom_config: Dict = None) -> None:
    """Load theme-specific configuration, custom configuration, or custom tracks."""
    global CURRENT_THEME, DIMENSIONS, SECTION_WEIGHTS
    
    # Try to load custom tracks from file
    custom_tracks = {}
    try:
        if os.path.exists("custom_tracks.json"):
            with open("custom_tracks.json", 'r') as f:
                custom_tracks = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load custom tracks: {e}")
    
    # Merge custom tracks with built-in themes
    all_themes = THEME_CONFIGS.copy()
    all_themes.update(custom_tracks)
    
    if custom_config:
        # Use custom configuration
        CURRENT_THEME = "custom"
        DIMENSIONS = custom_config.get("dimensions", THEME_CONFIGS["default"]["dimensions"])
        SECTION_WEIGHTS = custom_config.get("section_weights", THEME_CONFIGS["default"]["section_weights"])
        print(f"Loaded custom configuration")
    elif theme_name and theme_name in all_themes:
        # Use predefined or custom theme
        CURRENT_THEME = theme_name
        DIMENSIONS = all_themes[theme_name]["dimensions"]
        SECTION_WEIGHTS = all_themes[theme_name]["section_weights"]
        track_type = "custom" if theme_name in custom_tracks else "built-in"
        print(f"Loaded {track_type} theme configuration: {theme_name}")
    else:
        # Use default
        CURRENT_THEME = "default"
        DIMENSIONS = THEME_CONFIGS["default"]["dimensions"]
        SECTION_WEIGHTS = THEME_CONFIGS["default"]["section_weights"]
        print(f"Using default configuration (theme '{theme_name}' not found)")
    
    # Validate weights sum to 1.0
    dim_sum = sum(DIMENSIONS.values())
    sec_sum = sum(SECTION_WEIGHTS.values())
    
    if abs(dim_sum - 1.0) > 0.01:
        print(f"Warning: Dimension weights sum to {dim_sum:.3f}, not 1.0")
    if abs(sec_sum - 1.0) > 0.01:
        print(f"Warning: Section weights sum to {sec_sum:.3f}, not 1.0")
    
    print(f"Current dimensions: {DIMENSIONS}")
    print(f"Current section weights: {SECTION_WEIGHTS}")

def debug_model_performance() -> Dict[str, Any]:
    """Debug function to test model connectivity and response quality."""
    print("🔍 Testing model connectivity and response quality...")
    
    test_results = {}
    test_text = "This is a test solution for a hackathon project that uses AI to solve climate change by creating a carbon tracking app."
    
    try:
        client = env_client()
        text_models = {mid: get_inference(client, mid) for mid in TEXT_MODEL_IDS}
        
        for mid, mi in text_models.items():
            print(f"\nTesting model: {mid}")
            test_results[mid] = {"status": "unknown", "error": None, "response_length": 0}
            
            try:
                # Test basic connectivity
                prompt = get_detailed_prompt("problem_statement", test_text, "default")
                resp = mi.generate_text(prompt=prompt, params=GEN_PARAMS, raw_response=True)
                
                if isinstance(resp, dict):
                    text = resp["results"][0]["generated_text"]
                else:
                    text = str(resp)
                
                test_results[mid]["response_length"] = len(text)
                test_results[mid]["raw_response"] = text[:500] + "..." if len(text) > 500 else text
                
                # Test JSON parsing
                s, e = text.find("{"), text.rfind("}")
                if s != -1 and e != -1:
                    json_text = text[s:e+1]
                    data = json.loads(json_text)
                    test_results[mid]["status"] = "success"
                    test_results[mid]["parsed_scores"] = data.get("scores", {})
                else:
                    test_results[mid]["status"] = "no_json_found"
                    test_results[mid]["error"] = "No JSON structure found in response"
                    
            except json.JSONDecodeError as je:
                test_results[mid]["status"] = "json_error"
                test_results[mid]["error"] = str(je)
            except Exception as e:
                test_results[mid]["status"] = "connection_error"
                test_results[mid]["error"] = str(e)
                
    except Exception as e:
        print(f"❌ Could not initialize models: {e}")
        return {"error": f"Model initialization failed: {e}"}
    
    return test_results

def get_available_themes() -> List[str]:
    """Get list of available theme configurations."""
    return list(THEME_CONFIGS.keys())

def create_custom_config(dimensions: Dict[str, float] = None, section_weights: Dict[str, float] = None) -> Dict:
    """Create a custom configuration dictionary."""
    config = {
        "dimensions": dimensions or THEME_CONFIGS["default"]["dimensions"].copy(),
        "section_weights": section_weights or THEME_CONFIGS["default"]["section_weights"].copy()
    }
    return config

def env_client() -> APIClient:
    load_dotenv()
    url = os.getenv("WATSONX_URL")
    apikey = os.getenv("WATSONX_API_KEY")
    if not url or not apikey:
        raise RuntimeError("Missing WATSONX_URL or WATSONX_API_KEY in .env file.")
    return APIClient(credentials={"url": url, "apikey": apikey})

def get_inference(client: APIClient, model_id: str) -> ModelInference:
    load_dotenv()
    project_id = os.getenv("WATSONX_PROJECT_ID")
    if not project_id:
        raise RuntimeError("Missing WATSONX_PROJECT_ID in .env file.")
    return ModelInference(model_id=model_id, params=GEN_PARAMS, api_client=client, project_id=project_id)

def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

def title_to_section(title: str) -> Optional[str]:
    t = (title or "").strip().lower()
    for sec, pats in TITLE_PATTERNS.items():
        for p in pats:
            if re.search(p, t):
                return sec
    return None

def collapse_slides(df: pd.DataFrame) -> Dict[str,str]:
    sec_map = {sec: "" for sec in TARGET_SECTIONS}
    title_col = find_col(df, COL_HINTS["title"]) or "title"
    content_col = find_col(df, COL_HINTS["content"]) or "content"
    for _, r in df.iterrows():
        sec = title_to_section(str(r.get(title_col, "")))
        text = str(r.get(content_col, "") or "").strip()
        if not text:
            continue
        if sec is None:
            continue
        if len(text) > len(sec_map[sec]):
            sec_map[sec] = text
    return sec_map

def load_table(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_parquet(path)  # needs pyarrow or fastparquet

# Image to base64 conversion (NOT USED - we only use extracted text)
# def to_b64_image(path: str) -> Optional[str]:
#     try:
#         with Image.open(path) as im:
#             im.verify()
#         with open(path, "rb") as f:
#             return base64.b64encode(f.read()).decode("utf-8")
#     except Exception:
#         return None

# ---- score normalization (fixes case/spacing mismatches) ----
CANON_MAP = {
    # New column names - exact matches
    "Problem Statement": "Problem Statement",
    "Proposed Solution": "Proposed Solution", 
    "Technical Architecture": "Technical Architecture",
    "Novelty  Uniqueness/creativity": "Novelty  Uniqueness/creativity",
    "presentaion": "presentaion",
    "ethical consideration": "ethical consideration",
    "completeness of design solution": "completeness of design solution",
    "implementing ibm dpk/rag/aws/watsonx/granite": "implementing ibm dpk/rag/aws/watsonx/granite",
    
    # Alternative name variations for robust matching
    "problem statement": "Problem Statement",
    "problem_statement": "Problem Statement",
    "proposed solution": "Proposed Solution",
    "proposed_solution": "Proposed Solution",
    "technical architecture": "Technical Architecture",
    "technical_architecture": "Technical Architecture",
    "novelty": "Novelty  Uniqueness/creativity",
    "uniqueness": "Novelty  Uniqueness/creativity",
    "creativity": "Novelty  Uniqueness/creativity",
    "novelty uniqueness creativity": "Novelty  Uniqueness/creativity",
    "presentation": "presentaion",
    "presentation quality": "presentaion",
    "ethics": "ethical consideration",
    "ethical": "ethical consideration",
    "ethical considerations": "ethical consideration",
    "completeness": "completeness of design solution",
    "design solution": "completeness of design solution",
    "implementation": "implementing ibm dpk/rag/aws/watsonx/granite",
    "ibm": "implementing ibm dpk/rag/aws/watsonx/granite",
    "dpk": "implementing ibm dpk/rag/aws/watsonx/granite",
    "rag": "implementing ibm dpk/rag/aws/watsonx/granite",
    "aws": "implementing ibm dpk/rag/aws/watsonx/granite",
    "watsonx": "implementing ibm dpk/rag/aws/watsonx/granite",
    "granite": "implementing ibm dpk/rag/aws/watsonx/granite",
    
    # Legacy mappings for backward compatibility
    "certification": "Problem Statement",
    "novelty_uniqueness": "Novelty  Uniqueness/creativity",
    "presentation_quality": "presentaion",
    "technical_architecture_quality": "Technical Architecture",
    "ethical_considerations": "ethical consideration",
    "impact_scalability": "completeness of design solution",
    "completeness_implementation": "implementing ibm dpk/rag/aws/watsonx/granite",
    "Completeness of the solution": "completeness of design solution",
    "impact on the theme chosen": "completeness of design solution",
}

def normalize_scores(raw: Dict[str, Any]) -> Dict[str, float]:
    out = {k: 0.0 for k in DIMENSIONS}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        k2 = CANON_MAP.get(k, None)
        if k2 is None:
            # try case-insensitive match
            lk = k.lower()
            k2 = next((ck for ck in DIMENSIONS if ck.lower() == lk), None)
        if k2:
            try:
                out[k2] = max(0.0, min(10.0, float(v)))
            except Exception:
                out[k2] = 0.0
    return out

# =========================
# Scoring
# =========================

@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8),
       retry=retry_if_exception_type((LLMReplyError, TimeoutError)))
def score_text(mi: ModelInference, section_name: str, section_text: str) -> Dict[str, Any]:
    prompt = get_detailed_prompt(section_name, section_text, CURRENT_THEME)
    
    try:
        resp = mi.generate_text(prompt=prompt, params=GEN_PARAMS, raw_response=True)
        text = resp["results"][0]["generated_text"] if isinstance(resp, dict) else str(resp)
        
        # More robust JSON extraction
        s, e = text.find("{"), text.rfind("}")
        if s == -1 or e == -1:
            print(f"⚠️  No JSON found in response for {section_name}. Raw response: {text[:200]}...")
            # Try alternative parsing - look for any {...} pattern
            import re
            json_matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
            if json_matches:
                text = json_matches[-1]  # Take the last/largest JSON match
            else:
                raise LLMReplyError(f"No valid JSON structure found in response")
        else:
            text = text[s:e+1]
        
        # Parse JSON with better error handling
        try:
            data = json.loads(text)
        except json.JSONDecodeError as je:
            print(f"⚠️  JSON decode error for {section_name}: {je}")
            print(f"⚠️  Attempted to parse: {text}")
            # Try to fix common JSON issues
            text_fixed = text.replace("'", '"').replace('True', 'true').replace('False', 'false').replace('None', 'null')
            try:
                data = json.loads(text_fixed)
                print(f"✅ JSON fixed and parsed successfully")
            except:
                raise LLMReplyError(f"Could not parse JSON response: {text[:200]}...")
        
        # Ensure required structure
        if "scores" not in data or not isinstance(data["scores"], dict) or not data["scores"]:
            # Create complete fallback scores for all dimensions
            data["scores"] = {k: 5.0 for k in DIMENSIONS}  # Default to middle score instead of 0
            print(f"⚠️  Missing or invalid 'scores' in response for {section_name}, using fallback scores")
        else:
            # Ensure all dimensions have scores, fill missing ones with 5.0
            for dim in DIMENSIONS:
                if dim not in data["scores"]:
                    data["scores"][dim] = 5.0
                    print(f"⚠️  Missing dimension '{dim}' in scores for {section_name}, using fallback 5.0")
            
        data["scores"] = normalize_scores(data.get("scores", {}))
        return data
        
    except Exception as e:
        print(f"⚠️  Complete failure in score_text for {section_name}: {str(e)}")
        raise LLMReplyError(f"Model evaluation failed: {str(e)}")

# Vision scoring function (NOT USED - we only use extracted text)
# @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8),
#        retry=retry_if_exception_type((LLMReplyError, TimeoutError)))
# def score_vision(mi: ModelInference, img_b64: str) -> Dict[str, Any]:
#     messages = [{
#         "role": "user",
#         "content": [
#             {"type": "text", "text": VISION_USER_QUERY},
#             {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
#         ]
#     }]
#     resp = mi.chat(messages=messages)
#     text = resp["choices"][0]["message"]["content"]
#     s, e = text.find("{"), text.rfind("}")
#     if s != -1 and e != -1:
#         text = text[s:e+1]
#     data = json.loads(text)
#     data["scores"] = normalize_scores(data.get("scores", {}))
#     data["section"] = "technical_architecture"
#     return data

def ensemble(scores_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    agg = {k: 0.0 for k in DIMENSIONS}
    feedback_list = []
    missing_reqs = set()
    n = max(1, len(scores_list))
    
    for d in scores_list:
        for k in DIMENSIONS:
            agg[k] += float(d["scores"].get(k, 0.0))
        
        # Collect feedback
        if "feedback" in d:
            feedback_list.append(d["feedback"])
        
        # Collect missing requirements
        if "missing_requirements" in d and d["missing_requirements"]:
            missing_reqs.update(d["missing_requirements"])
    
    for k in agg:
        agg[k] /= n
    
    # Combine feedback from all models
    combined_feedback = {"strengths": [], "improvements": []}
    for fb in feedback_list:
        if isinstance(fb, dict):
            if "strengths" in fb and fb["strengths"]:
                combined_feedback["strengths"].append(fb["strengths"])
            if "improvements" in fb and fb["improvements"]:
                combined_feedback["improvements"].append(fb["improvements"])
    
    return {
        "scores": agg, 
        "feedback": combined_feedback,
        "missing_requirements": list(missing_reqs)
    }

def weighted_total(scores: Dict[str, float]) -> float:
    """Calculate weighted total for a section using current dimension weights."""
    return sum(scores[k] * DIMENSIONS[k] for k in DIMENSIONS)

def calculate_overall_score(section_scores: Dict[str, float]) -> float:
    """Calculate overall score using section weights."""
    total = 0.0
    for section in TARGET_SECTIONS:
        section_key = f"{section}_total"
        if section_key in section_scores:
            total += section_scores[section_key] * SECTION_WEIGHTS[section]
    return total

# =========================
# Main
# =========================

def process_file(path: str, client: APIClient, text_models: Dict[str, ModelInference]) -> Dict[str, Any]:
    df = load_table(path)
    sec_vals = {sec: "" for sec in TARGET_SECTIONS}

    # Check file size for scoring optimization
    file_size_mb = 0.0
    original_file_path = None
    try:
        # Try to find the original PPT file
        submission_name = pathlib.Path(path).stem
        possible_extensions = ['.pptx', '.ppt']
        for ext in possible_extensions:
            for search_path in ['.', '../input_submissions', 'input_submissions']:
                potential_path = os.path.join(search_path, f"{submission_name}{ext}")
                if os.path.exists(potential_path):
                    original_file_path = potential_path
                    file_size_mb = os.path.getsize(potential_path) / (1024 * 1024)
                    break
            if original_file_path:
                break
    except Exception as e:
        print(f"Could not determine file size for {path}: {e}")
    
    # File size scoring factor (optimal size ≤ 5MB gets bonus, larger files get penalty)
    file_size_factor = 1.0
    file_size_feedback = ""
    
    if file_size_mb > 0:
        if file_size_mb <= 17.0:
            file_size_factor = 1.05  # 5% bonus for optimal size
            file_size_feedback = f"Optimal file size ({file_size_mb:.1f}MB) - processing efficiency bonus applied"
        elif file_size_mb <= 10.0:
            file_size_factor = 1.0  # No penalty for reasonable size
            file_size_feedback = f"Good file size ({file_size_mb:.1f}MB) - no processing impact"
        elif file_size_mb <= 20.0:
            file_size_factor = 0.98  # 2% penalty for large files
            file_size_feedback = f"Large file size ({file_size_mb:.1f}MB) - minor processing efficiency impact"
        else:
            file_size_factor = 0.95  # 5% penalty for very large files
            file_size_feedback = f"Very large file size ({file_size_mb:.1f}MB) - processing efficiency penalty applied"

    # Try direct section columns
    for sec in TARGET_SECTIONS:
        col = next((c for c in COL_HINTS[sec] if c in df.columns), None)
        if col:
            texts = df[col].astype(str).fillna("").tolist()
            sec_vals[sec] = max(texts, key=len) if texts else ""

    # If not found, collapse slides by title
    if not any(sec_vals.values()):
        sec_vals = collapse_slides(df)

    # Track missing sections
    missing_sections = []
    section_content_status = {}
    
    # Check content length and quality for each section
    for sec in TARGET_SECTIONS:
        content = sec_vals.get(sec, "").strip()
        section_content_status[sec] = {
            'has_content': len(content) > 20,  # Minimum 20 characters for meaningful content
            'content_length': len(content),
            'is_substantial': len(content) > 100  # 100+ characters for substantial content
        }
        
        # Consider a section missing if it has very little content
        if not section_content_status[sec]['has_content']:
            missing_sections.append(sec.replace('_', ' ').title())

    # Extract text from images using OCR (if available)
    img_text_col = "extracted_images"  # Column for extracted text from images using OCR
    extracted_text = ""
    
    # Check for extracted text from images
    if img_text_col in df.columns:
        # Filter out null/None values and combine all extracted text
        extracted_texts = df[img_text_col].dropna().astype(str).tolist()
        # Remove empty strings and 'null' strings
        extracted_texts = [text.strip() for text in extracted_texts if text.strip() and text.strip().lower() not in ['null', 'none', 'nan', '']]
        
        if extracted_texts:
            # Check if these are file paths or actual OCR text
            first_item = extracted_texts[0]
            if first_item.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')) or 'pipeline_output' in first_item:
                # These are file paths, not OCR text - we need to read the images
                print(f"📁 Found {len(extracted_texts)} image file paths, attempting OCR...")
                ocr_texts = []
                
                for img_path in extracted_texts:
                    if os.path.exists(img_path):
                        try:
                            # Try to extract text from image using OCR (if available)
                            # For now, we'll just note that images exist but no OCR is available
                            print(f"  📷 Image found: {img_path} (OCR not implemented)")
                        except Exception as e:
                            print(f"  ⚠️ Could not process image {img_path}: {e}")
                    else:
                        print(f"  ❌ Image file not found: {img_path}")
                
                # For now, we'll indicate that images are present but no text was extracted
                if extracted_texts:
                    extracted_text = f"[{len(extracted_texts)} technical diagrams/images detected but OCR text extraction not available]"
                    print(f"✅ Found {len(extracted_texts)} images but no OCR text extraction")
            else:
                # These appear to be actual OCR text results
                extracted_text = " ".join(extracted_texts)
                if extracted_text:
                    print(f"✅ Found extracted image text: {len(extracted_text)} characters")
        else:
            print(f"📭 No images or extracted text found in {img_text_col} column")

    result_row: Dict[str, Any] = {"submission_id": pathlib.Path(path).stem}

    # ---------- Text sections ----------
    section_feedback = {}
    for sec in ["problem_statement", "proposed_solution"]:
        per_model = []
        
        # Check if this section has sufficient content
        content = sec_vals.get(sec, "").strip()
        if len(content) < 20:  # Very little content
            # Apply penalty for missing/insufficient content but not zero scores
            for mid, mi in text_models.items():
                # Create moderate penalty scores for missing sections (not complete zero)
                penalty_scores = {}
                for dim in DIMENSIONS:
                    penalty_scores[dim] = 3.0  # Reasonable penalty for missing content (3/10)
                
                data = {
                    "scores": penalty_scores,
                    "feedback": {
                        "strengths": "",
                        "improvements": f"Section appears to be missing or has insufficient content. Please provide detailed {sec.replace('_', ' ')} information."
                    },
                    "missing_requirements": [f"Detailed {sec.replace('_', ' ')} content"]
                }
                per_model.append(data)

                # per-model log
                row_log = {
                    "submission_id": result_row["submission_id"],
                    "section": sec,
                    "evaluator_model": mid,
                    "evaluator_type": "text",
                    **{k: data["scores"].get(k, 0.0) for k in DIMENSIONS},
                }
                row_log["section_total"] = weighted_total({k: row_log[k] for k in DIMENSIONS})
                MODEL_SCORE_ROWS.append(row_log)
        else:
            # Normal scoring for sections with content
            for mid, mi in text_models.items():
                try:
                    data = score_text(mi, sec, content)
                    print(f"✅ Successfully scored {sec} with model {mid}")
                except Exception as e:
                    print(f"⚠️  ERROR in score_text for model {mid}, section {sec}: {str(e)}")
                    # Create complete fallback scores for all dimensions
                    fallback_scores = {}
                    for dim in DIMENSIONS:
                        fallback_scores[dim] = 5.0  # Middle score for technical errors
                    
                    data = {
                        "scores": fallback_scores,
                        "feedback": {
                            "strengths": f"Content is present for {sec.replace('_', ' ')} evaluation",
                            "improvements": f"Evaluation could not be completed due to technical issue"
                        },
                        "notes": f"Technical evaluation failure - using default scores",
                        "missing_requirements": []
                    }
                per_model.append(data)

                # per-model log
                row_log = {
                    "submission_id": result_row["submission_id"],
                    "section": sec,
                    "evaluator_model": mid,
                    "evaluator_type": "text",
                    **{k: data["scores"].get(k, 0.0) for k in DIMENSIONS},
                }
                row_log["section_total"] = weighted_total({k: row_log[k] for k in DIMENSIONS})
                MODEL_SCORE_ROWS.append(row_log)

        result = ensemble(per_model)
        avg = result["scores"]
        section_feedback[sec] = result["feedback"]
        
        for k in DIMENSIONS:
            result_row[f"{sec}_{k}"] = avg[k]
        result_row[f"{sec}_total"] = weighted_total(avg)

    # ---------- Technical architecture ----------
    per_model = []
    
    # Check technical architecture content
    tech_arch_content = sec_vals.get("technical_architecture", "").strip()
    has_tech_arch_content = len(tech_arch_content) > 20
    
    # Check if we have extracted images (either as text or as detected image files)
    has_extracted_images = bool(extracted_text.strip())
    has_image_files = any(df['images_present']) if 'images_present' in df.columns else False
    
    # Technical architecture is considered missing if there's no substantial text AND no images
    if not has_tech_arch_content and not has_extracted_images and not has_image_files:
        if "Technical Architecture" not in missing_sections:
            missing_sections.append("Technical Architecture")
        
        # Apply penalty for missing technical architecture
        for mid, mi in text_models.items():
            penalty_scores = {}
            for dim in DIMENSIONS:
                penalty_scores[dim] = 3.0  # Reasonable penalty for missing content (3/10)
            
            data = {
                "scores": penalty_scores,
                "feedback": {
                    "strengths": "",
                    "improvements": "Technical architecture section is missing. Please provide system architecture diagrams, technology stack details, and implementation approach."
                },
                "missing_requirements": ["Technical architecture diagrams", "System design details", "Technology stack specification"]
            }
            per_model.append(data)

            row_log = {
                "submission_id": result_row["submission_id"],
                "section": "technical_architecture",
                "evaluator_model": mid,
                "evaluator_type": "text",
                **{k: data["scores"].get(k, 0.0) for k in DIMENSIONS},
                "used_image": False,
                "image_path": None,
                "used_extracted_text": False,
            }
            row_log["section_total"] = weighted_total({k: row_log[k] for k in DIMENSIONS})
            MODEL_SCORE_ROWS.append(row_log)
    
    elif extracted_text:
        # Use text models with extracted image text
        for mid, mi in text_models.items():
            try:
                # Combine regular text content with extracted image text
                combined_text = f"{tech_arch_content}\n\nExtracted text from architecture diagrams:\n{extracted_text}"
                data = score_text(mi, "technical_architecture", combined_text)
                print(f"✅ Successfully scored technical_architecture with model {mid} using extracted text")
            except Exception as e:
                print(f"⚠️  ERROR in technical_architecture scoring for model {mid}: {str(e)}")
                # Create complete fallback scores for all dimensions
                fallback_scores = {}
                for dim in DIMENSIONS:
                    fallback_scores[dim] = 5.0  # Middle score for technical errors
                
                data = {
                    "scores": fallback_scores,
                    "feedback": {
                        "strengths": "Has technical content and architecture diagrams", 
                        "improvements": "Evaluation could not be completed due to technical issue"
                    },
                    "notes": f"Technical evaluation failure - using default scores",
                    "missing_requirements": []
                }
            per_model.append(data)

            row_log = {
                "submission_id": result_row["submission_id"],
                "section": "technical_architecture",
                "evaluator_model": mid,
                "evaluator_type": "text",
                **{k: data["scores"].get(k, 0.0) for k in DIMENSIONS},
                "used_image": False,
                "image_path": None,
                "used_extracted_text": True,
            }
            row_log["section_total"] = weighted_total({k: row_log[k] for k in DIMENSIONS})
            MODEL_SCORE_ROWS.append(row_log)
    else:
        # Only use regular text content
        for mid, mi in text_models.items():
            try:
                data = score_text(mi, "technical_architecture", tech_arch_content)
                print(f"✅ Successfully scored technical_architecture with model {mid}")
            except Exception as e:
                print(f"⚠️  ERROR in technical_architecture scoring for model {mid}: {str(e)}")
                # Create complete fallback scores for all dimensions
                fallback_scores = {}
                for dim in DIMENSIONS:
                    fallback_scores[dim] = 5.0  # Middle score for technical errors
                
                data = {
                    "scores": fallback_scores,
                    "feedback": {
                        "strengths": "Has technical architecture content", 
                        "improvements": "Evaluation could not be completed due to technical issue"
                    },
                    "notes": f"Technical evaluation failure - using default scores",
                    "missing_requirements": []
                }
            per_model.append(data)

            row_log = {
                "submission_id": result_row["submission_id"],
                "section": "technical_architecture",
                "evaluator_model": mid,
                "evaluator_type": "text",
                **{k: data["scores"].get(k, 0.0) for k in DIMENSIONS},
                "used_image": False,
                "image_path": None,
                "used_extracted_text": False,
            }
            row_log["section_total"] = weighted_total({k: row_log[k] for k in DIMENSIONS})
            MODEL_SCORE_ROWS.append(row_log)

    result = ensemble(per_model)
    avg = result["scores"]
    section_feedback["technical_architecture"] = result["feedback"]
    
    for k in DIMENSIONS:
        result_row[f"technical_architecture_{k}"] = avg[k]
    result_row["technical_architecture_total"] = weighted_total(avg)

    # ---------- Overall ----------
    for sec in TARGET_SECTIONS:
        result_row.setdefault(f"{sec}_total", 0.0)
    
    # Calculate base overall score using section weights
    base_overall_score = calculate_overall_score(result_row)
    
    # Apply file size optimization factor
    result_row["overall_score"] = base_overall_score * file_size_factor
    result_row["file_size_mb"] = file_size_mb
    result_row["file_size_factor"] = file_size_factor
    
    # Add missing sections information
    result_row["missing_sections"] = ", ".join(missing_sections) if missing_sections else "None"
    result_row["missing_sections_count"] = len(missing_sections)
    
    # Add content status for debugging
    result_row["content_status"] = json.dumps(section_content_status)
    
    # ---------- Compile Feedback ----------
    overall_feedback = {"strengths": set(), "improvements": set()}  # Use sets to avoid duplicates
    missing_content = set()  # Separate missing content from improvements
    all_missing_requirements = set()
    
    for sec in TARGET_SECTIONS:
        if sec in section_feedback:
            sec_fb = section_feedback[sec]
            if isinstance(sec_fb, dict):
                # Add section-specific feedback without repetitive prefixes
                if sec_fb.get("strengths"):
                    for strength in sec_fb["strengths"]:
                        strength = strength.strip()
                        if strength and strength not in overall_feedback["strengths"]:
                            overall_feedback["strengths"].add(strength)
                
                if sec_fb.get("improvements"):
                    for improvement in sec_fb["improvements"]:
                        improvement = improvement.strip()
                        # Only add actual improvements, not missing content
                        if (improvement and improvement not in overall_feedback["improvements"] 
                            and not improvement.lower().startswith(('missing', 'should include', 'needs to include', 'lacks', 'section appears to be', 'please provide'))):
                            overall_feedback["improvements"].add(improvement)
                
                # Collect missing requirements separately
                if sec_fb.get("missing_requirements"):
                    all_missing_requirements.update(sec_fb["missing_requirements"])
    
    # Add missing sections as missing content, not improvements
    if missing_sections:
        missing_content.add(f"Missing required sections: {', '.join(missing_sections)}")
    
    # Check for missing diagrams/images as missing content
    if not extracted_text and not has_image_files and "Technical Architecture" not in missing_sections:
        missing_content.add("Missing visual diagrams or architectural drawings")
    
    # Add missing requirements as missing content
    if all_missing_requirements:
        missing_content.add("Missing requirements: " + ", ".join(all_missing_requirements))
    
    # Compile final feedback text with proper limits
    strengths_list = list(overall_feedback["strengths"])[:2]  # Max 2 strengths
    improvements_list = list(overall_feedback["improvements"])[:3]  # Max 3 improvements
    missing_list = list(missing_content)[:3]  # Max 3 missing items
    
    feedback_text = ""
    if strengths_list:
        feedback_text += "STRENGTHS:\n" + "\n".join(f"• {s}" for s in strengths_list)
    
    if improvements_list:
        if feedback_text:
            feedback_text += "\n\n"
        feedback_text += "AREAS FOR IMPROVEMENT:\n" + "\n".join(f"• {i}" for i in improvements_list)
    
    if missing_list:
        if feedback_text:
            feedback_text += "\n\n"
        feedback_text += "MISSING CONTENT:\n" + "\n".join(f"• {m}" for m in missing_list)
    
    # Add file size feedback
    if file_size_feedback:
        if feedback_text:
            feedback_text += "\n\n"
        feedback_text += f"FILE SIZE OPTIMIZATION:\n• {file_size_feedback}"
    
    if not feedback_text:
        feedback_text = "No specific feedback available. Please ensure all sections contain sufficient content for evaluation."
    
    result_row["feedback"] = feedback_text
    result_row["missing_requirements"] = list(all_missing_requirements)
    result_row["track"] = CURRENT_THEME

    # ========== ADD INDIVIDUAL CRITERIA SCORES AS MAIN COLUMNS ==========
    # Calculate average scores for each criterion across all sections
    for criterion in DIMENSIONS:
        section_scores = []
        for sec in TARGET_SECTIONS:
            section_criterion_key = f"{sec}_{criterion}"
            if section_criterion_key in result_row:
                section_scores.append(result_row[section_criterion_key])
        
        # Calculate weighted average based on section weights if scores available
        if section_scores:
            weighted_avg = 0.0
            total_weight = 0.0
            for i, sec in enumerate(TARGET_SECTIONS):
                if i < len(section_scores):
                    weighted_avg += section_scores[i] * SECTION_WEIGHTS[sec]
                    total_weight += SECTION_WEIGHTS[sec]
            
            if total_weight > 0:
                result_row[criterion] = weighted_avg / total_weight
            else:
                result_row[criterion] = sum(section_scores) / len(section_scores)
        else:
            result_row[criterion] = 0.0
    
    # Add additional columns you requested
    result_row["Certification(yes/no)"] = "no"  # Default to no, can be updated based on content analysis
    result_row["Feedback"] = feedback_text
    result_row["Missing content(anything missing)"] = ", ".join(all_missing_requirements) if all_missing_requirements else "None"
    
    # Check for certification mentions in content
    all_content = " ".join([sec_vals.get(sec, "") for sec in TARGET_SECTIONS]).lower()
    if any(cert_word in all_content for cert_word in ["certificate", "certification", "certified", "credential"]):
        result_row["Certification(yes/no)"] = "yes"

    return result_row

def main():
    ap = argparse.ArgumentParser(description="Evaluate PPTs (3 sections) with watsonx.ai, supporting images for architecture.")
    ap.add_argument("--input_dir", required=True, help="Directory with .parquet or .csv files (one submission per file).")
    ap.add_argument("--out_prefix", default="tri_scores", help="Output file prefix.")
    ap.add_argument("--theme", choices=get_available_themes(), default="default", 
                    help=f"Theme configuration to use. Available: {', '.join(get_available_themes())}")
    ap.add_argument("--config_file", help="Path to custom JSON configuration file")
    ap.add_argument("--list_themes", action="store_true", help="List available themes and exit")
    
    # Custom dimension weights
    ap.add_argument("--uniqueness_weight", type=float, help="Weight for uniqueness dimension")
    ap.add_argument("--completeness_weight", type=float, help="Weight for completeness dimension")
    ap.add_argument("--impact_weight", type=float, help="Weight for impact on theme dimension")
    ap.add_argument("--ethics_weight", type=float, help="Weight for ethical consideration dimension")
    
    # Custom section weights
    ap.add_argument("--problem_weight", type=float, help="Weight for problem statement section")
    ap.add_argument("--solution_weight", type=float, help="Weight for proposed solution section")
    ap.add_argument("--architecture_weight", type=float, help="Weight for technical architecture section")
    
    args = ap.parse_args()
    
    # List themes and exit if requested
    if args.list_themes:
        print("Available theme configurations:")
        print("=" * 60)
        for theme_name, config in THEME_CONFIGS.items():
            print(f"\n{config['name'].upper()} ({theme_name})")
            print(f"   Dimension Weights: {config['dimensions']}")
            print(f"   Section Weights: {config['section_weights']}")
            
            if 'detailed_criteria' in config:
                print(f"   Special Focus Areas:")
                for criteria_type, criteria_list in config['detailed_criteria'].items():
                    if criteria_type not in ['completeness', 'impact', 'uniqueness', 'ethics', 'presentation']:
                        print(f"   - {criteria_type.replace('_', ' ').title()}")
                        for criterion in criteria_list[:3]:  # Show first 3
                            print(f"     • {criterion}")
        
        print("\n" + "=" * 60)
        print("MANDATORY REQUIREMENTS FOR ALL TRACKS:")
        print("• YouTube video (Private with 'anybody with the link' access)")
        print("• Must mention IBM or AWS or both")
        print("• Working prototype demonstration")
        print("• DPK/IBM Granite usage (RAG/Agentic)")
        print("• Bias testing documentation")
        print("• Must-have technologies clearly identified")
        print("\nFor detailed criteria, see EVALUATION_CRITERIA.md")
        return
    
    # Load configuration
    custom_config = None
    
    # Check for custom configuration file
    if args.config_file:
        try:
            with open(args.config_file, 'r') as f:
                custom_config = json.load(f)
            print(f"Loaded custom configuration from {args.config_file}")
        except Exception as e:
            print(f"Error loading config file: {e}")
            return
    
    # Check for command line weight overrides
    elif any([args.uniqueness_weight, args.completeness_weight, args.impact_weight, args.ethics_weight,
              args.problem_weight, args.solution_weight, args.architecture_weight]):
        
        # Create custom config from command line arguments
        base_config = THEME_CONFIGS[args.theme]
        
        dimensions = base_config["dimensions"].copy()
        section_weights = base_config["section_weights"].copy()
        
        # Update dimension weights if provided
        if args.uniqueness_weight is not None:
            dimensions["uniqueness"] = args.uniqueness_weight
        if args.completeness_weight is not None:
            dimensions["Completeness of the solution"] = args.completeness_weight
        if args.impact_weight is not None:
            dimensions["impact on the theme chosen"] = args.impact_weight
        if args.ethics_weight is not None:
            dimensions["ethical consideration"] = args.ethics_weight
        
        # Update section weights if provided
        if args.problem_weight is not None:
            section_weights["problem_statement"] = args.problem_weight
        if args.solution_weight is not None:
            section_weights["proposed_solution"] = args.solution_weight
        if args.architecture_weight is not None:
            section_weights["technical_architecture"] = args.architecture_weight
        
        custom_config = create_custom_config(dimensions, section_weights)
        print("Using command line weight overrides")
    
    # Load the configuration
    if custom_config:
        load_theme_config(custom_config=custom_config)
    else:
        load_theme_config(args.theme)

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.parquet")) + glob.glob(os.path.join(args.input_dir, "*.csv")))
    if not files:
        raise SystemExit(f"No .parquet or .csv in {args.input_dir}")

    client = env_client()
    text_models = {mid: get_inference(client, mid) for mid in TEXT_MODEL_IDS}

    rows = []
    for f in files:
        row = process_file(f, client, text_models)
        if row:
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No rows produced. Check inputs and logs.")

    if "overall_score" not in out.columns:
        print("DEBUG columns:", list(out.columns))
        print("DEBUG first row:", out.iloc[0].to_dict() if len(out) else {})
        raise RuntimeError("overall_score missing; ensure process_file computes totals and returns the row.")

    out.sort_values("overall_score", ascending=False, inplace=True)
    
    # Add configuration info to the output
    config_info = {
        "theme": CURRENT_THEME,
        "dimensions": DIMENSIONS,
        "section_weights": SECTION_WEIGHTS
    }
    
    # Save configuration alongside results
    config_filename = f"{args.out_prefix}_config.json"
    with open(config_filename, 'w') as f:
        json.dump(config_info, f, indent=2)
    
    # Save all final tables as CSV files (primary format)
    out.to_csv(f"{args.out_prefix}.csv", index=False)
    out.to_parquet(f"{args.out_prefix}.parquet", index=False)

    model_df = pd.DataFrame(MODEL_SCORE_ROWS)
    model_prefix = f"{args.out_prefix}_per_model"
    model_df.to_csv(f"{model_prefix}.csv", index=False)
    model_df.to_parquet(f"{model_prefix}.parquet", index=False)
    
    # Save top results as separate CSV files
    top_20 = out.head(20)
    top_20.to_csv(f"{args.out_prefix}_top20.csv", index=False)

    print(out.head(10))
    print(f"Saved main results: {args.out_prefix}.csv, {args.out_prefix}.parquet")
    print(f"Saved per-model results: {model_prefix}.csv, {model_prefix}.parquet")
    print(f"Saved top 20 results: {args.out_prefix}_top20.csv")
    print(f"Saved top 20 results: {args.out_prefix}_top20.csv")
    print(f"Saved configuration: {config_filename}")
    print(f"\nResults Summary:")
    print(f"  Total submissions: {len(out)}")
    print(f"  Top 20 saved for detailed review")
    print(f"\nUsed configuration:")
    print(f"  Theme: {CURRENT_THEME}")
    print(f"  Dimensions: {DIMENSIONS}")
    print(f"  Section weights: {SECTION_WEIGHTS}")

if __name__ == "__main__":
    main()
