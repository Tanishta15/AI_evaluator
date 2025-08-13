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

# =========================
# CONFIG (edit as needed)
# =========================

MODEL_SCORE_ROWS: List[Dict[str, Any]] = []

TARGET_SECTIONS = ["problem_statement", "proposed_solution", "technical_architecture"]

# Canonical scoring dimensions (use these exact labels everywhere)
DIMENSIONS = {
    "uniqueness": 0.25,
    "Completeness of the solution": 0.30,
    "impact on the theme chosen": 0.30,
    "ethical consideration": 0.05,
}

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
    "ibm/granite-3-3-8b-instruct",  # Replaced mistral with reliable granite model
]

# Vision LLMs for images (send image + question)
VISION_MODEL_IDS = [
    "meta-llama/llama-3-2-90b-vision-instruct",
    "ibm/granite-vision-3-2-2b",
]

GEN_PARAMS = {
    GenParams.MAX_NEW_TOKENS: 256,
    GenParams.TEMPERATURE: 0.2,
    GenParams.TOP_P: 0.9,
    GenParams.REPETITION_PENALTY: 1.1,
}

TEXT_PROMPT = """You are a strict hackathon evaluator. Rate the SECTION_CONTENT for "{section_name}".
Scores must be 0-10 (decimals allowed) for:
- uniqueness
- Completeness of the solution
- impact on the theme chosen
- ethical consideration

Rules:
- Judge ONLY what is provided. If missing/irrelevant => 0.
- Respond as pure JSON, exactly:
{{
  "section": "{section_name}",
  "scores": {{
    "uniqueness": <float>,
    "Completeness of the solution": <float>,
    "impact on the theme chosen": <float>,
    "ethical consideration": <float>
  }},
  "notes": "<1-2 line justification>"
}}

SECTION_CONTENT:
\"\"\"{section_text}\"\"\""""

VISION_USER_QUERY = (
    "Evaluate this technical architecture image for a hackathon PPT. "
    "Rate (0-10) the same criteria: uniqueness, Completeness of the solution, impact on the theme chosen, ethical consideration. "
    "Return JSON ONLY with the exact schema used before. Be concise in 'notes'."
)

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

# =========================
# Helpers
# =========================

class LLMReplyError(Exception): pass

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

def to_b64_image(path: str) -> Optional[str]:
    try:
        with Image.open(path) as im:
            im.verify()
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None

# ---- score normalization (fixes case/spacing mismatches) ----
CANON_MAP = {
    "uniqueness": "uniqueness",
    "completeness of the solution": "Completeness of the solution",
    "Completeness of the solution": "Completeness of the solution",
    "impact on the theme chosen": "impact on the theme chosen",
    "ethical consideration": "ethical consideration",
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
    prompt = TEXT_PROMPT.format(section_name=section_name, section_text=section_text)
    resp = mi.generate_text(prompt=prompt, params=GEN_PARAMS, raw_response=True)
    text = resp["results"][0]["generated_text"] if isinstance(resp, dict) else str(resp)
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1:
        text = text[s:e+1]
    data = json.loads(text)
    data["scores"] = normalize_scores(data.get("scores", {}))
    return data

@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8),
       retry=retry_if_exception_type((LLMReplyError, TimeoutError)))
def score_vision(mi: ModelInference, img_b64: str) -> Dict[str, Any]:
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": VISION_USER_QUERY},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
        ]
    }]
    resp = mi.chat(messages=messages)
    text = resp["choices"][0]["message"]["content"]
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1:
        text = text[s:e+1]
    data = json.loads(text)
    data["scores"] = normalize_scores(data.get("scores", {}))
    data["section"] = "technical_architecture"
    return data

def ensemble(scores_list: List[Dict[str, Any]]) -> Dict[str, float]:
    agg = {k: 0.0 for k in DIMENSIONS}
    n = max(1, len(scores_list))
    for d in scores_list:
        for k in DIMENSIONS:
            agg[k] += float(d["scores"].get(k, 0.0))
    for k in agg:
        agg[k] /= n
    return agg

def weighted_total(scores: Dict[str, float]) -> float:
    return sum(scores[k] * DIMENSIONS[k] for k in DIMENSIONS)

# =========================
# Main
# =========================

def process_file(path: str, client: APIClient, text_models: Dict[str, ModelInference], vision_models: Dict[str, ModelInference]) -> Dict[str, Any]:
    df = load_table(path)
    sec_vals = {sec: "" for sec in TARGET_SECTIONS}

    # Try direct section columns
    for sec in TARGET_SECTIONS:
        col = next((c for c in COL_HINTS[sec] if c in df.columns), None)
        if col:
            texts = df[col].astype(str).fillna("").tolist()
            sec_vals[sec] = max(texts, key=len) if texts else ""

    # If not found, collapse slides by title
    if not any(sec_vals.values()):
        sec_vals = collapse_slides(df)

    # Image detection and extracted text
    img_flag_col = find_col(df, COL_HINTS["images_present"])
    img_path_col = find_col(df, COL_HINTS["tech_arch_image_path"])
    img_text_col = "image_extracted"  # Column for extracted text from images using OCR

    has_image, img_b64, chosen_img_path, extracted_text = False, None, None, ""
    # Check for extracted text from images
    if img_text_col in df.columns:
        extracted_text = " ".join(df[img_text_col].dropna().astype(str).tolist())
    
    if img_flag_col and df[img_flag_col].astype(str).str.lower().isin(["true", "1", "yes"]).any():
        if img_path_col:
            for p in df[img_path_col].dropna().astype(str):
                if os.path.isfile(p):
                    chosen_img_path = p
                    img_b64 = to_b64_image(p)
                    if img_b64:
                        has_image = True
                        break

    result_row: Dict[str, Any] = {"submission_id": pathlib.Path(path).stem}

    # ---------- Text sections ----------
    for sec in ["problem_statement", "proposed_solution"]:
        per_model = []
        for mid, mi in text_models.items():
            try:
                data = score_text(mi, sec, sec_vals.get(sec, ""))
            except Exception:
                data = {"scores": {k: 0.0 for k in DIMENSIONS}}
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

        avg = ensemble(per_model)
        for k in DIMENSIONS:
            result_row[f"{sec}_{k}"] = avg[k]
        result_row[f"{sec}_total"] = weighted_total(avg)

    # ---------- Technical architecture ----------
    per_model = []
    if has_image and img_b64:
        for mid, mi in vision_models.items():
            try:
                data = score_vision(mi, img_b64)
            except Exception:
                data = {"scores": {k: 0.0 for k in DIMENSIONS}}
            per_model.append(data)

            row_log = {
                "submission_id": result_row["submission_id"],
                "section": "technical_architecture",
                "evaluator_model": mid,
                "evaluator_type": "vision",
                **{k: data["scores"].get(k, 0.0) for k in DIMENSIONS},
                "used_image": True,
                "image_path": chosen_img_path,
            }
            row_log["section_total"] = weighted_total({k: row_log[k] for k in DIMENSIONS})
            MODEL_SCORE_ROWS.append(row_log)
    else:
        for mid, mi in text_models.items():
            try:
                # Combine regular text content with extracted image text if available
                tech_arch_text = sec_vals.get("technical_architecture", "")
                if extracted_text:
                    tech_arch_text = f"{tech_arch_text}\n\nExtracted text from architecture diagrams:\n{extracted_text}"
                data = score_text(mi, "technical_architecture", tech_arch_text)
            except Exception:
                data = {"scores": {k: 0.0 for k in DIMENSIONS}}
            per_model.append(data)

            row_log = {
                "submission_id": result_row["submission_id"],
                "section": "technical_architecture",
                "evaluator_model": mid,
                "evaluator_type": "text",
                **{k: data["scores"].get(k, 0.0) for k in DIMENSIONS},
                "used_image": False,
                "image_path": None,
                "used_extracted_text": bool(extracted_text),
            }
            row_log["section_total"] = weighted_total({k: row_log[k] for k in DIMENSIONS})
            MODEL_SCORE_ROWS.append(row_log)

    avg = ensemble(per_model)
    for k in DIMENSIONS:
        result_row[f"technical_architecture_{k}"] = avg[k]
    result_row["technical_architecture_total"] = weighted_total(avg)

    # ---------- Overall ----------
    for sec in TARGET_SECTIONS:
        result_row.setdefault(f"{sec}_total", 0.0)
    totals = [result_row[f"{s}_total"] for s in TARGET_SECTIONS]
    result_row["overall_score"] = sum(totals) / max(1, len(totals))

    return result_row

def main():
    ap = argparse.ArgumentParser(description="Evaluate PPTs (3 sections) with watsonx.ai, supporting images for architecture.")
    ap.add_argument("--input_dir", required=True, help="Directory with .parquet or .csv files (one submission per file).")
    ap.add_argument("--out_prefix", default="tri_scores", help="Output file prefix.")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.parquet")) + glob.glob(os.path.join(args.input_dir, "*.csv")))
    if not files:
        raise SystemExit(f"No .parquet or .csv in {args.input_dir}")

    client = env_client()
    text_models = {mid: get_inference(client, mid) for mid in TEXT_MODEL_IDS}
    vision_models = {mid: get_inference(client, mid) for mid in VISION_MODEL_IDS}

    rows = []
    for f in files:
        row = process_file(f, client, text_models, vision_models)
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
    out.to_csv(f"{args.out_prefix}.csv", index=False)
    out.to_parquet(f"{args.out_prefix}.parquet", index=False)

    model_df = pd.DataFrame(MODEL_SCORE_ROWS)
    model_prefix = f"{args.out_prefix}_per_model"
    model_df.to_csv(f"{model_prefix}.csv", index=False)
    model_df.to_parquet(f"{model_prefix}.parquet", index=False)

    print(out.head(10))
    print(f"Saved: {args.out_prefix}.csv, {args.out_prefix}.parquet")
    print(f"Saved per-model: {model_prefix}.csv, {model_prefix}.parquet")

if __name__ == "__main__":
    main()
