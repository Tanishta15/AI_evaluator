import os, re, json, argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

# ---------- utils ----------
def load_table(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_parquet(path)  # needs pyarrow or fastparquet

def strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]
    return out

def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def canonical(s: str) -> str:
    s = normalize_space(s).lower()
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def resolve_path(candidate: str, base_file: str, base_dir: Optional[str] = None) -> str:
    """Resolve relative image paths using (1) absolute, (2) next to parquet, (3) --base-dir."""
    if not candidate:
        return ""
    p = Path(candidate)
    if p.is_absolute() and p.exists():
        return str(p)
    rel1 = Path(base_file).parent / candidate
    if rel1.exists():
        return str(rel1)
    if base_dir:
        rel2 = Path(base_dir) / candidate
        if rel2.exists():
            return str(rel2)
    return str(p)  # last fallback (may not exist)

# ---------- OCR ----------
def ensure_tesseract() -> bool:
    try:
        import pytesseract  # noqa
        return True
    except Exception:
        print("[ERROR] pytesseract not installed. pip install pytesseract (and install the Tesseract binary).")
        return False

def _preprocess(img: Image.Image) -> Image.Image:
    g = ImageOps.grayscale(img)
    g = ImageEnhance.Contrast(g).enhance(1.6)
    g = g.filter(ImageFilter.SHARPEN)
    g = g.point(lambda x: 0 if x < 175 else 255, mode="1")  # light binarization
    return g

def ocr_image(path: str, lang: str = "eng") -> str:
    import pytesseract
    with Image.open(path) as im:
        proc = _preprocess(im)
        txt = pytesseract.image_to_string(proc, lang=lang)
    txt = (txt or "").strip()
    txt = re.sub(r"[^\S\r\n]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt

# ---------- parse the certificate text ----------
NAME_PATTERNS = [
    r"presented to\s*\n\s*(?P<name>[A-Za-z][A-Za-z .'-]{2,})",
]
CODE_PATTERNS = [
    r"\b(?:URL|Code)\s*[-: ]\s*(?P<code>[A-Z0-9]{6,})\b",
    r"\((?:URL|Code)\s*[-: ]\s*(?P<code2>[A-Z0-9]{6,})\)",
]

def extract_fields(ocr_text: str) -> Dict[str, str]:
    text = ocr_text or ""
    name, code, program = "", "", ""

    # name
    for pat in NAME_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            name = normalize_space(m.groupdict().get("name", ""))
            break

    # code
    for pat in CODE_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            gd = m.groupdict()
            code = normalize_space(gd.get("code") or gd.get("code2") or "")
            break

    # program: grab the text after "for the completion of"
    m = re.search(r"for the completion of\s*(?P<p>.+)", text, flags=re.IGNORECASE)
    if m:
        start = m.start("p")
        window = text[start:start + 600]
        # prune at obvious anchors
        for stop in ["\nCompletion", "(URL", "Learning hours", "\nURL", "\nCode"]:
            window = window.split(stop, 1)[0]
        program = normalize_space(window)

    return {"name": name, "program": program, "code": code}

# ---------- match against datasheet ----------
def similarity(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    return SequenceMatcher(None, canonical(a), canonical(b)).ratio()

def match_record(extracted: Dict[str, str], ds: pd.DataFrame) -> Tuple[bool, float, Dict[str, Any]]:
    ds = strip_cols(ds)
    cmap = {c.lower(): c for c in ds.columns}
    name_col = cmap.get("name of the participant")
    prog_col = cmap.get("name of the program")
    code_col = cmap.get("code of the program")
    if not (name_col and prog_col and code_col):
        raise ValueError("data sheet must have columns: 'Name of the Program', ' Code of the Program', ' Name of the Participant'")

    # exact
    exact = ds[
        (ds[name_col].map(canonical) == canonical(extracted.get("name",""))) &
        (ds[prog_col].map(canonical) == canonical(extracted.get("program",""))) &
        (ds[code_col].map(canonical) == canonical(extracted.get("code","")))
    ]
    if not exact.empty:
        return True, 1.0, exact.iloc[0].to_dict()

    # fuzzy average of three
    best_row, best_score = {}, -1.0
    for _, r in ds.iterrows():
        s1 = similarity(extracted.get("name",""), str(r[name_col]))
        s2 = similarity(extracted.get("program",""), str(r[prog_col]))
        s3 = similarity(extracted.get("code",""), str(r[code_col]))
        score = (s1 + s2 + s3) / 3.0
        if score > best_score:
            best_score, best_row = score, r.to_dict()
    return (best_score >= 0.82), best_score, best_row

# ---------- discover certificate images from parquet ----------
def find_certificate_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame()
    if "title" in df.columns:
        rows = df[df["title"].astype(str).str.lower().str.contains("certificate", na=False)]
    if rows.empty and "content" in df.columns:
        rows = df[df["content"].astype(str).str.lower().str.contains("certificate", na=False)]
    return rows

def parse_images_info(val: Any) -> List[str]:
    """images_info is JSON like {"count":1, "paths":[...]}"""
    if isinstance(val, str):
        try:
            obj = json.loads(val)
        except Exception:
            return []
    elif isinstance(val, dict):
        obj = val
    else:
        return []
    paths = obj.get("paths") or []
    return [str(p) for p in paths]

def get_certificate_image_paths(parquet_path: str, base_dir: Optional[str]) -> List[str]:
    df = load_table(parquet_path)
    cert_rows = find_certificate_rows(df)
    paths = []
    if cert_rows.empty:
        return paths
    for _, row in cert_rows.iterrows():
        info = row.get("images_info", "")
        for p in parse_images_info(info):
            resolved = resolve_path(p, parquet_path, base_dir=base_dir)
            paths.append(resolved)
    # de-dup while preserving order
    seen, uniq = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p); uniq.append(p)
    return uniq

# ---------- orchestrator ----------
def verify_from_parquet(parquet_path: str, data_csv: str, base_dir: Optional[str] = None, lang: str = "eng") -> Dict[str, Any]:
    if not ensure_tesseract():
        return {"ok": False, "error": "Tesseract not available"}
    img_paths = get_certificate_image_paths(parquet_path, base_dir)
    if not img_paths:
        return {"ok": False, "error": "No certificate images found in parquet"}
    # use first image for this run (extend to loop if you expect multiple)
    img_path = img_paths[0]
    if not os.path.isfile(img_path):
        return {"ok": False, "error": f"Resolved image not found on disk: {img_path}"}

    ocr_text = ocr_image(img_path, lang=lang)
    fields = extract_fields(ocr_text)
    ds = load_table(data_csv)
    ok, score, match = match_record(fields, ds)

    return {
        "ok": ok,
        "similarity": round(float(score), 4),
        "image_path": img_path,
        "extracted": fields,
        "datasheet_match": match if ok else {},
        "ocr_preview": ocr_text[:400],
    }

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Verify certificate image referenced inside a parquet against a datasheet.")
    ap.add_argument("--parquet", required=True, help="Path to the parquet that contains the certificate slide and images_info")
    ap.add_argument("--data", required=True, help="Path to data.csv (datasheet)")
    ap.add_argument("--base-dir", default="", help="Optional base dir to resolve relative image paths")
    ap.add_argument("--lang", default="eng", help="Tesseract language (default eng)")
    ap.add_argument("--out", default="cert_verification.csv", help="CSV summary output")
    args = ap.parse_args()

    res = verify_from_parquet(args.parquet, args.data, base_dir=(args.base_dir or None), lang=args.lang)
    print(json.dumps(res, indent=2, ensure_ascii=False))

    # save tiny summary
    try:
        pd.DataFrame([{
            "ok": res.get("ok"),
            "similarity": res.get("similarity"),
            "image_path": res.get("image_path"),
            "name_extracted": res.get("extracted", {}).get("name"),
            "program_extracted": res.get("extracted", {}).get("program"),
            "code_extracted": res.get("extracted", {}).get("code"),
        }]).to_csv(args.out, index=False)
        print(f"[INFO] wrote {args.out}")
    except Exception as e:
        print(f"[WARN] could not write CSV: {e}")

if __name__ == "__main__":
    main()