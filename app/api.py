# app/api.py
import os
import json
import requests
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

# relative imports
from .data_prep import load_and_prep
from .recommender import build_tfidf, get_recommendations

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app.api")

# ---------- Load .env ----------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------- App ----------
app = FastAPI(title="Internship Recommender API (with Gemini)")

# ---------- Config ----------
DATA_CSV = os.environ.get(
    "INTERNSHIP_CSV",
    os.path.join(os.path.dirname(__file__), "..", "data", "Internship549.csv"),
)
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
# removed Translate API key usage as requested
USE_GEMINI_FLAG = str(os.environ.get("USE_GEMINI", "")).lower() in ("1", "true", "yes")

# ---------- Load Data + Models ----------
df = load_and_prep(DATA_CSV)
vec_desc, mat_desc, vec_skill, mat_skill = build_tfidf(df)

# ---------- Models ----------
class Candidate(BaseModel):
    education: str = ""
    skills: List[str] = []
    sector: List[str] = []
    location: List[str] = []

class RecommendResponseItem(BaseModel):
    Internship_ID: Optional[str]
    Title: str
    Company: Optional[str]
    Location: Optional[str]
    Skills: Optional[str]
    Sector: Optional[str]
    score: float
    explain: str
    llm_summary: Optional[str] = None

# ---------- Gemini Helper ----------
def gemini_generate(prompt: str, model: str = "models/gemini-1.5-flash-latest", timeout: int = 15) -> Optional[str]:
    """Call Gemini API with a prompt and return generated text"""
    if not GEMINI_KEY:
        logger.warning("Gemini key not set, skipping generation.")
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/{model}:generateContent?key={GEMINI_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        logger.info(f"Calling Gemini: model={model}, prompt length={len(prompt)}")
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        j = r.json()
        logger.debug(f"Gemini raw response: {json.dumps(j)[:400]}...")

        candidates = j.get("candidates") or []
        if candidates:
            first = candidates[0]
            content = first.get("content", {}) or {}
            parts = content.get("parts") or []
            if parts and isinstance(parts, list):
                text = parts[0].get("text") if isinstance(parts[0], dict) else None
                if isinstance(text, str):
                    return text.strip()

        logger.warning("Gemini returned no text parts.")
        return None
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return None

# ---------- Translation (Gemini-only fallback) ----------
def translate_text(text: str, target_lang: str) -> str:
    """
    Translate text -> target_lang.
    Since Google Translate API key is removed, we use Gemini if available.
    If Gemini not available or fails, return original text.
    """
    if not text:
        return text
    if not GEMINI_KEY:
        # no external translate available
        return text
    prompt = f"Translate the following text into {target_lang}. Return only the translated text without extra commentary:\n\n{text}"
    resp = gemini_generate(prompt)
    return resp or text

# ---------- LLM Refinement ----------
def llm_refine_and_explain(candidate: dict, items: List[dict], top_k: int = 5) -> List[dict]:
    if not GEMINI_KEY:
        logger.warning("No Gemini key set, skipping refinement.")
        return items

    prompt_lines = [
        "Rank these internship postings for this candidate.",
        f"Candidate: Education={candidate.get('education','')}; Skills={', '.join(candidate.get('skills',[]))}; Sectors={', '.join(candidate.get('sector',[]))}; Locations={', '.join(candidate.get('location',[]))}",
        "",
        "Internships (id: title | skills):",
    ]
    for it in items:
        prompt_lines.append(f"{it.get('Internship_ID')}: {it.get('Title','')[:200]} | {it.get('Skills','')[:200]}")
    prompt_lines.append("")
    prompt_lines.append(
        f"Return a JSON array of up to {top_k} objects ordered best-to-worst. "
        "Each object must be {\"id\":<id>,\"rank\":<1..N>,\"reason\":\"one short sentence\"}. Return JSON only."
    )

    prompt = "\n".join(prompt_lines)
    text = gemini_generate(prompt)

    if not text:
        logger.warning("Gemini returned empty response. Skipping refinement.")
        return items

    logger.info(f"Gemini raw text snippet: {text[:200]}...")

    try:
        parsed = json.loads(text)
    except Exception:
        # try extracting JSON substring
        import re
        m = re.search(r"(\[.*\])", text, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(1))
            except Exception as e:
                logger.error(f"Gemini parse error after regex extract: {e}")
                return items
        else:
            logger.error("Gemini response not JSON.")
            return items

    if isinstance(parsed, list):
        mapped = {str(p.get("id")): p for p in parsed if p.get("id") is not None}
        out = []
        for it in items:
            iid = str(it.get("Internship_ID"))
            rec = it.copy()
            if iid in mapped:
                rec["llm_reason"] = mapped[iid].get("reason", "")
                rec["llm_rank"] = mapped[iid].get("rank")
            out.append(rec)
        if any("llm_rank" in o for o in out):
            out.sort(key=lambda x: int(x.get("llm_rank", 999)))
        logger.info("Gemini refinement applied successfully.")
        return out

    logger.error("Gemini response format invalid.")
    return items

# ---------- Endpoints ----------
@app.get("/health")
def health():
    generative_ok = False
    gemini_present = bool(GEMINI_KEY)
    translate_usable = bool(GEMINI_KEY)  # since no separate translate key, use Gemini presence

    try:
        r = requests.head("https://generativelanguage.googleapis.com", timeout=3)
        generative_ok = r.status_code in (200, 301, 302, 401, 403)
    except Exception:
        generative_ok = False

    return {
        "backend": "ok",
        "use_gemini_flag": USE_GEMINI_FLAG,
        "gemini_key_present": gemini_present,
        "translate_key_present": False,          # removed translate key completely
        "generative_host_reachable": generative_ok,
        "translate_usable": translate_usable,
    }

@app.post("/recommend", response_model=List[RecommendResponseItem])
def recommend(candidate: Candidate, top_k: int = 5, lang: str = "en", refine_with_llm: bool = True):
    candidate_dict = candidate.dict()
    logger.info(f"Received recommend request: {candidate_dict}")

    recs = get_recommendations(df, vec_desc, mat_desc, vec_skill, mat_skill, candidate_dict, top_k=top_k)
    if recs is None or recs.empty:
        return []

    rec_list = recs.to_dict(orient="records")

    # --- Try Gemini refine ---
    if USE_GEMINI_FLAG and GEMINI_KEY and refine_with_llm:
        try:
            pool = get_recommendations(df, vec_desc, mat_desc, vec_skill, mat_skill, candidate_dict, top_k=min(20, len(df)))
            pool_items = pool.to_dict(orient="records")
            refined = llm_refine_and_explain(candidate_dict, pool_items, top_k=top_k)
            if refined:
                rec_list = refined[:top_k]
        except Exception as e:
            logger.error(f"Gemini refine error: {e}")

    # --- Translate fields if needed (Gemini-only) ---
    if lang and lang != "en":
        for r in rec_list:
            try:
                r["Title"] = translate_text(r.get("Title", ""), lang)
                r["explain"] = translate_text(r.get("explain", ""), lang)
                if r.get("llm_reason"):
                    r["llm_reason"] = translate_text(r.get("llm_reason", ""), lang)
            except Exception as e:
                logger.error(f"Translate (Gemini) error: {e}")

    # --- Build Response ---
    out = []
    for r in rec_list:
        out.append({
            "Internship_ID": str(r.get("Internship_ID")),
            "Title": r.get("Title", "") or "",
            "Company": r.get("Company", "") or "",
            "Location": r.get("Location", "") or "",
            "Skills": r.get("Skills", "") or "",
            "Sector": r.get("Sector", "") or "",
            "score": float(r.get("score", 0.0)),
            "explain": r.get("explain", "") or "",
            "llm_summary": r.get("llm_reason", None),
        })
    return out

@app.post("/translate")
def translate_endpoint(payload: dict):
    text = payload.get("text", "")
    target = payload.get("target_lang", "hi")
    translated = translate_text(text, target)
    return {"original": text, "translated": translated}
