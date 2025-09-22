# app/ui_streamlit.py
import os
import streamlit as st
import requests
from datetime import datetime
from data_prep import load_and_prep
from recommender import build_tfidf, get_recommendations

# ----- Config -----
DATA_PATH = os.environ.get("INTERNSHIP_CSV", "data/Internship549.csv")
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
USE_GEMINI = str(os.environ.get("USE_GEMINI", "")).lower() in ("1", "true", "yes")
GEMINI_KEY_PRESENT = bool(os.environ.get("GEMINI_API_KEY"))

# ----- UI strings (multi-language kept minimal) -----
OFFLINE_UI = {
    "en": {"title": "InternLink", "education": "Education Level",
           "undergrad_field": "Undergraduate Field (required)", "postgrad_field": "Postgraduate Field (required)",
           "skills": "Your Skills", "sectors": "Preferred Sectors", "locations": "Preferred Locations",
           "get_recs": "Get Recommendations", "no_recs": "No recommendations generated yet. Fill inputs and click 'Get Recommendations'.",
           "message_select_field": "Please select a field to proceed.", "why_recommended": "Why Recommended",
           "backend": "Backend", "gemini": "Gemini AI"},
    "hi": {"title":"InternLink","education":"शैक्षिक स्तर","undergrad_field":"अंडरग्रेजुएट फ़ील्ड (अनिवार्य)",
           "postgrad_field":"पोस्टग्रेज्यूएट फ़ील्ड (अनिवार्य)","skills":"आपके कौशल","sectors":"पसंदीदा क्षेत्र","locations":"पसंदीदा स्थान",
           "get_recs":"सिफारिशें प्राप्त करें","no_recs":"कोई सिफारिशें नहीं बनीं। इनपुट भरें और 'सिफारिशें प्राप्त करें' पर क्लिक करें।",
           "message_select_field":"कृपया आगे बढ़ने के लिए एक फ़ील्ड चुनें।","why_recommended":"सुझाई क्यों गई",
           "backend":"बैकेंड","gemini":"जेमिनी एआई"},
    "bn": {"title":"InternLink","education":"শিক্ষাগত স্তর","undergrad_field":"অন্ডারগ্রাজুয়েট ফিল্ড (প্রয়োজনীয়)",
           "postgrad_field":"পোস্টগ্রাজুয়েট ফিল্ড (প্রয়োজনীয়)","skills":"আপনার দক্ষতা","sectors":"পছন্দসই ক্ষেত্র","locations":"পছন্দসই স্থান",
           "get_recs":"সুপারিশ পান","no_recs":"কোনও সুপারিশ তৈরি হয়নি। ইনপুট পূরণ করে 'সুপারিশ পান' চাপুন।",
           "message_select_field":"অগ্রসর হতে একটি ফিল্ড নির্বাচন করুন।","why_recommended":"কেন সুপারিশ করা হয়েছে",
           "backend":"ব্যাকএন্ড","gemini":"জেমিনি এআই"},
    "mr": {"title":"InternLink","education":"शैक्षणिक स्तर","undergrad_field":"अंडरग्रॅज्युएट फील्ड (आवश्यक)",
           "postgrad_field":"पोस्टग्रॅज्युएट फील्ड (आवश्यक)","skills":"आपले कौशल्य","sectors":"प्राधान्य क्षेत्र","locations":"प्राधान्य ठिकाणे",
           "get_recs":"शिफारशी मिळवा","no_recs":"कुठल्या शिफारशी तयार झाल्या नाहीत. इनपुट भरा आणि 'शिफारशी मिळवा' वर क्लिक करा.",
           "message_select_field":"कृपया पुढे जाण्यासाठी एक फील्ड निवडा.","why_recommended":"शिफारस का केली",
           "backend":"बॅकएंड","gemini":"जेमिनी एआय"},
    "te": {"title":"InternLink","education":"విద్యా స్థాయి","undergrad_field":"అండర్‌గ్రాడ్యుయేట్ ఫీల్డ్ (అవసరం)",
           "postgrad_field":"పోస్ట్‌గ్రాడ్యుయేట్ ఫీల్డ్ (అవసరం)","skills":"మీ నైపుణ్యాలు","sectors":"ఇష్టమైన రంగాలు","locations":"ఇష్ట స్థలాలు",
           "get_recs":"సిఫారసులు పొందు","no_recs":"ఏ సిఫారసులు కనిపించలేదు. ఇన్‌పుట్ నింపి 'సిఫారసులు పొందు'ను క్లిక్ చేయండి.",
           "message_select_field":"దయచేసి ముందుకు వెళ్ల ఫీల్డ్ ఎంచుకోండి.","why_recommended":"ఎందుకు సిఫారసు చేయబడింది",
           "backend":"బ్యాక్‌ఎండ్","gemini":"జెమిని ఎఐ"}
}

def ui_str(key, lang):
    return OFFLINE_UI.get(lang, OFFLINE_UI["en"]).get(key, OFFLINE_UI["en"].get(key, key))

# ----- Theme handling (session_state) -----
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

def set_theme(mode: str):
    st.session_state["theme"] = mode

def inject_theme_css(theme: str):
    # font + color tokens
    st.markdown("<link href='https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap' rel='stylesheet'>", unsafe_allow_html=True)

    if theme == "dark":
        bg = "#071027"
        hero_bg = "linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01))"
        card = "#081226"
        text = "#E6F0FA"
        secondary = "#9bb7d6"
        accent = "#1e90ff"
        btn_bg = "#0ea5e9"
        btn_text = "#042027"
        page_light = "#071a2b"
        input_label = "#dbeefd"
    else:
        bg = "#eaf6ff"
        hero_bg = "linear-gradient(90deg, rgba(0,0,0,0.02), rgba(0,0,0,0.01))"
        card = "#ffffff"
        text = "#072033"
        secondary = "#31577a"
        accent = "#0066cc"
        btn_bg = "#0066cc"
        btn_text = "#ffffff"
        page_light = "#f3fbff"
        input_label = "#0b1720"

    css = """
    <style>
    :root {{ --bg: {bg}; --card: {card}; --text: {text}; --secondary: {secondary};
             --accent: {accent}; --btn-bg: {btn_bg}; --btn-text: {btn_text}; --page-light: {page_light}; --input-label: {input_label}; }}
    html, body, .stApp {{ background: var(--bg) !important; color: var(--text) !important; font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; }}

    .hero {{ padding: 18px; border-radius: 12px; background: {hero_bg}; margin-bottom: 14px; border: 1px solid rgba(0,0,0,0.04); }}
    .hero h1 {{ margin: 0; font-size: 34px; letter-spacing: 0.3px; color: var(--text); }}
    .hero p {{ margin: 6px 0 0 0; color: var(--secondary); }}

    .big-recommend-btn button {{
        background-color: var(--btn-bg) !important;
        color: var(--btn-text) !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        padding: 12px 18px !important;
        border-radius: 10px !important;
        width: 100% !important;
        box-shadow: 0 8px 24px rgba(2,6,23,0.12) !important;
        border: none !important;
    }}
    .big-recommend-btn button:hover {{ transform: translateY(-1px); opacity: .98; }}

    .rec-card {{
        background: var(--card);
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 18px;
        border: 1px solid rgba(0,0,0,0.06);
        box-shadow: 0 8px 28px rgba(2,6,23,0.06);
    }}
    .rec-title {{ font-size: 24px; font-weight: 800; color: var(--text); margin-bottom:6px; }}
    .rec-sub {{ font-size: 16px; color: var(--secondary); margin-bottom:10px; }}
    .rec-meta {{ font-size:15px; color:var(--secondary); margin-right:12px; }}
    .rec-score {{ font-weight:800; font-size:20px; color:var(--accent); }}

    .badge {{ display:inline-block;padding:8px 10px;border-radius:999px;font-size:13px;margin-right:8px;background: rgba(0,0,0,0.03); color:var(--text); }}

    .dot-pulse {{ display:inline-block;width:14px;height:14px;border-radius:50%;background-color:#28a745;box-shadow:0 0 0 0 rgba(40,167,69,0.7);animation:pulse 1.6s infinite;margin-right:8px; }}
    @keyframes pulse {{ 0% {{ box-shadow: 0 0 0 0 rgba(40,167,69,0.7); }} 70% {{ box-shadow: 0 0 0 10px rgba(40,167,69,0); }} 100% {{ box-shadow: 0 0 0 0 rgba(40,167,69,0); }} }}
    .dot-yellow {{ display:inline-block;width:14px;height:14px;border-radius:50%;background:#ffc107;margin-right:8px;box-shadow:none; }}

    /* ---- Force color of labels / dropdown headers / slider labels ---- */
    /* generic label tags */
    label, .stLabel, .stText, .stMarkdown p, .css-1v3fvcr, .css-1q8dd3e {{ color: var(--input-label) !important; font-weight:600 !important; }}
    /* Streamlit's selectbox/multiselect wrappers */
    div[data-testid="stSelectbox"] label, div[data-testid="stMultiSelect"] label, div[data-testid="stSlider"] label,
    div[data-testid="stNumberInput"] label, div[data-testid="stTextInput"] label {{
        color: var(--input-label) !important;
        font-weight:700 !important;
    }}
    /* Selected chips / pills text (attempt to color items inside multiselect) */
    .stMultiSelect .css-7ybxfn, .stMultiSelect .css-1noyq3l, .stMultiSelect .css-1gkcyyc {{
        color: var(--text) !important;
    }}
    /* Force the big CTA text color too */
    .big-recommend-btn button {{ color: var(--btn-text) !important; }}

    /* increase label font sizes for better presence */
    .stSelectbox label, .stMultiSelect label, .stSlider label, label {{ font-size:15px !important; }}

    .top-right-controls {{ display:flex; justify-content:flex-end; align-items:center; gap:12px; margin-bottom:8px; }}
    </style>
    """.format(bg=bg, card=card, text=text, secondary=secondary, accent=accent,
               btn_bg=btn_bg, btn_text=btn_text, page_light=page_light, input_label=input_label, hero_bg=hero_bg)

    st.markdown(css, unsafe_allow_html=True)

# Inject initial theme CSS
inject_theme_css(st.session_state["theme"])

# ----- Data and models (cached) ----- #
@st.cache_data
def load_data_cached(fp):
    df = load_and_prep(fp)
    if "Skills_list" in df.columns:
        df["Skills_list"] = df["Skills_list"].apply(lambda x: tuple(x) if isinstance(x, list) else x)
    return df

@st.cache_resource
def build_models_cached(df):
    return build_tfidf(df)

df = load_data_cached(DATA_PATH)
vec_desc, mat_desc, vec_skill, mat_skill = build_models_cached(df)

# ----- Backend health ----- #
@st.cache_data(ttl=15)
def check_backend_health():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if r.ok:
            return r.json()
        return {"backend": "unreachable"}
    except Exception:
        return {"backend": "unreachable"}

health = check_backend_health()
backend_ok = health.get("backend") == "ok"
generative_host_reachable = bool(health.get("generative_host_reachable"))
gemini_key_present = GEMINI_KEY_PRESENT

# ----- Top row: language + theme toggle + indicators ----- #
col_left, col_right = st.columns([5, 2])
with col_left:
    lang_options = {"English":"en","हिन्दी":"hi","বাংলা":"bn","मराठी":"mr","తెలుగు":"te"}
    lang_choice = st.selectbox("Language / भाषा", list(lang_options.keys()), index=0)
    LANG = lang_options[lang_choice]
with col_right:
    theme_toggle = st.checkbox("Light Mode", value=(st.session_state["theme"] == "light"))
    if theme_toggle and st.session_state["theme"] != "light":
        set_theme("light")
        inject_theme_css("light")
    elif (not theme_toggle) and st.session_state["theme"] != "dark":
        set_theme("dark")
        inject_theme_css("dark")

    # backend indicator (green pulse if ok, yellow if not)
    if backend_ok:
        st.markdown("<div style='display:flex;align-items:center;gap:6px;margin-top:6px;'><span class='dot-pulse'></span><strong>Backend: <span style='color:green'>Online</span></strong></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='display:flex;align-items:center;gap:6px;margin-top:6px;'><span class='dot-yellow'></span><strong>Backend: <span style='color:orange'>Offline</span></strong></div>", unsafe_allow_html=True)

    # Gemini indicator: pulse green if key present (per your request), else yellow
    if gemini_key_present:
        st.markdown("<div style='display:flex;align-items:center;gap:6px;margin-top:6px;'><span class='dot-pulse'></span><strong>Gemini AI: <span style='color:green'>Online (Key)</span></strong></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='display:flex;align-items:center;gap:6px;margin-top:6px;'><span class='dot-yellow'></span><strong>Gemini AI: <span style='color:orange'>Unavailable</span></strong></div>", unsafe_allow_html=True)

# ----- Title / hero -----
st.markdown("<div class='hero'><h1 style='display:inline-block;margin-right:12px'>{}</h1><p>Simple, mobile-friendly internship suggestions — skills-first</p></div>".format(ui_str("title", LANG)), unsafe_allow_html=True)

# ----- Inputs -----
education_level_options_internal = ["any","10th","12th","undergraduate","post graduate","phd","others"]
education_display = [x.title() if x.isalpha() else x.upper() for x in education_level_options_internal]
education_map = dict(zip(education_display, education_level_options_internal))
education_sel = st.selectbox(ui_str("education", LANG), education_display, index=0)
education_level = education_map[education_sel]

undergrad_fields_internal = ["b.tech","b.sc","b.a","bba","bcom","mca","others"]
postgrad_fields_internal = ["mba","m.sc","m.tech","m.a","others"]
undergrad_display = [x.upper() if len(x)<=5 else x.title() for x in undergrad_fields_internal]
postgrad_display = [x.upper() if len(x)<=5 else x.title() for x in postgrad_fields_internal]
undergrad_map = dict(zip(["None"]+undergrad_display, [""]+undergrad_fields_internal))
postgrad_map = dict(zip(["None"]+postgrad_display, [""]+postgrad_fields_internal))

education_field = ""
if education_level == "undergraduate":
    sel = st.selectbox(ui_str("undergrad_field", LANG), ["None"]+undergrad_display, index=0)
    education_field = undergrad_map.get(sel, "")
elif education_level == "post graduate":
    sel = st.selectbox(ui_str("postgrad_field", LANG), ["None"]+postgrad_display, index=0)
    education_field = postgrad_map.get(sel, "")

# Skills / sectors / locations
all_skills_internal = sorted({s for skills in df.get("Skills_list", []) for s in (skills or [])}) if "Skills_list" in df.columns else []
all_skills_display = [s.title() for s in all_skills_internal]
skill_map = dict(zip(all_skills_display, all_skills_internal))

all_sectors_internal = sorted(df["Sector"].dropna().unique()) if "Sector" in df.columns else []
all_sectors_display = [s.title() for s in all_sectors_internal]
sector_map = dict(zip(all_sectors_display, all_sectors_internal))

all_locations_internal = sorted(df["Location"].dropna().unique()) if "Location" in df.columns else []
all_locations_display = [l.title() for l in all_locations_internal]
location_map = dict(zip(all_locations_display, all_locations_internal))

# <-- Removed default selections so multiselects start empty -->
skills_selected_display = st.multiselect(ui_str("skills", LANG), options=all_skills_display)
sectors_selected_display = st.multiselect(ui_str("sectors", LANG), options=all_sectors_display)
locations_selected_display = st.multiselect(ui_str("locations", LANG), options=all_locations_display)

skills_selected = [skill_map[d] for d in skills_selected_display]
sectors_selected = [sector_map[d] for d in sectors_selected_display]
locations_selected = [location_map[d] for d in locations_selected_display]

top_k = st.slider("Number of suggestions", min_value=1, max_value=10, value=5)

# ----- CTA -----
st.markdown("<div class='big-recommend-btn'>", unsafe_allow_html=True)
get_btn = st.button(ui_str("get_recs", LANG), key="big_recommend")
st.markdown("</div>", unsafe_allow_html=True)

# ----- Action -----
if get_btn:
    if (education_level == "undergraduate" and not education_field) or (education_level == "post graduate" and not education_field):
        st.warning(ui_str("message_select_field", LANG))
    else:
        candidate_education = education_field.lower().strip() if education_field else ("" if education_level == "any" else education_level.lower().strip())
        candidate = {"education": candidate_education, "skills": [s.lower().strip() for s in skills_selected],
                     "sector": [s.lower().strip() for s in sectors_selected], "location": [l.lower().strip() for l in locations_selected]}

        results = None
        if backend_ok and USE_GEMINI:
            try:
                r = requests.post(f"{BACKEND_URL}/recommend?top_k={top_k}&lang=en&refine_with_llm=true", json=candidate, timeout=10)
                if r.ok:
                    results = r.json()
            except Exception:
                results = None

        if results is None:
            recs = get_recommendations(df, vec_desc, mat_desc, vec_skill, mat_skill, candidate, top_k=top_k)
            if recs is None or recs.empty:
                st.info(ui_str("no_recs", LANG))
                results = []
            else:
                results = []
                for r in recs.to_dict(orient="records"):
                    results.append({
                        "Internship_ID": str(r.get("Internship_ID")),
                        "Title": r.get("Title",""),
                        "Company": r.get("Company",""),
                        "Location": r.get("Location",""),
                        "Skills": r.get("Skills",""),
                        "Sector": r.get("Sector",""),
                        "score": float(r.get("score", 0.0)),
                        "explain": r.get("explain", "")
                    })

        st.session_state["last_recs"] = results
        st.session_state["last_candidate"] = candidate

# ----- Render results -----
if st.session_state.get("last_recs"):
    for rec in st.session_state["last_recs"]:
        title = (rec.get("Title") or "").title()
        company = (rec.get("Company") or "").title()
        loc = (rec.get("Location") or "").title()
        skills = rec.get("Skills") or ""
        sector = (rec.get("Sector") or "").title()
        score = rec.get("score", 0.0)
        explain = rec.get("explain","")
        llm_summary = rec.get("llm_summary","") or rec.get("llm_reason","")

        card_html = f"""
        <div class="rec-card">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <div style="flex:1;">
              <div class="rec-title">{title}</div>
              <div class="rec-sub">{company} &nbsp; • &nbsp; <span class="muted">{sector}</span></div>
            </div>
            <div style="text-align:right;min-width:120px;">
              <div class="rec-score">{score:.1f}</div>
              <div class="muted">Match score</div>
            </div>
          </div>
          <div style="margin-top:10px;">
            <span class="rec-meta">📍 {loc}</span>
            <span class="rec-meta">🛠️ {skills}</span>
          </div>
          <div style="margin-top:12px;">
            <strong>🔍 {ui_str('why_recommended', LANG)}:</strong> <span class="muted">{explain}</span>
          </div>
        """
        if llm_summary:
            card_html += f"<div style='margin-top:10px;color:var(--secondary);'>🤖 {llm_summary}</div>"
        card_html += "</div>"

        st.markdown(card_html, unsafe_allow_html=True)
else:
    st.info(ui_str("no_recs", LANG))
