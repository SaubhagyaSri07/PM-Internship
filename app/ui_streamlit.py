# app/ui_streamlit.py
import streamlit as st
from data_prep import load_and_prep
from recommender import build_tfidf, get_recommendations
import os
import pandas as pd
import sqlite3
from datetime import datetime
import hashlib
import json

# ----- Config -----
DATA_PATH = os.environ.get('INTERNSHIP_CSV', 'data/Internship549.csv')
FEEDBACK_CSV = os.environ.get('FEEDBACK_CSV', 'feedback_log.csv')
DB_PATH = os.environ.get('FEEDBACK_DB', 'feedback_log.db')
ADMIN_KEY = os.environ.get('ADMIN_KEY', '')  # set to view admin logs
MAX_VIEW_ROWS = 200

# ----- Helpers: DB / CSV logging and anonymization -----
def init_db(path=DB_PATH):
    conn = sqlite3.connect(path, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS impressions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT,
        anon_id TEXT,
        candidate_json TEXT,
        internship_id TEXT,
        title TEXT,
        score REAL
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT,
        anon_id TEXT,
        candidate_json TEXT,
        internship_id TEXT,
        title TEXT,
        score REAL,
        feedback TEXT
    )
    """)
    conn.commit()
    return conn

def anonymize_candidate(candidate):
    payload = {
        'education': candidate.get('education',''),
        'skills': sorted([s.lower().strip() for s in candidate.get('skills', [])]),
        'sector': sorted([s.lower().strip() for s in candidate.get('sector', [])]),
        'location': sorted([l.lower().strip() for l in candidate.get('location', [])]),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    h = hashlib.sha256(raw.encode('utf-8')).hexdigest()
    return h, payload

def _append_csv(rowdict, csv_path=FEEDBACK_CSV):
    df = pd.DataFrame([rowdict])
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode='a', header=header, index=False)

def log_impression(conn, anon_id, candidate_payload, internship_id, title, score):
    ts = datetime.utcnow().isoformat()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO impressions (ts, anon_id, candidate_json, internship_id, title, score) VALUES (?,?,?,?,?,?)",
        (ts, anon_id, json.dumps(candidate_payload, ensure_ascii=False), str(internship_id), str(title), float(score) if score is not None else None)
    )
    conn.commit()
    _append_csv({
        "ts": ts, "type": "impression", "anon_id": anon_id,
        "candidate": json.dumps(candidate_payload, ensure_ascii=False),
        "internship_id": internship_id, "title": title, "score": score
    })

def log_feedback(conn, anon_id, candidate_payload, internship_id, title, score, feedback):
    ts = datetime.utcnow().isoformat()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO feedback (ts, anon_id, candidate_json, internship_id, title, score, feedback) VALUES (?,?,?,?,?,?,?)",
        (ts, anon_id, json.dumps(candidate_payload, ensure_ascii=False), str(internship_id), str(title), float(score) if score is not None else None, feedback)
    )
    conn.commit()
    _append_csv({
        "ts": ts, "type": "feedback", "anon_id": anon_id,
        "candidate": json.dumps(candidate_payload, ensure_ascii=False),
        "internship_id": internship_id, "title": title, "score": score, "feedback": feedback
    })

def fetch_feedback_counts(conn, internship_ids):
    """
    Return mapping: {internship_id: {'useful': count, 'not_useful': count}}
    """
    if not internship_ids:
        return {}
    ids = [str(i) for i in internship_ids]
    placeholders = ','.join(['?'] * len(ids))
    q = f"""
    SELECT internship_id, feedback, COUNT(*) as cnt
    FROM feedback
    WHERE internship_id IN ({placeholders})
    GROUP BY internship_id, feedback
    """
    cur = conn.cursor()
    cur.execute(q, ids)
    rows = cur.fetchall()
    mapping = {}
    for internship_id, feedback, cnt in rows:
        mapping.setdefault(str(internship_id), {'useful': 0, 'not_useful': 0})
        if feedback == 'useful':
            mapping[str(internship_id)]['useful'] = cnt
        else:
            mapping[str(internship_id)]['not_useful'] = cnt
    for i in ids:
        mapping.setdefault(i, {'useful': 0, 'not_useful': 0})
    return mapping

# ----- Load data & models -----
@st.cache_data
def load_data(fp):
    df = load_and_prep(fp)
    if 'Skills_list' in df.columns:
        df['Skills_list'] = df['Skills_list'].apply(lambda x: tuple(x) if isinstance(x, list) else x)
    return df

@st.cache_resource
def build_models(df):
    return build_tfidf(df)

df = load_data(DATA_PATH)
vec_desc, mat_desc, vec_skill, mat_skill = build_models(df)

conn = init_db(DB_PATH)

# ----- UI -----
st.title("Internship Recommender — Prototype")

# Education selection (display-friendly)
education_level_options_internal = ['any', '10th', '12th', 'undergraduate', 'post graduate', 'phd', 'others']
education_level_display = [opt.title() if opt.isalpha() else opt.upper() if opt.isnumeric() else opt.title() for opt in education_level_options_internal]
education_level_map = dict(zip(education_level_display, education_level_options_internal))
education_level_choice = st.selectbox("Education Level", education_level_display, index=0)
education_level = education_level_map[education_level_choice]  # internal value

# UG/PG field options
undergrad_fields_internal = ['b.tech', 'b.sc', 'b.a', 'bba', 'bcom', 'mca', 'others']
postgrad_fields_internal = ['mba', 'm.sc', 'm.tech', 'm.a', 'others']
undergrad_display = [x.replace('.', '.').upper() if len(x)<=5 else x.title() for x in undergrad_fields_internal]
postgrad_display = [x.replace('.', '.').upper() if len(x)<=5 else x.title() for x in postgrad_fields_internal]
undergrad_map = dict(zip(['None'] + undergrad_display, [''] + undergrad_fields_internal))
postgrad_map = dict(zip(['None'] + postgrad_display, [''] + postgrad_fields_internal))

education_field = ''
if education_level == 'undergraduate':
    sel = st.selectbox("Undergraduate Field (required)", ['None'] + undergrad_display, index=0)
    education_field = undergrad_map.get(sel, '')
elif education_level == 'post graduate':
    sel = st.selectbox("Postgraduate Field (required)", ['None'] + postgrad_display, index=0)
    education_field = postgrad_map.get(sel, '')

# Skills / sectors / locations (display then map back)
all_skills_internal = sorted({skill for skills in df['Skills_list'] for skill in skills}) if 'Skills_list' in df.columns else []
all_skills_display = [s.title() for s in all_skills_internal]
skill_map = dict(zip(all_skills_display, all_skills_internal))

all_sectors_internal = sorted(df['Sector'].dropna().unique()) if 'Sector' in df.columns else []
all_sectors_display = [s.title() for s in all_sectors_internal]
sector_map = dict(zip(all_sectors_display, all_sectors_internal))

all_locations_internal = sorted(df['Location'].dropna().unique()) if 'Location' in df.columns else []
all_locations_display = [s.title() for s in all_locations_internal]
location_map = dict(zip(all_locations_display, all_locations_internal))

default_skills_display = [s.title() for s in all_skills_internal if s in ["python", "sql"]]
default_sectors_display = [s.title() for s in all_sectors_internal if s in ["it"]]
default_locations_display = [s.title() for s in all_locations_internal if s in ["jaipur"]]

skills_selected_display = st.multiselect("Your Skills", options=all_skills_display, default=default_skills_display)
sectors_selected_display = st.multiselect("Preferred Sectors", options=all_sectors_display, default=default_sectors_display)
locations_selected_display = st.multiselect("Preferred Locations", options=all_locations_display, default=default_locations_display)

# map back to internal tokens
skills_selected = [skill_map[d] for d in skills_selected_display]
sectors_selected = [sector_map[d] for d in sectors_selected_display]
locations_selected = [location_map[d] for d in locations_selected_display]

top_k = st.slider("Number of suggestions", min_value=1, max_value=10, value=5)

# Admin analytics reveal
with st.expander("Admin / Analytics (hidden)", expanded=False):
    admin_input = st.text_input("Enter admin key to view logs (leave blank to hide):", type="password")
    if ADMIN_KEY and admin_input == ADMIN_KEY:
        st.success("Admin access granted — showing logs.")
        imps = pd.read_sql_query(f"SELECT ts, anon_id, internship_id, title, score FROM impressions ORDER BY id DESC LIMIT {MAX_VIEW_ROWS}", conn)
        feeds = pd.read_sql_query(f"SELECT ts, anon_id, internship_id, title, score, feedback FROM feedback ORDER BY id DESC LIMIT {MAX_VIEW_ROWS}", conn)
        st.subheader("Recent impressions")
        st.dataframe(imps)
        st.subheader("Recent feedback")
        st.dataframe(feeds)
    else:
        st.info("Logs are hidden. Provide correct admin key to view.")

# session_state initialization
if 'last_candidate' not in st.session_state:
    st.session_state['last_candidate'] = None
if 'last_recs' not in st.session_state:
    st.session_state['last_recs'] = []
# feedback_status maps internship_id (str) -> 'useful'/'not_useful' for this session (prevents multiple votes)
if 'feedback_status' not in st.session_state:
    st.session_state['feedback_status'] = {}

# Build candidate & recommend (handler)
def build_candidate_and_recommend():
    # enforce UG/PG field selection
    if education_level == 'undergraduate' and not education_field:
        st.session_state['message'] = "Please select an Undergraduate field (e.g., B.Tech) to proceed."
        return
    if education_level == 'post graduate' and not education_field:
        st.session_state['message'] = "Please select a Postgraduate field (e.g., MBA) to proceed."
        return

    candidate_education = education_field.lower().strip() if education_field else ('' if education_level == 'any' else education_level.lower().strip())

    candidate = {
        'education': candidate_education,
        'skills': [s.lower().strip() for s in skills_selected],
        'sector': [s.lower().strip() for s in sectors_selected],
        'location': [l.lower().strip() for l in locations_selected]
    }
    st.session_state['last_candidate'] = candidate

    recs_df = get_recommendations(df, vec_desc, mat_desc, vec_skill, mat_skill, candidate, top_k=top_k)

    if recs_df is None or recs_df.empty:
        st.session_state['last_recs'] = []
        st.session_state['message'] = "No internships match your education level or filters. Try relaxing filters."
    else:
        st.session_state['last_recs'] = recs_df.to_dict(orient='records')
        st.session_state['message'] = f"Showing top {len(st.session_state['last_recs'])} internships (filtered by your education level)."

# feedback handler (called by button on_click). It will log and set session flag to prevent re-vote.
def handle_feedback(internship_id, title, score, feedback_label):
    candidate = st.session_state.get('last_candidate') or {'education':'','skills':[],'sector':[],'location':[]}
    anon_id, anon_payload = anonymize_candidate(candidate)
    try:
        # if user already voted this session, don't log again
        iid_str = str(internship_id)
        if st.session_state['feedback_status'].get(iid_str):
            return
        log_feedback(conn, anon_id, anon_payload, internship_id, title, score, feedback_label)
        st.session_state['feedback_status'][iid_str] = feedback_label
    except Exception:
        st.session_state['feedback_status'][iid_str] = 'error'

# Trigger recommendations (BIGGER + CLEAR BUTTON)
st.markdown(
    """
    <style>
    .big-recommend-btn button {
        background-color: #0066cc;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 0.75em 1.5em;
        border-radius: 10px;
        width: 100%;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.2);
    }
    .big-recommend-btn button:hover {
        background-color: #0052a3;
        transform: scale(1.02);
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.container():
    recommend_clicked = st.container()
    with recommend_clicked:
        if st.button("Get Recommendations", key="big_recommend", help="Generate tailored internship suggestions", use_container_width=True):
            build_candidate_and_recommend()

# show message if any
if st.session_state.get('message'):
    st.info(st.session_state.get('message'))

# Render recommendations
if st.session_state.get('last_recs'):
    intern_ids = [str(r.get('Internship_ID')) for r in st.session_state['last_recs']]
    counts_map = fetch_feedback_counts(conn, intern_ids)

    for rec in st.session_state['last_recs']:
        iid = str(rec.get('Internship_ID'))
        title = rec.get('Title','')
        score = float(rec.get('score', 0.0))

        st.markdown(f"### {rec.get('Title','').title()} — *{rec.get('Company','').title()}*")
        st.markdown(f"📍 Location: `{rec.get('Location','').title()}` — **Score:** `{score:.2f}`")
        st.markdown(f"🛠️ Skills Required: `{rec.get('Skills','')}`")
        st.markdown(f"📂 Sector: `{rec.get('Sector','').title()}`")
        st.markdown(f"🔍 Why Recommended: *{rec.get('explain','').capitalize()}*")

        # component badges
        skill_sim = float(rec.get('skill_sim', 0.0))
        desc_sim = float(rec.get('desc_sim', 0.0))
        edu_score = float(rec.get('education_score', 0.0)) if 'education_score' in rec else 0.0
        sector_match = int(rec.get('sector_match', 0)) if 'sector_match' in rec else 0
        location_match = int(rec.get('location_match', 0)) if 'location_match' in rec else 0

        st.markdown(
            f"<small>🔑 skill_sim: {skill_sim:.3f} | 📖 desc_sim: {desc_sim:.3f} | 🎓 edu: {edu_score:.2f} | 🏭 sector: {sector_match} | 🌍 loc: {location_match}</small>",
            unsafe_allow_html=True
        )

        # log impression once per session
        if not rec.get('_impression_logged'):
            try:
                anon_id, anon_payload = anonymize_candidate(st.session_state.get('last_candidate', {}))
                log_impression(conn, anon_id, anon_payload, rec.get('Internship_ID'), rec.get('Title'), score)
                rec['_impression_logged'] = True
            except Exception:
                pass

        # feedback counts
        counts = counts_map.get(iid, {'useful': 0, 'not_useful': 0})
        useful_count = counts.get('useful', 0)
        not_useful_count = counts.get('not_useful', 0)

        # determine if user already voted in this session
        user_vote = st.session_state['feedback_status'].get(iid)  # None / 'useful' / 'not_useful'

        # labels: change label to show checkmark if user voted that option
        label_useful = f"👍 Useful ({useful_count})"
        label_not = f"👎 Not useful ({not_useful_count})"
        if user_vote == 'useful':
            label_useful = f"✅ Useful ({useful_count})"
        if user_vote == 'not_useful':
            label_not = f"🚫 Not useful ({not_useful_count})"

        cols = st.columns([1,1,4])
        with cols[0]:
            # disable if user already voted
            st.button(label_useful, key=f"useful_{iid}", on_click=handle_feedback,
                      args=(rec.get('Internship_ID'), title, score, 'useful'),
                      disabled=(user_vote is not None))
        with cols[1]:
            st.button(label_not, key=f"notuseful_{iid}", on_click=handle_feedback,
                      args=(rec.get('Internship_ID'), title, score, 'not_useful'),
                      disabled=(user_vote is not None))
        with cols[2]:
            st.write("")

        st.markdown("---")

else:
    st.info("No recommendations generated yet. Fill inputs and click 'Get Recommendations'.")
