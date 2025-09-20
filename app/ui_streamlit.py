# ui_streamlit.py
import streamlit as st
from data_prep import load_and_prep
from recommender import build_tfidf, get_recommendations
import os

DATA_PATH = os.environ.get('INTERNSHIP_CSV', 'data/Internship549.csv')

@st.cache_data
def load_data(fp):
    return load_and_prep(fp)

@st.cache_resource
def build_model(df):
    return build_tfidf(df)

df = load_data(DATA_PATH)
vec_desc, mat_desc, vec_skill, mat_skill = build_model(df)

# Safely extract options for multiselects (guard for missing columns)
all_skills = sorted({skill for skills in df['Skills_list'] for skill in skills}) if 'Skills_list' in df.columns else []
all_sectors = sorted(df['Sector'].dropna().unique()) if 'Sector' in df.columns else []
all_locations = sorted(df['Location'].dropna().unique()) if 'Location' in df.columns else []

st.title("Internship Recommender (Prototype)")

# --- Education hierarchy UI ---
education_level_options = ['any', '10th', '12th', 'undergraduate', 'post graduate', 'phd', 'others']
education_level = st.selectbox("Education Level", education_level_options, index=0)

# fields for undergraduate and postgraduate
undergrad_fields = ['b.tech', 'b.sc', 'b.a', 'bba', 'bcom', 'mca', 'others']
postgrad_fields = ['mba', 'm.sc', 'm.tech', 'm.a', 'others']

# show dependent field selector only when needed
education_field = ''
if education_level == 'undergraduate':
    education_field = st.selectbox("Undergraduate Field", ['none'] + undergrad_fields, index=0)
    if education_field == 'none':
        education_field = ''  # treat as no specific field selected
elif education_level == 'post graduate':
    education_field = st.selectbox("Postgraduate Field", ['none'] + postgrad_fields, index=0)
    if education_field == 'none':
        education_field = ''

# --- Skills / Sectors / Locations (unchanged dropdowns) ---
# safe defaults if typical values exist in options
default_skills = [s for s in ["python", "sql"] if s in all_skills]
default_sectors = [s for s in ["it"] if s in all_sectors]
default_locations = [l for l in ["jaipur"] if l in all_locations]

skills_selected = st.multiselect("Your Skills", options=all_skills, default=default_skills)
sectors_selected = st.multiselect("Preferred Sectors", options=all_sectors, default=default_sectors)
locations_selected = st.multiselect("Preferred Locations", options=all_locations, default=default_locations)

top_k = st.slider("Number of suggestions", min_value=1, max_value=10, value=5)

# --- Run Recommendation ---
if st.button("Get Recommendations"):
    # Build education string to pass to recommender:
    # Priority: if a specific field was chosen (e.g., b.tech) use that, else use level (e.g., 12th / phd)
    if education_field:
        candidate_education = education_field.lower().strip()
    else:
        # use the level, but do not use 'any' as a string to avoid filtering
        candidate_education = '' if education_level == 'any' else education_level.lower().strip()

    candidate = {
        'education': candidate_education,
        'skills': [s.lower().strip() for s in skills_selected],
        'sector': [s.lower().strip() for s in sectors_selected],
        'location': [l.lower().strip() for l in locations_selected]
    }

    recs = get_recommendations(df, vec_desc, mat_desc, vec_skill, mat_skill, candidate, top_k=top_k)

    if recs.empty:
        st.info("No matches found. Try less restrictive filters (e.g., remove location or sector).")
    else:
        for _, r in recs.iterrows():
            # display same UI as before
            st.markdown(f"### {r['Title'].title()} — *{r['Company'].title()}*")
            st.markdown(f"📍 Location: `{r['Location'].title()}` — **Score:** `{r['score']:.2f}`")
            st.markdown(f"🛠️ Skills Required: `{r['Skills']}`")
            st.markdown(f"📂 Sector: `{r['Sector'].title()}`")
            st.markdown(f"🔍 Why Recommended: *{r['explain'].capitalize()}*")
            st.markdown("---")
