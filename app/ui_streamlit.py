import streamlit as st
from data_prep import load_and_prep
from recommender import build_tfidf, get_recommendations

@st.cache_data
def load_data(fp):
    return load_and_prep(fp)

df = load_data('data/Internship549.csv')
vec_desc, mat_desc, vec_skill, mat_skill = build_tfidf(df)

all_skills = sorted({skill for skills in df['Skills_list'] for skill in skills})
all_sectors = sorted(df['Sector'].dropna().unique())
all_locations = sorted(df['Location'].dropna().unique())

st.title("Internship Recommender (Prototype)")

education = st.selectbox("Education", ['any', 'b.tech', 'mba', 'ba', 'bsc', 'mca', 'others'])
skills_selected = st.multiselect("Your Skills", options=all_skills, default=["python", "sql"])
sectors_selected = st.multiselect("Preferred Sectors", options=all_sectors, default=["it"])
locations_selected = st.multiselect("Preferred Locations", options=all_locations, default=["jaipur"])

top_k = st.slider("Number of suggestions", min_value=1, max_value=10, value=5)

if st.button("Get Recommendations"):
    candidate = {
        'education': education if education != 'any' else '',
        'skills': skills_selected,
        'sector': [s.lower() for s in sectors_selected],
        'location': [l.lower() for l in locations_selected]
    }
    recs = get_recommendations(df, vec_desc, mat_desc, vec_skill, mat_skill, candidate, top_k=top_k)

    for _, r in recs.iterrows():
        st.markdown(f"### {r['Title'].title()} — *{r['Company'].title()}*")
        st.markdown(f"📍 Location: `{r['Location'].title()}` — **Score:** `{r['score']:.2f}`")
        st.markdown(f"🛠️ Skills Required: `{r['Skills']}`")
        st.markdown(f"📂 Sector: `{r['Sector'].title()}`")
        st.markdown(f"🔍 Why Recommended: *{r['explain'].capitalize()}*")
        st.markdown("---")
