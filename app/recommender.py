import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configurable weights
WEIGHTS = {
    'skill': 0.6,
    'sector': 0.2,
    'semantic': 0.2,
    'location': 0.1  # extra small boost
}

def build_tfidf(df):
    corpus = (df['Skills'].fillna('') + ' ' + df['Description'].fillna('')).tolist()
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
    mat = vec.fit_transform(corpus)
    return vec, mat

def candidate_to_text(candidate):
    parts = []
    if candidate.get('education'): parts.append(candidate['education'])
    if candidate.get('skills'): parts.append(' '.join(candidate['skills']))
    if candidate.get('sector'): parts.append(' '.join(candidate['sector']))
    if candidate.get('location'): parts.append(' '.join(candidate['location']))
    return ' '.join(parts)

def skill_overlap_score(candidate_skills, internship_skills):
    if not candidate_skills: return 0.0
    cset = set([s.lower().strip() for s in candidate_skills if s])
    iset = set([s.lower().strip() for s in internship_skills if s])
    if not iset: return 0.0
    overlap = cset.intersection(iset)
    # normalized by candidate skill count to favor matches to user's skills
    return len(overlap) / max(1, len(cset))

def get_recommendations(df, vec, tfidf_mat, candidate, top_k=5):
    # Precompute candidate semantic vector
    cand_text = candidate_to_text(candidate)
    cand_vec = vec.transform([cand_text])
    sem_sim = cosine_similarity(cand_vec, tfidf_mat).flatten()  # shape (n_jobs,)

    scores = []
    for i, row in df.iterrows():
        s_skill = skill_overlap_score(candidate.get('skills', []), row.get('Skills_list', []))
        s_sector = 1.0 if ('sector' in candidate and row.get('Sector','') in [x.lower() for x in candidate.get('sector', [])]) else 0.0
        s_loc = 1.0 if ('location' in candidate and row.get('Location','') in [x.lower() for x in candidate.get('location', [])]) else 0.0
        s_sem = float(sem_sim[i])
        # normalize sem similarity (it's between 0-1 normally); skills & other signals already 0-1
        final = (WEIGHTS['skill'] * s_skill
                 + WEIGHTS['sector'] * s_sector
                 + WEIGHTS['semantic'] * s_sem
                 + WEIGHTS['location'] * s_loc)
        scores.append(final*100)  # scale to 0-100 for easier interpretability

    df['score'] = scores
    top = df.sort_values('score', ascending=False).head(top_k)
    # Add simple explanation
    def explain(row):
        reasons = []
        if skill_overlap_score(candidate.get('skills',[]), row.get('Skills_list',[]))>0:
            reasons.append('skill match')
        if row.get('Sector','') in [x.lower() for x in candidate.get('sector', [])]:
            reasons.append('sector match')
        if row.get('Location','') in [x.lower() for x in candidate.get('location', [])]:
            reasons.append('location match')
        return ', '.join(reasons) if reasons else 'semantic match'
    top = top.copy()
    top['explain'] = top.apply(explain, axis=1)
    return top[['Internship_ID','Title','Company','Location','Skills','Sector','score','explain']]
