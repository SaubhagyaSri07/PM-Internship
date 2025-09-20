# recommender.py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Hierarchy: skills > sector > location > education
WEIGHTS = {
    'skill_sim': 0.60,   # highest priority
    'desc_sim': 0.20,    # semantic support (title+description)
    'sector': 0.12,      # sector match
    'location': 0.05,    # location preference (small boost)
    'education': 0.03    # smallest boost
}
# Normalize weights so they sum to 1
_total = sum(WEIGHTS.values())
for k in WEIGHTS:
    WEIGHTS[k] = WEIGHTS[k] / _total


def build_tfidf(df, max_features_desc=5000, max_features_skill=1000):
    desc_corpus = (df['Title'].fillna('') + ' ' + df['Description'].fillna('')).tolist()
    vec_desc = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features_desc)
    mat_desc = vec_desc.fit_transform(desc_corpus)

    def skill_tokenizer(s):
        return [tok.strip() for tok in str(s).split(',') if tok.strip()]

    vec_skill = TfidfVectorizer(tokenizer=skill_tokenizer, lowercase=True, token_pattern=None,
                                max_features=max_features_skill)
    mat_skill = vec_skill.fit_transform(df['Skills'].fillna('').tolist())

    return vec_desc, mat_desc, vec_skill, mat_skill


def candidate_text_for_desc(candidate):
    parts = []
    if candidate.get('education'):
        parts.append(candidate['education'])
    if candidate.get('skills'):
        parts.append(' '.join(candidate['skills']))
    if candidate.get('sector'):
        parts.append(' '.join(candidate['sector']))
    if candidate.get('location'):
        parts.append(' '.join(candidate['location']))
    return ' '.join(parts)


def candidate_text_for_skills(candidate):
    skills = candidate.get('skills', [])
    return ', '.join([s.strip() for s in skills if s.strip()])


def education_match_score(candidate_edu, intern_req_edu):
    """
    Simple education matching:
    - if internship says 'any' or '' -> treat as match (score 1).
    - if candidate_edu equals intern_req_edu -> 1
    - if candidate_edu contains intern_req_edu or vice-versa -> 0.8
    - else 0
    """
    if not intern_req_edu or intern_req_edu.strip() == '' or intern_req_edu.strip() == 'any':
        return 1.0
    if not candidate_edu:
        return 0.0
    c = candidate_edu.lower().strip()
    r = intern_req_edu.lower().strip()
    if c == r:
        return 1.0
    if c in r or r in c:
        return 0.8
    return 0.0


def get_recommendations(df, vec_desc, mat_desc, vec_skill, mat_skill, candidate, top_k=5):
    df_local = df.copy().reset_index(drop=True)

    # Candidate vectors
    cand_desc_vec = vec_desc.transform([candidate_text_for_desc(candidate)])
    cand_skill_vec = vec_skill.transform([candidate_text_for_skills(candidate)])

    desc_sim = cosine_similarity(cand_desc_vec, mat_desc).flatten()
    skill_sim = cosine_similarity(cand_skill_vec, mat_skill).flatten()

    cand_sectors = [s.lower().strip() for s in candidate.get('sector', [])]
    cand_locations = [l.lower().strip() for l in candidate.get('location', [])]
    cand_education = candidate.get('education', '').lower().strip() if candidate.get('education') else ''

    scores = []
    sector_matches = []
    location_matches = []
    education_scores = []

    for i, row in df_local.iterrows():
        s_skill = float(skill_sim[i]) if skill_sim.shape[0] > i else 0.0
        s_desc = float(desc_sim[i]) if desc_sim.shape[0] > i else 0.0

        s_sector = 1.0 if (row.get('Sector', '') and row.get('Sector', '') in cand_sectors) else 0.0
        sector_matches.append(s_sector)

        s_loc = 1.0 if (row.get('Location', '') and row.get('Location', '') in cand_locations) else 0.0
        location_matches.append(s_loc)

        s_edu = education_match_score(cand_education, row.get('Required_Education', ''))
        education_scores.append(s_edu)

        final = (
            WEIGHTS['skill_sim'] * s_skill +
            WEIGHTS['desc_sim'] * s_desc +
            WEIGHTS['sector'] * s_sector +
            WEIGHTS['location'] * s_loc +
            WEIGHTS['education'] * s_edu
        )
        scores.append(final)

    scores = np.array(scores)
    # Do NOT min-max normalize here: weights already keep score in 0-1 range (approx)
    # Convert to 0-100 percentage
    perc_scores = scores * 100.0

    df_local['score'] = perc_scores
    df_local['skill_sim'] = skill_sim
    df_local['desc_sim'] = desc_sim
    df_local['sector_match'] = sector_matches
    df_local['location_match'] = location_matches
    df_local['education_score'] = education_scores

    # Sorting with deterministic tie-breakers:
    # primary: score, then higher skill_sim, then sector_match, then location_match, then desc_sim
    top = df_local.sort_values(
        by=['score', 'skill_sim', 'sector_match', 'location_match', 'desc_sim'],
        ascending=[False, False, False, False, False]
    ).head(top_k).reset_index(drop=True)

    def explain(row):
        reasons = []
        if row['skill_sim'] > 0:
            reasons.append('skill match')
        if row.get('sector_match', 0) > 0:
            reasons.append('sector match')
        if row.get('location_match', 0) > 0:
            reasons.append('location match')
        if row.get('education_score', 0) > 0:
            reasons.append('education match')
        if not reasons:
            reasons.append('semantic match')
        return ', '.join(reasons)

    top = top.copy()
    top['explain'] = top.apply(explain, axis=1)

    cols = ['Internship_ID', 'Title', 'Company', 'Location', 'Skills', 'Sector',
            'Required_Education', 'score', 'explain', 'skill_sim', 'desc_sim']
    # return columns that exist
    return top[[c for c in cols if c in top.columns]]
