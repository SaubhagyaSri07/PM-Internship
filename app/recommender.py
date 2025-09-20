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


# --- top-level tokenizer (PICKLABLE) ---
def skill_tokenizer(s):
    """
    Tokenize skills by comma while preserving multi-word skills.
    This is defined at module level so objects using it can be pickled.
    """
    return [tok.strip() for tok in str(s).split(',') if tok.strip()]


def build_tfidf(df, max_features_desc=5000, max_features_skill=1000):
    desc_corpus = (df['Title'].fillna('') + ' ' + df['Description'].fillna('')).tolist()
    vec_desc = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features_desc)
    mat_desc = vec_desc.fit_transform(desc_corpus)

    # Use the top-level tokenizer; set token_pattern=None to silence sklearn warnings
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


# Education groups mapping (same as before)
EDU_GROUPS = {
    'undergraduate': {
        'b.tech', 'btech', 'b.e', 'be', 'b.sc', 'bsc', 'b.a', 'ba', 'bba', 'bcom', 'b.com', 'bca', 'bachelor', 'bachelors'
    },
    'post graduate': {
        'm.tech', 'mtech', 'm.sc', 'msc', 'm.a', 'ma', 'mba', 'mcom', 'masters', 'master'
    },
    'phd': {'phd', 'doctor', 'doctoral'}
}


def normalize_edu_string(s):
    if not s:
        return ''
    s = str(s).lower()
    s = s.replace('.', ' ').replace('/', ' ').replace('-', ' ')
    s = ' '.join(s.split())
    return s.strip()


def internship_edu_tokens(intern_req_edu):
    r = normalize_edu_string(intern_req_edu)
    tokens = set()
    for part in [p.strip() for p in r.replace(',', ' ').split() if p.strip()]:
        tokens.add(part)
    return tokens


def education_match_score(candidate_edu, intern_req_edu):
    intern_req = (intern_req_edu or '').strip().lower()
    if not intern_req or intern_req in ('any', 'any graduate', 'any degree'):
        return 1.0

    cand = normalize_edu_string(candidate_edu or '')
    intern_tokens = internship_edu_tokens(intern_req)

    if not cand:
        return 1.0

    if cand in EDU_GROUPS:
        group = EDU_GROUPS[cand]
        for t in intern_tokens:
            if t in group:
                return 1.0
        for t in intern_tokens:
            for g in group:
                if g in t or t in g:
                    return 1.0
        return 0.0

    cand_token = cand.replace(' ', '')
    for t in intern_tokens:
        if cand_token == t.replace(' ', ''):
            return 1.0
        if cand_token in t or t in cand_token:
            return 0.8

    for group_name, group_tokens in EDU_GROUPS.items():
        if any(t in group_tokens or any(gt in t or t in gt for gt in group_tokens) for t in intern_tokens):
            if cand in group_name:
                return 1.0
            return 0.5

    return 0.0


def get_recommendations(df, vec_desc, mat_desc, vec_skill, mat_skill, candidate, top_k=5):
    df_local = df.copy().reset_index(drop=True)

    cand_desc_vec = vec_desc.transform([candidate_text_for_desc(candidate)])
    cand_skill_vec = vec_skill.transform([candidate_text_for_skills(candidate)])

    desc_sim = cosine_similarity(cand_desc_vec, mat_desc).flatten()
    skill_sim = cosine_similarity(cand_skill_vec, mat_skill).flatten()

    cand_sectors = [s.lower().strip() for s in candidate.get('sector', [])]
    cand_locations = [l.lower().strip() for l in candidate.get('location', [])]
    cand_education = (candidate.get('education') or '').lower().strip()

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
    perc_scores = scores * 100.0

    df_local['score'] = perc_scores
    df_local['skill_sim'] = skill_sim
    df_local['desc_sim'] = desc_sim
    df_local['sector_match'] = sector_matches
    df_local['location_match'] = location_matches
    df_local['education_score'] = education_scores

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
    return top[[c for c in cols if c in top.columns]]
