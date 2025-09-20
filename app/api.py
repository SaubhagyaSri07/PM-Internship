from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from data_prep import load_and_prep
from recommender import build_tfidf, get_recommendations

app = FastAPI()

# Load dataset and build TF-IDF models once
df = load_and_prep('data/Internship549.csv')
vec_desc, mat_desc, vec_skill, mat_skill = build_tfidf(df)

class Candidate(BaseModel):
    education: str = ''
    skills: list = []
    sector: list = []
    location: list = []

@app.post('/recommend')
def recommend(candidate: Candidate, top_k: int = 5):
    candidate_dict = candidate.dict()
    recs = get_recommendations(df, vec_desc, mat_desc, vec_skill, mat_skill, candidate_dict, top_k=top_k)
    return recs.to_dict(orient='records')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
