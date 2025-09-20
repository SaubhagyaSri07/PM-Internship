import pandas as pd
import re
import os

def load_and_prep(Internship549_csv):
    csv_path = os.path.normpath(Internship549_csv)
    df = pd.read_csv(Internship549_csv)

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Identify skill columns (Skill_1..Skill_4)
    skill_cols = [c for c in df.columns if c.lower().startswith('skill')]

    # Merge skills using comma so we preserve multi-word skills (e.g., "data analysis")
    df['Skills'] = df[skill_cols].fillna('').agg(', '.join, axis=1).str.replace(r'\s+,', ',', regex=True).str.strip()

    # Clean text function (preserve commas)
    def clean(s):
        if pd.isna(s): return ''
        s = str(s).lower()
        s = re.sub(r'[^a-z0-9, ]+', ' ', s)   # keep commas and spaces
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    text_cols = ['Title','Company','Description','Required_Education','Sector','Location','Skills']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(clean)

    # Split Skills into list by comma to preserve multi-word skills
    df['Skills_list'] = df['Skills'].apply(lambda s: [x.strip() for x in s.split(',') if x.strip()])

    # Lowercase Sector/Location/Education and strip whitespace (already cleaned above)
    for col in ['Sector','Location','Required_Education']:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('').str.strip().str.lower()

    # Ensure Internship_ID exists and is unique
    if 'Internship_ID' not in df.columns:
        df.insert(0, 'Internship_ID', range(1, len(df) + 1))
    df = df.drop_duplicates(subset=['Internship_ID'])

    return df
