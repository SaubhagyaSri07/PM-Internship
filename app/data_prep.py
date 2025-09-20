import pandas as pd
import re

def load_and_prep(Internship549_csv):
    df = pd.read_csv(Internship549_csv)
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    # Merge skills into a single list-string
    skill_cols = [c for c in df.columns if c.lower().startswith('skill')]
    df['Skills'] = df[skill_cols].fillna('').agg(' '.join, axis=1)
    # Clean text function
    def clean(s):
        if pd.isna(s): return ''
        s = str(s).lower()
        s = re.sub(r'[^a-z0-9, ]+', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s
    for col in ['Title','Company','Description','Required_Education','Sector','Location','Skills']:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(clean)
    # Optionally split skills into list
    df['Skills_list'] = df['Skills'].apply(lambda s: [x.strip() for x in s.split() if x.strip()])
    return df