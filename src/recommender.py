import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

def build_model(df):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["notes"].fillna(""))
    return vectorizer, tfidf_matrix

def find_perfume(df, user_perfume):
    # Exact perfume name
    if user_perfume in df["name"].values:
        return user_perfume, "exact"
    
    # Partially matched name
    matches = df[df["name"].str.contains(user_perfume, case=False, na=False)]

    if not matches.empty:
        return matches.iloc[0]["name"], "partial"
    
    # Fuzzy matching
    suggestioins = get_close_matches(
        user_perfume,
        df["name"].tolist(),
        n=1,
        cutoff=0.6
    )
    if suggestioins:
        return suggestioins[0], "fuzzy"
    return None, None

def recommend(df, perfume_name, tfidf_matrix, top_n=5):
    perfume_index = df.index[df["name"] == perfume_name][0]
    print("\n======================")
    print("SEARCHED PERFUME:", perfume_name)
    print("PERFUME INDEX:", perfume_index)
    print("======================")

    similarity_scores = cosine_similarity(
        tfidf_matrix[perfume_index],
        tfidf_matrix
    ).flatten()

    # Set the selected perfume's score to zero
    similarity_scores[perfume_index] = 0

    # Get the selected perfume's notes
    selected_notes = set(
        str(df.iloc[perfume_index]["notes"]).lower().split()
    )

    perfume_recommendations = []
    seen_notes = {tuple(sorted(selected_notes))}
    seen_names = {perfume_name.strip().casefold()}

    # Sort perfumes by similarity
    top_indices = similarity_scores.argsort()[::-1]
    print("TOP 5:")

    for idx in top_indices[:5]:
        print(
            df.iloc[idx]["name"],
            round(similarity_scores[idx] * 100, 2)
        )

    for idx in top_indices:
        name = str(df.iloc[idx]["name"]).strip()
        brand = str(df.iloc[idx]["brand"]).strip()

        key = (name.casefold(), brand.casefold())

        if key in seen_names:
            continue

        seen_names.add(key)
        
        if similarity_scores[idx] == 0:
            continue
        
        candidate_name = str(df.iloc[idx]["name"]).strip()

        if candidate_name.casefold() in seen_names:
            continue

        seen_names.add(candidate_name.casefold())

        

        # Get the candidate perfume's notes
        
        candidate_notes = set(
            str(df.iloc[idx]["notes"]).lower().split()
        )

        # Skip perfumes with identical note profiles
        note_key = tuple(sorted(candidate_notes))

        if note_key in seen_notes:
            continue

        seen_notes.add(note_key)

        perfume_recommendations.append({
            "name": df.iloc[idx]["name"],
            "brand": df.iloc[idx]["brand"],
            "similarity": similarity_scores[idx] * 100
        })

        if len(perfume_recommendations) == top_n:
            break
    
    return perfume_recommendations
    
    
    
    #return df.iloc[[i[0] for i in similarity_scores]][["name", "brand"]].reset_index(drop=True)