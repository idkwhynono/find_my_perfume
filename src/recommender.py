import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches


def build_model(df):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(df["notes"])
    similarity = cosine_similarity(matrix)
    return similarity

def find_perfume(df, user_perfume):
    # Exact perfume name
    if user_perfume in df["name"].values:
        return user_perfume, "exact"
    
    # Partially matched name
    matches = df[df["name"].str.contains(user_perfume, case=False)]

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

def recommend(df, perfume_name, similarity_matrix, top_n=5):
    idx = df[df["name"] == perfume_name].index[0]
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    perfume_recommendations = []
    
    for index, score in similarity_scores:
        perfume_recommendations.append({
            "name": df.iloc[index]["name"],
            "brand": df.iloc[index]["brand"],
            "similarity": score * 100
        })
    return perfume_recommendations
    
    
    
    #return df.iloc[[i[0] for i in similarity_scores]][["name", "brand"]].reset_index(drop=True)

