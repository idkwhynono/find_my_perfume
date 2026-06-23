import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_model(df):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(df["notes"])
    similarity = cosine_similarity(matrix)
    return similarity


def recommend(df, perfume_name, similarity_matrix, top_n=5):
    idx = df[df["name"] == perfume_name].index[0]
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    return df.iloc[[i[0] for i in similarity_scores]][["name", "brand"]].reset_index(drop=True)