from difflib import get_close_matches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
 
 
def build_model(df):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["notes"].fillna(""))
    return vectorizer, tfidf_matrix
 
 
def find_perfume(df, user_perfume):
    """Return (official_name, match_type) or (None, None).
 
    match_type is "exact", "partial" or "fuzzy".
    """
    query = user_perfume.strip().lower()
    if not query:
        return None, None
 
    names = df["name"]
    lowered = names.str.lower()
 
    # 1. Exact match, ignoring case
    exact = names[lowered == query]
    if not exact.empty:
        return exact.iloc[0], "exact"
 
    # 2. The query appears somewhere inside a name
    partial = names[lowered.str.contains(query, regex=False)]
    if not partial.empty:
        return partial.iloc[0], "partial"
 
    # 3. Closest spelling
    close = get_close_matches(query, lowered.tolist(), n=1, cutoff=0.6)
    if close:
        return names[lowered == close[0]].iloc[0], "fuzzy"
 
    return None, None
 
 
def recommend(df, perfume_name, tfidf_matrix, top_n=5):
    perfume_index = df.index[df["name"] == perfume_name][0]
 
    scores = linear_kernel(tfidf_matrix[perfume_index], tfidf_matrix).ravel()
    scores[perfume_index] = 0  # don't recommend the perfume itself
 
    names = df["name"].tolist()
    brands = df["brand"].tolist()
    notes = df["notes"].tolist()
 
    seen_names = {perfume_name.casefold()}
    seen_notes = {frozenset(notes[perfume_index].lower().split())}
    results = []
 
    for idx in scores.argsort()[::-1]:
        if scores[idx] <= 0:
            break 
 
        name = names[idx].strip()
        if name.casefold() in seen_names:
            continue
 
        note_set = frozenset(notes[idx].lower().split())
        if note_set in seen_notes:
            continue
 
        seen_names.add(name.casefold())
        seen_notes.add(note_set)
 
        results.append({
            "name": name,
            "brand": str(brands[idx]).strip(),
            "similarity": scores[idx] * 100,
        })
 
        if len(results) == top_n:
            break
 
    return results
 