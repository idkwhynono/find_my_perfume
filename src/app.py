from PIL import Image
from html import escape
import streamlit as st
from data_loader import load_data
from recommender import build_model, find_perfume, recommend
import base64

st.set_page_config(page_title="Find My Perfume", page_icon="🧙🏼")

DATA_PATH = "data/perfumes.csv"
IMAGE_PATH = "assets/perfume_bottle.png"

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@600;0,700:1,500&family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*= "css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp,
[data-testid= "stAppViewContainer"],
[data-testid= "stHeader"] {
    background-color: #243B4A !important;
}

[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {
    background-color: #5E786C !important;
    color: #F2DFA0;
}

.title {
    font-size: 56px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 0:
}

h1.title,
.stMarkdown h1.title {
    font-family: 'Playfair Display', serif !important;
    font-size: 56px;
    font-weight: 700;
    text-align: center;
    color: #C9A64B !important;
    letter-spacing: 2px;
    margin-bottom: 0;
}
.subtitle {
    text-align: center;
    color: #A9BEC8;
    font-size: 17px;
}

div.stButton > button {
    background: #78B7B5;
    color: #1B303D;
    border: none;
    border-radius: 5px;
    font-weight: 700;
    padding: 12px;
}

div.stButton > button:hover {
    background: #A0D0CC;
}

/* Search box */
div[data-baseweb="input"] input {
    border-radius: 8px;
}

div[data-baseweb="input"]:focus-within {
    border-color: #426B8A;
}

/* Floating bottle */
@keyframes float {
    0%  { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
    100%    { transform: translateY(0px); }
}

.floatImage {
    display: flex;
    justify-content: center;
    margin-bottom: 20px;
}

.floatImage img {
    width: 170px;
    animation: float 4s ease-in-out infinite;
}
</style>
"""



@st.cache_data
def get_base64(path):
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode()

@st.cache_resource
def load_model():
    df = load_data(DATA_PATH)
    _, tfidf_matrix = build_model(df)
    return df, tfidf_matrix

def show_recommendations(df, perfume, tfidf_matrix):
    with st.spinner("🪄 Looking for your perfect fragrance..."):
        recommendations = recommend(df, perfume, tfidf_matrix)

    st.subheader("Recommended Perfumes")
    

    if not recommendations:
        st.info("Nothing similar enough to suggest for this one.")
        return
 
    for rec in recommendations:
        st.html(f"""
        <div style="background-color:#E5EDF0;
                    padding:12px 16px;
                    border-radius:3px;
                    margin-bottom:10px;
                    border-left:3px solid #78B7B5;">
            <div style="font-size:16px; font-weight:700; color:#243B4A;">
                {escape(str(rec['name']))}
            </div>
            <div style="font-size:12px; color:#607783;">
                {escape(str(rec['brand']))}
            </div>
            <div style="font-size:12px; font-weight:600; color:#367E7D; margin-top:6px;">
                Similarity: {rec['similarity']:.2f}%
            </div>
        </div>
        """)
 
        st.progress(min(rec["similarity"] / 100, 1.0))
        st.divider()
        


# Styling and header
st.markdown(STYLE, unsafe_allow_html=True)
 
st.markdown(
    f"""
    <div class="floatImage">
        <img src="data:image/png;base64,{get_base64(IMAGE_PATH)}">
    </div>
    <h1 class="title" style="color:#C9A64B !important;">🪄 Find My Perfume</h1>
    <p class="subtitle">Find your next signature fragrance.</p>
    """,
    unsafe_allow_html=True,
)
 
# Sidebar
st.sidebar.title("🍃 About")
st.sidebar.markdown(
    """
    <div style="color:#F2DFA0;">
        Find perfumes similar to fragrances you already enjoy.
        <h3 style="color:#C9A64B;">Built with</h3>
        <ul>
            <li>Python</li>
            <li>Streamlit</li>
            <li>TF-IDF</li>
            <li>Cosine Similarity</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)
 
df, tfidf_matrix = load_model()
 
# Search form
_, center, _ = st.columns([0.7, 2.6, 0.7])
 
with center:
    with st.form("search_form"):
        user_perfume = st.text_input(
            "🔎 Search for a perfume",
            placeholder="e.g. Dior Sauvage",
        )
        search = st.form_submit_button("Find Similar Perfumes")
 
if search:
    perfume, match_type = find_perfume(df, user_perfume)
 
    if perfume is None:
        st.error("❌ No similar perfume found.")
    else:
        if match_type != "exact":
            st.markdown(
                f"""
                <div style="color:#F2DFA0; font-size:16px; font-weight:600; margin:12px 0;">
                    🧙🏼 Showing results for {escape(perfume)}
                </div>
                """,
                unsafe_allow_html=True,
            )
        show_recommendations(df, perfume, tfidf_matrix)
 

        