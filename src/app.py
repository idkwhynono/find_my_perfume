from PIL import Image
import streamlit as st
from data_loader import load_data
from recommender import build_model, find_perfume, recommend
import base64



def get_base64(path):
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode()
    
def show_recommendations(df, perfume, similarity_matrix):

    with st.spinner("🪄 Looking for your perfect fragrance..."):
        recommendations = recommend(df, perfume, similarity_matrix)

    recommendations = [
        recommendation
        for recommendation in recommendations
        if recommendation["similarity"] > 0
    ]

    st.subheader("Recommended Perfumes")
    

    for recommendation in recommendations:

        card_html = f"""
        <div style="background-color:#E5EDF0;
                    padding:12px 16px;
                    border-radius:3px;
                    margin-bottom:10px;
                    border-left:3px solid #78B7B5;">

            <div style="font-size:16px; font-weight:700; color:#243B4A;">
                {recommendation['name']}
            </div>

            <div style="font-size:12px; color:#607783;">
                {recommendation['brand']}
            </div>

            <div style="font-size:12px; font-weight:600; color:#367E7D; margin-top:6px;">
                Similarity: {recommendation['similarity']:.2f}%
            </div>

        </div>
        """

        st.html(card_html)

        st.progress(recommendation["similarity"] / 100)

        st.divider()

img = get_base64("assets/perfume_bottle.png")

st.markdown(f"""
<style>

@keyframes float {{

0% {{
transform: translateY(0px);
}}

50% {{
transform: translateY(-10px);
}}

100% {{
transform: translateY(0px);
}}

}}

.floatImage {{

display:flex;

justify-content:center;

margin-bottom:20px;

}}

.floatImage img {{

width:170px;

animation:float 4s ease-in-out infinite;

}}

</style>

<div class="floatImage">

<img src="data:image/png;base64,{img}">

</div>

""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🍃 About")

left, center, right = st.columns([1.5,1,1.5])

st.sidebar.markdown("""
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
""", unsafe_allow_html=True)

# Load data
df = load_data("data/perfumes.csv")
df = df.reset_index(drop=True)

vectorizer, tfidf_matrix = build_model(df)

# CSS
st.markdown("""
    <style>
    /* Main background */
    .stApp,
    [data-testid="stAppViewContainer"] {
        background-color: #243B4A !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1B303D !important;
        color: #F2DFA0;
    }

    /* Header */
    [data-testid="stHeader"] {
        background-color: #243B4A !important;
    }

    /* Main title */
    .title {
        font-size: 56px;
        font-weight: 800;
        text-align: center;
        color: #C9A64B !important;
        margin-bottom: 0;
}

    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #A9BEC8;
        font-size: 17px;
    }

    /* Search button */
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
            
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"]{
    font-family:'Poppins',sans-serif;
}

# Search input

div[data-baseweb="input"] input {

    border-radius: 8px;

}
# Input focus

div[data-baseweb="input"]:focus-within {

    border-color: #426B8A;

}      

</style>

<h1 class="title">🪄 Find My Perfume</h1>

<p class="subtitle">
Find your next signature fragrance.
</p>

""", unsafe_allow_html=True)

left, center, right = st.columns([0.7, 2.6, 0.7])

with center:

    with st.form("search_form"):

        user_perfume = st.text_input(
            "🔎 Search for a perfume",
            placeholder="e.g. Dior Sauvage"
        )

        search = st.form_submit_button("Find Similar Perfumes")


if search:

    perfume, matched_tag = find_perfume(df, user_perfume)

    if perfume is None:

        st.error("❌ No similar perfume found.")

    elif matched_tag == "exact":

        show_recommendations(df, perfume, tfidf_matrix)

    elif matched_tag in ("partial", "fuzzy"):

        st.markdown(
            f"""
            <div style="
                color: #F2DFA0;
                font-size: 16px;
                font-weight: 600;
                margin: 12px 0;
            ">
                🧙🏼 Showing results for {perfume}
            </div>
            """,
            unsafe_allow_html=True
)

        show_recommendations(df, perfume, tfidf_matrix)

   
print(df.loc[df["name"] == "Bleu de Chanel All-Over Spray", "notes"].iloc[0])

print(df.loc[df["name"] == "De Blue", "notes"].iloc[0])

print(df.loc[df["name"] == "Cool Blue", "notes"].iloc[0])

print(df.loc[df["name"] == "Indigo Vetiver", "notes"].iloc[0])

print(df.loc[df["name"] == "Dusk", "notes"].iloc[0])
        

        