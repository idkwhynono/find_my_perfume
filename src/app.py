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

        st.markdown(f"""
        <div style="
            background:white;
            padding:20px;
            border-radius:18px;
            margin-bottom:18px;
            box-shadow:0px 4px 12px rgba(0,0,0,.08);
            border-left:7px solid #F7B38A;
        ">

        <h3 style="color:#D26A8C;">
            {recommendation['name']}
        </h3>

        <b>Brand:</b> {recommendation['brand']}<br><br>

        <b style="color:#9CCFEA;">
            Similarity:
            {recommendation['similarity']:.1f}%
            
        </b>

        </div>
        """, unsafe_allow_html=True)

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

st.sidebar.info("""
Find perfumes similar to fragrances you already enjoy.

### Built with

- Python
- Streamlit
- TF-IDF
- Cosine Similarity
""")

# Load data
df = load_data("data/perfumes.csv")
df = df.reset_index(drop=True)

similarity_matrix = build_model(df)

# CSS
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"]{
    font-family:'Poppins',sans-serif;
}

@keyframes fadeIn{
    from{
        opacity:0;
        transform:translateY(-25px);
    }

    to{
        opacity:1;
        transform:translateY(0px);
    }
}

.title{
    font-size:60px;
    font-weight:700;
    text-align:center;
    color:#D26A8C;
    animation:fadeIn 1.2s ease;
    margin-bottom:0px;
}

.subtitle{
    text-align:center;
    color:#8A7A7A;
    font-size:18px;
    margin-bottom:10px;
}

.bottle{
    text-align:center;
    font-size:75px;
    animation:fadeIn 2s ease;
    margin-bottom:20px;
}

div.stButton > button{

    width:100%;

    background:#F7B38A;

    color:white;

    border:none;

    border-radius:15px;

    padding:12px;

    font-size:18px;

    transition:0.3s;
}

div.stButton > button:hover{

    background:#D97B93;

    transform:scale(1.03);

}

</style>

<h1 class="title">🪄 Find My Perfume</h1>

<p class="subtitle">
Find your next signature fragrance.
</p>

""", unsafe_allow_html=True)

left, center, right = st.columns([1, 2, 1])

with center:

    with st.form("search_form"):

        user_perfume = st.text_input(
            "🔎 Search for a perfume",
            placeholder="e.g. Dior Sauvage"
        )

        search = st.form_submit_button("Find Similar")




if search:

    perfume, matched_tag = find_perfume(df, user_perfume)

    if perfume is None:

        st.error("❌ No similar perfume found.")

    elif matched_tag == "exact":

        show_recommendations(df, perfume, similarity_matrix)

    elif matched_tag in ("partial", "fuzzy"):

        st.info(f"🧙🏼 Showing results for **{perfume}**")

        show_recommendations(df, perfume, similarity_matrix)

   

        

        