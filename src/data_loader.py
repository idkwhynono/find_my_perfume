import numpy as np
import pandas as pd
from recommender import build_model, recommend

def load_data(path):
    data = pd.read_csv(path)
    return pd.read_csv(file_path)

def preview_data(data, n=5):
    print(data.head(n))


if __name__ == "__main__":
   file_path = "data/perfumes.csv"
   try:
        df = load_data(file_path)
        df = df.reset_index(drop=True)

        """
        building cosine similarity

        """
        simi = build_model(df)
        print(recommend(df, "Dior Sauvage", simi))

   except FileNotFoundError:
        print("Dataset not found yet. We will add it in next step")
                   

 