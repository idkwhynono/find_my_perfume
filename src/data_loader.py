import numpy as np
import pandas as pd
from recommender import build_model, recommend, find_perfume


def load_data(path):
    df = pd.read_csv(path)

    # Remove rows without a perfume name or notes
    df = df.dropna(subset=["name", "notes"])

    # Make sure both columns are strings
    df["name"] = df["name"].astype(str)
    df["notes"] = df["notes"].astype(str)

    df = df.reset_index(drop=True)

    return df

def preview_data(data, n=5):
    print(data.head(n))

def load_data(path):

    df = pd.read_csv(path)

    df = df.dropna(subset=["notes"])

    df = df.reset_index(drop=True)

    return df


if __name__ == "__main__":
   file_path = "data/perfumes.csv"
   
   try:
        df = load_data(file_path)
        df = df.reset_index(drop=True)
        
        # Build the cosine similarity matrix
        similarity_matrix = build_model(df)
        
        while True:
            user_perfume = input("Enter a perfume: ").strip()

            perfume, matched_tag = find_perfume(df, user_perfume)

            if perfume is None:
                print("No similar perfume found.")
                
            elif matched_tag == "exact":
                recommendations = recommend(df, perfume, similarity_matrix)

                print("\nRecommended perfumes:\n")

                for recom in recommendations:
                    print(
                        f"{recom['name']:<25}"
                        f"{recom['brand']:<15}"
                        f"{recom['similarity']:>6.1f}%"

                    )


            elif matched_tag in ("partial", "fuzzy"):
                while True:
                    answer = input(f"Did you mean '{perfume}'?  (Y/N): ").strip().lower()
                    if answer == "y":
                        recommendations = recommend(df, perfume, similarity_matrix)

                        print("\nRecommended perfumes:\n")

                        for recom in recommendations:
                            print(
                                f"{recom['name']:<25}"
                                f"{recom['brand']:<15}"
                                f"{recom['similarity']:>6.1f}%"

                            )
                        break

                    elif answer == "n":
                        print("Sorry, no matches found.")
                        break

                    else:
                        print("Please only enter Y or N!")

                print()
            
            
        
        
   except FileNotFoundError:
        print("Dataset not found yet. We will add it in next step")
                   

 