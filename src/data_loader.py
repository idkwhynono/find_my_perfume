import pandas as pd
 
 
def load_data(path):
    df = pd.read_csv(path)
 
    df = df.dropna(subset=["name", "notes"]).copy()
 
    df["name"] = df["name"].astype(str).str.strip()
    df["notes"] = df["notes"].astype(str)
 
    df = df.drop_duplicates(subset=["name"], keep="first")
    return df.reset_index(drop=True)
 
 
def print_recommendations(recommendations):
    print("\nRecommended perfumes:\n")
    for rec in recommendations:
        print(f"{rec['name']:<25}{rec['brand']:<15}{rec['similarity']:>6.1f}%")
    print()
 
 
def ask_yes_no(question):
    while True:
        answer = input(question + " (Y/N): ").strip().lower()
        if answer in ("y", "n"):
            return answer == "y"
        print("Please only enter Y or N!")
 
 
def run_cli(path="data/perfumes.csv"):
    from recommender import build_model, find_perfume, recommend
 
    df = load_data(path)
    _, tfidf_matrix = build_model(df)
 
    while True:
        query = input("Enter a perfume (or 'q' to quit): ").strip()
        if query.lower() == "q":
            break
 
        perfume, match_type = find_perfume(df, query)
 
        if perfume is None:
            print("No similar perfume found.\n")
            continue
 
        if match_type != "exact" and not ask_yes_no(f"Did you mean '{perfume}'?"):
            print("Sorry, no matches found.\n")
            continue
 
        print_recommendations(recommend(df, perfume, tfidf_matrix))
 
 
if __name__ == "__main__":
    try:
        run_cli()
    except FileNotFoundError:
        print("Dataset not found. Check that data/perfumes.csv exists.")
 
