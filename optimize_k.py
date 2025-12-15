import pandas as pd
from src.preprocessing import TextPreprocessor
from src.clustering import NetflixClusterer
from src.optimization import find_optimal_clusters

def main():
    # 1. Load Data
    print("--- Loading Data for Optimization ---")
    try:
        df = pd.read_csv('data/netflix_dataset.csv')
        # IMPORTANT: If using the full dataset (8000 rows), this might take a minute.
        # If it's too slow, sample it: df = df.sample(2000, random_state=42)
        print(f"Loaded {len(df)} titles.")
    except FileNotFoundError:
        print("Error: Dataset not found.")
        return

    # 2. Preprocess
    print("--- Cleaning Text ---")
    preprocessor = TextPreprocessor()
    clean_text = df['description'].apply(preprocessor.clean_text)

    # 3. Vectorize (We need the numbers, not the clusters yet)
    print("--- Vectorizing ---")
    # We initialize the clusterer just to use its vectorizer
    clusterer = NetflixClusterer() 
    tfidf_matrix = clusterer.vectorizer.fit_transform(clean_text)

    # 4. Find Optimal K
    find_optimal_clusters(tfidf_matrix, max_k=30)

if __name__ == "__main__":
    main()