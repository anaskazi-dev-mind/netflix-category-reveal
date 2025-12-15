import pandas as pd
from src.preprocessing import TextPreprocessor
from src.clustering import NetflixClusterer
from src.visualization import plot_clusters

def main():
    # 1. Load Data
    print("--- Loading Data ---")
    try:
        df = pd.read_csv('data/netflix_dataset.csv')
        print(f"Loaded {len(df)} titles.")
    except FileNotFoundError:
        print("Error: 'netflix_dataset.csv' not found in 'data/' folder.")
        return

    # 2. Preprocess Descriptions
    print("\n--- Cleaning Text ---")
    preprocessor = TextPreprocessor()
    # Create a new column for clean text
    df['clean_description'] = df['description'].apply(preprocessor.clean_text)
    
    # 3. Apply Clustering
    print("\n--- Running AI Clustering ---")
    # You can change n_clusters to find more/fewer categories
    clusterer = NetflixClusterer(n_clusters=15) 
    df['cluster_id'] = clusterer.create_clusters(df['clean_description'])

    # 4. Reveal Categories (Analysis)
    print("\n--- REVEALED CATEGORIES ---")
    
    # Loop through each cluster to show results
    for i in range(clusterer.n_clusters):
        print(f"\n[Category {i}]")
        
        # Get top keywords
        keywords = clusterer.get_cluster_keywords(i)
        print(f"Keywords: {', '.join(keywords)}")
        
        # Get 3 example titles from this cluster
        examples = df[df['cluster_id'] == i]['title'].head(3).tolist()
        print(f"Examples: {examples}")

    # 5. Visualize Results (New Step)
    print("\n--- Visualizing ---")
    # We re-transform the text to get the matrix for plotting
    matrix = clusterer.vectorizer.transform(df['clean_description'])
    
    try:
        plot_clusters(matrix, df['cluster_id'], save_path='cluster_plot.png')
    except Exception as e:
        print(f"Visualization failed: {e}")
        print("Tip: Make sure you installed matplotlib and seaborn.")

    # 6. Save CSV Results
    output_file = 'netflix_clusters_revealed.csv'
    df[['title', 'cluster_id', 'description']].to_csv(output_file, index=False)
    print(f"\nSuccess! Full results saved to '{output_file}'")

if __name__ == "__main__":
    main()