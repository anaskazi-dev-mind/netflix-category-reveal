import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def find_optimal_clusters(tfidf_matrix, max_k=30):
    """
    Runs KMeans for k=2 to max_k and plots the Elbow Curve.
    """
    print(f"Calculating optimal clusters (testing 2 to {max_k})...")
    
    inertias = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        # We use a lower n_init here to speed up the search
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
        kmeans.fit(tfidf_matrix)
        inertias.append(kmeans.inertia_)
        
        # specific print to track progress
        if k % 5 == 0:
            print(f"... Tested k={k}")

    # Plotting the Elbow Curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, marker='o', linestyle='--', color='b')
    plt.title('Elbow Method: Finding the Optimal Number of Categories')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (Error)')
    plt.grid(True)
    
    output_path = 'elbow_curve.png'
    plt.savefig(output_path)
    print(f"\nOptimization complete! Check '{output_path}' to find the elbow point.")