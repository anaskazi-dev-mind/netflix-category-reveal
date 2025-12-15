import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def plot_clusters(tfidf_matrix, cluster_labels, save_path='cluster_plot.png'):
    """
    Reduces the data to 2 dimensions using PCA and plots the clusters.
    """
    print("Generating visualization...")
    
    # 1. Reduce dimensions to 2D
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(tfidf_matrix.toarray())
    
    # 2. Setup the plot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x=pca_result[:, 0], 
        y=pca_result[:, 1], 
        hue=cluster_labels, 
        palette='tab20', # Colorful palette
        s=100,           # Dot size
        alpha=0.8        # Transparency
    )
    
    plt.title('Netflix Movie Clusters (Visualized via PCA)', fontsize=16)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Cluster ID")
    
    # 3. Save the image
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to '{save_path}'")