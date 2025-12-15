import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

class NetflixClusterer:
    def __init__(self, n_clusters=20):
        self.n_clusters = n_clusters
        # TF-IDF Configuration
        self.vectorizer = TfidfVectorizer(
            max_features=5000, 
            stop_words='english',
            ngram_range=(1, 2) # Use single words and pairs (e.g., "high school")
        )
        # K-Means Configuration
        self.kmeans = KMeans(
            n_clusters=self.n_clusters, 
            random_state=42, 
            n_init=10
        )
        self.model = None
        self.feature_names = None

    def create_clusters(self, text_data):
        """
        Takes a list/series of text and returns cluster labels.
        """
        print("Vectorizing text data...")
        tfidf_matrix = self.vectorizer.fit_transform(text_data)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        print(f"Clustering into {self.n_clusters} categories...")
        self.kmeans.fit(tfidf_matrix)
        
        return self.kmeans.labels_

    def get_cluster_keywords(self, cluster_id, top_n=10):
        """
        Returns the top keywords for a specific cluster.
        """
        # Get the center of the cluster (average of all docs in it)
        centroid = self.kmeans.cluster_centers_[cluster_id]
        
        # Sort indices to find top terms
        top_indices = centroid.argsort()[::-1][:top_n]
        return [self.feature_names[i] for i in top_indices]