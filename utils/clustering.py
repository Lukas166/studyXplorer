"""
Clustering Module - Clustering algorithms and evaluation
"""

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


class ClusteringEngine:
    """Handle clustering operations and evaluation"""
    
    def __init__(self):
        self.model = None
        self.labels = None
        self.clustering_results = None
        
    def find_optimal_k(self, X_reduced, max_k=10):
        """Cari nilai k optimal untuk KMeans"""
        wcss = []
        silhouette_scores = []
        db_scores = []
        
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_reduced)
            
            wcss.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_reduced, labels))
            db_scores.append(davies_bouldin_score(X_reduced, labels))
        
        # Find optimal k based on silhouette score
        optimal_k_silhouette = np.argmax(silhouette_scores) + 2
        optimal_k_elbow = self._find_elbow_point(wcss) + 2
        
        k_analysis = {
            'k_range': list(range(2, max_k + 1)),
            'wcss': wcss,
            'silhouette_scores': silhouette_scores,
            'db_scores': db_scores,
            'optimal_k_silhouette': optimal_k_silhouette,
            'optimal_k_elbow': optimal_k_elbow
        }
        
        return k_analysis
    
    def _find_elbow_point(self, wcss):
        """Find elbow point in WCSS curve"""
        n = len(wcss)
        if n < 3:
            return 0
        
        # Calculate distances from line connecting first and last point
        first_point = np.array([0, wcss[0]])
        last_point = np.array([n-1, wcss[-1]])
        
        distances = []
        for i in range(n):
            point = np.array([i, wcss[i]])
            distance = np.abs(np.cross(last_point-first_point, first_point-point)) / np.linalg.norm(last_point-first_point)
            distances.append(distance)
        
        return np.argmax(distances)
    
    def perform_clustering(self, X_reduced, method='kmeans', n_clusters=3, **kwargs):
        """Perform clustering dengan berbagai metode"""
        if method == 'kmeans':
            self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == 'hierarchical':
            self.model = AgglomerativeClustering(n_clusters=n_clusters)
        elif method == 'dbscan':
            eps = kwargs.get('eps', 0.5)
            min_samples = kwargs.get('min_samples', 5)
            self.model = DBSCAN(eps=eps, min_samples=min_samples)
        else:
            self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        self.labels = self.model.fit_predict(X_reduced)
        
        # Calculate metrics
        n_clusters_found = len(np.unique(self.labels[self.labels != -1])) if method == 'dbscan' else n_clusters
        noise_points = np.sum(self.labels == -1) if method == 'dbscan' else 0
        
        if n_clusters_found > 1:
            silhouette = silhouette_score(X_reduced, self.labels)
            db_index = davies_bouldin_score(X_reduced, self.labels)
            calinski = calinski_harabasz_score(X_reduced, self.labels)
        else:
            silhouette = db_index = calinski = -1
        
        self.clustering_results = {
            'method': method,
            'labels': self.labels,
            'n_clusters': n_clusters_found,
            'noise_points': noise_points,
            'silhouette_score': silhouette,
            'davies_bouldin_score': db_index,
            'calinski_harabasz_score': calinski,
            'model': self.model
        }
        
        return self.clustering_results
