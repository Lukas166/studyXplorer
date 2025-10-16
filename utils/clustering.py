import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

class ClusteringEngine:
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
    
    def recommend_dbscan_eps(self, X_reduced, min_samples=5):
        """Rekomendasikan nilai eps optimal untuk DBSCAN menggunakan k-distance graph"""
        from sklearn.neighbors import NearestNeighbors
        
        # Hitung k-nearest neighbors (k = min_samples)
        neighbors = NearestNeighbors(n_neighbors=min_samples)
        neighbors.fit(X_reduced)
        distances, indices = neighbors.kneighbors(X_reduced)
        
        # Ambil distance ke neighbor terjauh (min_samples-th neighbor)
        k_distances = np.sort(distances[:, -1])
        
        # Cari "elbow point" di k-distance graph
        # Gunakan metode yang sama seperti find_elbow_point
        n = len(k_distances)
        first_point = np.array([0, k_distances[0]])
        last_point = np.array([n-1, k_distances[-1]])
        
        max_distance = 0
        elbow_index = 0
        for i in range(n):
            point = np.array([i, k_distances[i]])
            distance = np.abs(np.cross(last_point-first_point, first_point-point)) / np.linalg.norm(last_point-first_point)
            if distance > max_distance:
                max_distance = distance
                elbow_index = i
        
        recommended_eps = k_distances[elbow_index]
        
        # Return recommended eps dan statistik
        return {
            'recommended_eps': float(recommended_eps),
            'min_eps': float(np.percentile(k_distances, 5)),  # 5th percentile
            'max_eps': float(np.percentile(k_distances, 95)),  # 95th percentile
            'median_eps': float(np.median(k_distances)),
            'k_distances': k_distances
        }
    
    def perform_clustering(self, X_reduced, method='kmeans', n_clusters=3, **kwargs):
        """Perform clustering dengan berbagai metode"""
        if method == 'kmeans':
            self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == 'hierarchical':
            # Hierarchical clustering - jumlah cluster ditentukan otomatis dari dendrogram
            from scipy.cluster.hierarchy import linkage as scipy_linkage
            linkage = kwargs.get('linkage', 'ward')
            
            # Calculate linkage matrix to determine optimal cut height
            Z = scipy_linkage(X_reduced, method=linkage)
            
            # Use automatic distance threshold to determine number of clusters
            # Using 70% of max distance as threshold (heuristic)
            distance_threshold = 0.7 * Z[-1, 2]
            
            self.model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                linkage=linkage
            )
        elif method == 'dbscan':
            eps = kwargs.get('eps', 0.5)
            min_samples = kwargs.get('min_samples', 5)
            self.model = DBSCAN(eps=eps, min_samples=min_samples)
        else:
            self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        self.labels = self.model.fit_predict(X_reduced)
        
        # Calculate metrics
        if method == 'hierarchical':
            n_clusters_found = len(np.unique(self.labels))
        else:
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
    
    def create_dendrogram(self, X_reduced, linkage='ward', max_samples=1000, n_clusters=None):
        """Create dendrogram for hierarchical clustering visualization with cut line"""
        from scipy.cluster.hierarchy import dendrogram, linkage as scipy_linkage, fcluster
        import plotly.graph_objects as go
        
        # Limit samples for performance
        if len(X_reduced) > max_samples:
            indices = np.random.choice(len(X_reduced), max_samples, replace=False)
            X_sample = X_reduced[indices]
        else:
            X_sample = X_reduced
        
        # Perform hierarchical clustering
        Z = scipy_linkage(X_sample, method=linkage)
        
        # Create dendrogram
        dend = dendrogram(Z, no_plot=True)
        
        # Create plotly figure
        fig = go.Figure()
        
        # Add dendrogram lines
        icoord = np.array(dend['icoord'])
        dcoord = np.array(dend['dcoord'])
        
        for i in range(len(icoord)):
            fig.add_trace(go.Scatter(
                x=icoord[i],
                y=dcoord[i],
                mode='lines',
                line=dict(color='#000000', width=1.5),
                hoverinfo='skip',
                showlegend=False
            ))
        
        # Add horizontal cut line if n_clusters is specified
        if n_clusters and n_clusters > 1:
            # Calculate the height to cut for desired number of clusters
            # Get the last n_clusters-1 merges
            heights = Z[:, 2]
            cut_height = heights[-(n_clusters-1)]
            
            # Add horizontal line
            fig.add_hline(
                y=cut_height,
                line_dash="dash",
                line_color="red",
                line_width=2,
                annotation_text=f"Cut for {n_clusters} clusters",
                annotation_position="right",
                annotation_font_color="red"
            )
        
        fig.update_layout(
            title={
                'text': 'Dendrogram - Hierarchical Clustering',
                'font': {'size': 20, 'color': '#000000', 'family': 'Inter'}
            },
            xaxis={
                'title': 'Sample Index',
                'showgrid': False,
                'zeroline': False,
                'showticklabels': False,
                'color': '#000000'
            },
            yaxis={
                'title': 'Distance',
                'showgrid': True,
                'gridcolor': '#e0e0e0',
                'color': '#000000'
            },
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            font={'color': '#000000'},
            height=500,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig, Z
