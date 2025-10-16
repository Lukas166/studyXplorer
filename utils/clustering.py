import numpy as np
from utils.custom_algorithms import (
    KMeans, AgglomerativeClustering, DBSCAN, StandardScaler,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    linkage as scipy_linkage, dendrogram, fcluster
)

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
    
    # Recommendation UI removed from app; keep code minimal here
    
    def perform_clustering(self, X_reduced, method='kmeans', n_clusters=3, **kwargs):
        """Perform clustering dengan berbagai metode"""
        if method == 'kmeans':
            self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == 'hierarchical':
            # Hierarchical clustering - jumlah cluster ditentukan otomatis dari dendrogram
            linkage_method = kwargs.get('linkage', 'ward')
            
            # Calculate linkage matrix to determine optimal cut height
            Z = scipy_linkage(X_reduced, method=linkage_method)
            
            # Smart threshold: find the largest gap in distances
            # This represents the most significant merge
            distances = Z[:, 2]
            if len(distances) > 1:
                # Calculate gaps between consecutive merges
                gaps = np.diff(distances)
                # Find where the largest gap occurs
                max_gap_idx = np.argmax(gaps)
                # Cut just before the largest gap (i.e., at the distance of max_gap_idx merge)
                distance_threshold = distances[max_gap_idx + 1]
                # If threshold is too small, use a percentage of max distance
                if distance_threshold < 0.1 * Z[-1, 2]:
                    distance_threshold = 0.3 * Z[-1, 2]
            else:
                distance_threshold = 0.5 * Z[-1, 2]
            
            self.model = AgglomerativeClustering(
                n_clusters=None,
                linkage=linkage_method,
                distance_threshold=distance_threshold
            )
        elif method == 'dbscan':
            eps = float(kwargs.get('eps', 0.5))
            min_samples = int(kwargs.get('min_samples', 5))
            self.model = DBSCAN(eps=eps, min_samples=min_samples)
        else:
            self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        # For DBSCAN, re-scale the reduced space to unit variance to stabilize epsilon scale
        if method == 'dbscan':
            X_used = StandardScaler().fit_transform(X_reduced)
        else:
            X_used = X_reduced

        self.labels = self.model.fit_predict(X_used)

        # Calculate clusters and noise
        if method == 'dbscan':
            core_mask = self.labels != -1
            unique_clusters = np.unique(self.labels[core_mask])
            n_clusters_found = len(unique_clusters)
            noise_points = int(np.sum(~core_mask))
        elif method == 'hierarchical':
            unique_clusters = np.unique(self.labels)
            n_clusters_found = len(unique_clusters)
            noise_points = 0
        else:
            unique_clusters = np.unique(self.labels)
            n_clusters_found = len(unique_clusters)
            noise_points = 0

        # Metrics: for DBSCAN, compute on core samples if at least 2 clusters; else set to -1
        try:
            if n_clusters_found > 1:
                if method == 'dbscan':
                    X_eval = X_used[core_mask]
                    labels_eval = self.labels[core_mask]
                else:
                    X_eval = X_used
                    labels_eval = self.labels

                silhouette = silhouette_score(X_eval, labels_eval)
                db_index = davies_bouldin_score(X_eval, labels_eval)
                calinski = calinski_harabasz_score(X_eval, labels_eval)
            else:
                silhouette = db_index = calinski = -1
        except Exception:
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
        if n_clusters and n_clusters > 1 and n_clusters <= len(Z) + 1:
            # Calculate the height to cut for desired number of clusters
            # n total samples → (n-1) merges in Z
            # To get k clusters, cut at height just below the (n-k)th merge from end
            # Example: 100 samples, want 2 clusters → cut before last merge (index -1)
            #          100 samples, want 3 clusters → cut before 2nd-to-last merge (index -2)
            heights = Z[:, 2]
            
            # Index of merge: to get k clusters, we cut between merge (n-k-1) and (n-k)
            # So we take height at index (n-k) which is -(k-1) from the end
            merge_index = len(heights) - n_clusters + 1
            
            if 0 <= merge_index < len(heights):
                # Cut just above this merge height (add small epsilon to be above it)
                cut_height = heights[merge_index] + 0.01 * (heights[-1] - heights[0])
                
                # Add horizontal line across the full width
                x_min = np.min(icoord)
                x_max = np.max(icoord)
                
                fig.add_trace(go.Scatter(
                    x=[x_min, x_max],
                    y=[cut_height, cut_height],
                    mode='lines',
                    line=dict(color='red', width=3, dash='dash'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Add annotation
                fig.add_annotation(
                    x=x_max,
                    y=cut_height,
                    text=f"Cut for {n_clusters} clusters",
                    showarrow=False,
                    xanchor='right',
                    yanchor='bottom',
                    font=dict(color='red', size=12, family='Arial Black'),
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='red',
                    borderwidth=1
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
