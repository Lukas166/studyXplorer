# Semua code pre-process dan clustering

import numpy as np

# PREPROCESSING

class StandardScaler:
    """Standardize features by removing mean and scaling to unit variance"""
    
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.n_features_in_ = None
    
    def fit(self, X):
        """Compute mean and std for later scaling"""
        X = np.array(X, dtype=np.float64)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        # Avoid division by zero
        self.scale_[self.scale_ == 0] = 1.0
        self.n_features_in_ = X.shape[1]
        return self
    
    def transform(self, X):
        """Standardize features"""
        X = np.array(X, dtype=np.float64)
        return (X - self.mean_) / self.scale_
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        """Reverse the standardization"""
        X = np.array(X, dtype=np.float64)
        return X * self.scale_ + self.mean_


class MinMaxScaler:
    """Scale features to a given range (default [0, 1])"""
    
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.scale_ = None
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None
        self.n_features_in_ = None
    
    def fit(self, X):
        """Compute min and max for later scaling"""
        X = np.array(X, dtype=np.float64)
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        # Avoid division by zero
        self.data_range_[self.data_range_ == 0] = 1.0
        
        feature_min, feature_max = self.feature_range
        self.scale_ = (feature_max - feature_min) / self.data_range_
        self.min_ = feature_min - self.data_min_ * self.scale_
        self.n_features_in_ = X.shape[1]
        return self
    
    def transform(self, X):
        """Scale features to range"""
        X = np.array(X, dtype=np.float64)
        X_scaled = X * self.scale_ + self.min_
        return X_scaled
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        """Reverse the scaling"""
        X = np.array(X, dtype=np.float64)
        return (X - self.min_) / self.scale_


class RobustScaler:
    """Scale features using statistics that are robust to outliers (median, IQR)"""
    
    def __init__(self, quantile_range=(25.0, 75.0)):
        self.quantile_range = quantile_range
        self.center_ = None
        self.scale_ = None
        self.n_features_in_ = None
    
    def fit(self, X):
        """Compute median and IQR for later scaling"""
        X = np.array(X, dtype=np.float64)
        self.center_ = np.median(X, axis=0)
        
        q_min, q_max = self.quantile_range
        quantiles = np.percentile(X, [q_min, q_max], axis=0)
        self.scale_ = quantiles[1] - quantiles[0]
        # Avoid division by zero
        self.scale_[self.scale_ == 0] = 1.0
        self.n_features_in_ = X.shape[1]
        return self
    
    def transform(self, X):
        """Scale features using median and IQR"""
        X = np.array(X, dtype=np.float64)
        return (X - self.center_) / self.scale_
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        """Reverse the scaling"""
        X = np.array(X, dtype=np.float64)
        return X * self.scale_ + self.center_


# DIMENSIONALITY REDUCTION

class PCA:
    """Principal Component Analysis"""
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.n_features_ = None
        self.n_samples_ = None
    
    def fit(self, X):
        """Fit the PCA model"""
        X = np.array(X, dtype=np.float64)
        self.n_samples_, self.n_features_ = X.shape
        
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select components
        if self.n_components is None:
            self.n_components = self.n_features_
        
        # Store explained variance (only for selected components)
        total_var = np.sum(eigenvalues)
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / total_var
        
        self.components_ = eigenvectors[:, :self.n_components].T
        
        return self
    
    def transform(self, X):
        """Apply dimensionality reduction"""
        X = np.array(X, dtype=np.float64)
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        """Transform back to original space"""
        X = np.array(X, dtype=np.float64)
        return np.dot(X, self.components_) + self.mean_


# CLUSTERING ALGORITHMS

class KMeans:
    """K-Means Clustering Algorithm"""
    
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None, n_init=10):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_init = n_init
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None
    
    def _initialize_centroids(self, X, random_state):
        """Initialize centroids using k-means++"""
        n_samples = X.shape[0]
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        
        # Choose first centroid randomly
        centroids[0] = X[random_state.choice(n_samples)]
        
        # Choose remaining centroids with k-means++ method
        for i in range(1, self.n_clusters):
            # Calculate distances to nearest centroid
            distances = np.min([np.sum((X - c) ** 2, axis=1) for c in centroids[:i]], axis=0)
            # Choose next centroid with probability proportional to distance squared
            probabilities = distances / distances.sum()
            cumulative_probs = probabilities.cumsum()
            r = random_state.rand()
            for j, p in enumerate(cumulative_probs):
                if r < p:
                    centroids[i] = X[j]
                    break
        
        return centroids
    
    def _assign_clusters(self, X, centroids):
        """Assign each point to nearest centroid"""
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
    
    def _update_centroids(self, X, labels):
        """Update centroids as mean of assigned points"""
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                centroids[i] = cluster_points.mean(axis=0)
            else:
                # If cluster is empty, reinitialize randomly
                centroids[i] = X[np.random.choice(len(X))]
        return centroids
    
    def _compute_inertia(self, X, labels, centroids):
        """Compute sum of squared distances to closest centroid"""
        inertia = 0
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centroids[i]) ** 2)
        return inertia
    
    def fit(self, X):
        """Fit K-Means model"""
        X = np.array(X, dtype=np.float64)
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        best_inertia = np.inf
        best_centroids = None
        best_labels = None
        best_n_iter = 0
        
        # Run k-means multiple times and keep best result
        for _ in range(self.n_init):
            random_state = np.random.RandomState(self.random_state)
            centroids = self._initialize_centroids(X, random_state)
            
            for iteration in range(self.max_iter):
                # Assign clusters
                labels = self._assign_clusters(X, centroids)
                
                # Update centroids
                new_centroids = self._update_centroids(X, labels)
                
                # Check convergence
                centroid_shift = np.sum((new_centroids - centroids) ** 2)
                centroids = new_centroids
                
                if centroid_shift < self.tol:
                    break
            
            # Compute inertia
            inertia = self._compute_inertia(X, labels, centroids)
            
            # Keep best result
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
                best_n_iter = iteration + 1
        
        self.cluster_centers_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        
        return self
    
    def predict(self, X):
        """Predict cluster for new data"""
        X = np.array(X, dtype=np.float64)
        return self._assign_clusters(X, self.cluster_centers_)
    
    def fit_predict(self, X):
        """Fit and predict in one step"""
        self.fit(X)
        return self.labels_


class DBSCAN:
    """Density-Based Spatial Clustering of Applications with Noise"""
    
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        self.core_sample_indices_ = None
    
    def _get_neighbors(self, X, point_idx):
        """Find all neighbors within eps distance"""
        distances = np.sqrt(np.sum((X - X[point_idx]) ** 2, axis=1))
        return np.where(distances <= self.eps)[0]
    
    def fit(self, X):
        """Fit DBSCAN model"""
        X = np.array(X, dtype=np.float64)
        n_samples = len(X)
        
        # Initialize all points as noise (-1)
        self.labels_ = np.full(n_samples, -1, dtype=int)
        
        # Track visited points
        visited = np.zeros(n_samples, dtype=bool)
        
        # Track core points
        core_samples = []
        
        cluster_id = 0
        
        for point_idx in range(n_samples):
            if visited[point_idx]:
                continue
            
            visited[point_idx] = True
            neighbors = self._get_neighbors(X, point_idx)
            
            # Check if core point
            if len(neighbors) < self.min_samples:
                # Mark as noise (might be changed to border later)
                continue
            
            # Core point found - start new cluster
            core_samples.append(point_idx)
            self.labels_[point_idx] = cluster_id
            
            # Expand cluster
            seed_set = list(neighbors)
            i = 0
            while i < len(seed_set):
                neighbor_idx = seed_set[i]
                
                if not visited[neighbor_idx]:
                    visited[neighbor_idx] = True
                    neighbor_neighbors = self._get_neighbors(X, neighbor_idx)
                    
                    # If neighbor is also core point, add its neighbors
                    if len(neighbor_neighbors) >= self.min_samples:
                        core_samples.append(neighbor_idx)
                        seed_set.extend([n for n in neighbor_neighbors if n not in seed_set])
                
                # Assign to cluster if not already assigned
                if self.labels_[neighbor_idx] == -1:
                    self.labels_[neighbor_idx] = cluster_id
                
                i += 1
            
            cluster_id += 1
        
        self.core_sample_indices_ = np.array(core_samples, dtype=int)
        return self
    
    def fit_predict(self, X):
        """Fit and predict in one step"""
        self.fit(X)
        return self.labels_


class AgglomerativeClustering:
    """Agglomerative Hierarchical Clustering - Full from scratch implementation"""
    
    def __init__(self, n_clusters=2, linkage='ward', distance_threshold=None):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.distance_threshold = distance_threshold
        self.labels_ = None
        self.n_clusters_ = None
    
    def _compute_linkage_distance(self, X, cluster_i, cluster_j):
        """Compute distance between two clusters based on linkage method"""
        if self.linkage == 'single':
            # Single linkage: minimum distance between any pair of points
            min_dist = np.inf
            for p1 in cluster_i:
                for p2 in cluster_j:
                    dist = np.sqrt(np.sum((X[p1] - X[p2]) ** 2))
                    if dist < min_dist:
                        min_dist = dist
            return min_dist
        
        elif self.linkage == 'complete':
            # Complete linkage: maximum distance between any pair of points
            max_dist = 0.0
            for p1 in cluster_i:
                for p2 in cluster_j:
                    dist = np.sqrt(np.sum((X[p1] - X[p2]) ** 2))
                    if dist > max_dist:
                        max_dist = dist
            return max_dist
        
        elif self.linkage == 'average':
            # Average linkage: mean distance between all pairs
            total_dist = 0.0
            count = 0
            for p1 in cluster_i:
                for p2 in cluster_j:
                    dist = np.sqrt(np.sum((X[p1] - X[p2]) ** 2))
                    total_dist += dist
                    count += 1
            return total_dist / count if count > 0 else 0.0
        
        else:  # ward
            # Ward's method: minimum variance criterion
            # Distance = sqrt(2 * n_i * n_j / (n_i + n_j)) * ||center_i - center_j||
            n_i = len(cluster_i)
            n_j = len(cluster_j)
            
            # Compute centroids
            center_i = np.mean([X[p] for p in cluster_i], axis=0)
            center_j = np.mean([X[p] for p in cluster_j], axis=0)
            
            # Squared Euclidean distance between centroids
            squared_dist = np.sum((center_i - center_j) ** 2)
            
            # Ward's distance formula
            ward_dist = np.sqrt(2.0 * n_i * n_j / (n_i + n_j) * squared_dist)
            
            return ward_dist
    
    def fit(self, X):
        """Fit Agglomerative Clustering model"""
        X = np.array(X, dtype=np.float64)
        n_samples = len(X)
        
        # Initialize: each point is its own cluster
        # Use dictionary to map cluster_id -> list of point indices
        clusters = {i: [i] for i in range(n_samples)}
        
        # Initialize distance matrix between clusters using cluster IDs
        distances = {}
        
        # Compute initial pairwise distances
        cluster_ids = list(clusters.keys())
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                id_i, id_j = cluster_ids[i], cluster_ids[j]
                dist = self._compute_linkage_distance(X, clusters[id_i], clusters[id_j])
                distances[(id_i, id_j)] = dist
        
        # Merge clusters iteratively
        next_cluster_id = n_samples
        
        while len(clusters) > 1:
            # Check stopping criteria
            if self.distance_threshold is None:
                if len(clusters) <= self.n_clusters:
                    break
            else:
                # For hierarchical with distance_threshold, ensure minimum 2 clusters
                if len(clusters) <= 2:
                    break
                # Find minimum distance
                if not distances:
                    break
                min_dist = min(distances.values())
                if min_dist > self.distance_threshold:
                    break
            
            # Find the pair of clusters with minimum distance
            min_pair = min(distances.items(), key=lambda x: x[1])
            (cluster_i, cluster_j), min_dist = min_pair
            
            # Merge cluster_i and cluster_j into a new cluster
            merged_cluster = clusters[cluster_i] + clusters[cluster_j]
            
            # Remove old distance entries involving cluster_i or cluster_j
            keys_to_remove = []
            for key in distances.keys():
                if cluster_i in key or cluster_j in key:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del distances[key]
            
            # Remove the two old clusters
            del clusters[cluster_i]
            del clusters[cluster_j]
            
            # Add the new merged cluster
            clusters[next_cluster_id] = merged_cluster
            
            # Compute distances from new cluster to all remaining clusters
            for other_id in clusters.keys():
                if other_id == next_cluster_id:
                    continue
                dist = self._compute_linkage_distance(X, clusters[next_cluster_id], clusters[other_id])
                # Always store with smaller ID first for consistency
                if next_cluster_id < other_id:
                    distances[(next_cluster_id, other_id)] = dist
                else:
                    distances[(other_id, next_cluster_id)] = dist
            
            next_cluster_id += 1
        
        # Assign final labels
        self.labels_ = np.zeros(n_samples, dtype=int)
        for cluster_id, (cluster_key, cluster_points) in enumerate(clusters.items()):
            for point_idx in cluster_points:
                self.labels_[point_idx] = cluster_id
        
        self.n_clusters_ = len(clusters)
        
        return self
    
    def fit_predict(self, X):
        """Fit and predict in one step"""
        self.fit(X)
        return self.labels_


# METRICS

def silhouette_score(X, labels):
    """Compute the mean Silhouette Coefficient of all samples"""
    X = np.array(X, dtype=np.float64)
    labels = np.array(labels)
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters == 1 or n_clusters >= len(X):
        return 0.0
    
    silhouette_scores = []
    
    for i in range(len(X)):
        # Get label of current point
        label_i = labels[i]
        
        # Compute a(i): mean distance to points in same cluster
        same_cluster_mask = (labels == label_i)
        same_cluster_points = X[same_cluster_mask]
        
        if len(same_cluster_points) > 1:
            a_i = np.mean([np.sqrt(np.sum((X[i] - p) ** 2)) 
                          for p in same_cluster_points if not np.array_equal(p, X[i])])
        else:
            a_i = 0.0
        
        # Compute b(i): mean distance to points in nearest cluster
        b_i = np.inf
        for label_j in unique_labels:
            if label_j == label_i:
                continue
            
            other_cluster_mask = (labels == label_j)
            other_cluster_points = X[other_cluster_mask]
            
            if len(other_cluster_points) > 0:
                mean_dist = np.mean([np.sqrt(np.sum((X[i] - p) ** 2)) 
                                    for p in other_cluster_points])
                b_i = min(b_i, mean_dist)
        
        # Compute silhouette coefficient
        if max(a_i, b_i) > 0:
            s_i = (b_i - a_i) / max(a_i, b_i)
        else:
            s_i = 0.0
        
        silhouette_scores.append(s_i)
    
    return np.mean(silhouette_scores)


def davies_bouldin_score(X, labels):
    """Compute the Davies-Bouldin score (lower is better)"""
    X = np.array(X, dtype=np.float64)
    labels = np.array(labels)
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters == 1:
        return 0.0
    
    # Compute centroids
    centroids = np.array([X[labels == label].mean(axis=0) for label in unique_labels])
    
    # Compute average distances within clusters
    avg_distances = []
    for label in unique_labels:
        cluster_points = X[labels == label]
        centroid = centroids[label]
        avg_dist = np.mean([np.sqrt(np.sum((p - centroid) ** 2)) for p in cluster_points])
        avg_distances.append(avg_dist)
    
    # Compute Davies-Bouldin index
    db_scores = []
    for i in range(n_clusters):
        max_ratio = 0
        for j in range(n_clusters):
            if i != j:
                centroid_dist = np.sqrt(np.sum((centroids[i] - centroids[j]) ** 2))
                if centroid_dist > 0:
                    ratio = (avg_distances[i] + avg_distances[j]) / centroid_dist
                    max_ratio = max(max_ratio, ratio)
        db_scores.append(max_ratio)
    
    return np.mean(db_scores)


def calinski_harabasz_score(X, labels):
    """Compute the Calinski-Harabasz score (Variance Ratio Criterion, higher is better)"""
    X = np.array(X, dtype=np.float64)
    labels = np.array(labels)
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    n_samples = len(X)
    
    if n_clusters == 1 or n_clusters >= n_samples:
        return 0.0
    
    # Overall centroid
    overall_centroid = X.mean(axis=0)
    
    # Between-cluster dispersion
    between_dispersion = 0
    for label in unique_labels:
        cluster_points = X[labels == label]
        n_k = len(cluster_points)
        centroid_k = cluster_points.mean(axis=0)
        between_dispersion += n_k * np.sum((centroid_k - overall_centroid) ** 2)
    
    # Within-cluster dispersion
    within_dispersion = 0
    for label in unique_labels:
        cluster_points = X[labels == label]
        centroid_k = cluster_points.mean(axis=0)
        within_dispersion += np.sum((cluster_points - centroid_k) ** 2)
    
    if within_dispersion == 0:
        return 0.0
    
    # Calinski-Harabasz score
    score = (between_dispersion / (n_clusters - 1)) / (within_dispersion / (n_samples - n_clusters))
    
    return score


# HIERARCHICAL CLUSTERING UTILITIES

def linkage(X, method='ward'):
    """
    Perform hierarchical clustering and return linkage matrix
    Compatible with scipy.cluster.hierarchy.linkage format
    
    Returns:
        Z: linkage matrix (n_samples-1, 4) containing:
           [idx1, idx2, distance, n_samples_in_cluster]
    """
    X = np.array(X, dtype=np.float64)
    n_samples = len(X)
    
    # Initialize: each point is its own cluster
    # Track which original samples are in each cluster
    clusters = {i: [i] for i in range(n_samples)}
    # Track the centroid/data for each cluster
    cluster_data = {i: X[i:i+1] for i in range(n_samples)}
    
    # Active cluster IDs
    active = set(range(n_samples))
    
    # Linkage matrix
    Z = []
    next_cluster_id = n_samples
    
    def compute_distance(i, j):
        """Compute distance between two clusters"""
        if method == 'single':
            # Minimum distance between any pair
            return np.min([np.linalg.norm(X[p1] - X[p2]) 
                          for p1 in clusters[i] for p2 in clusters[j]])
        elif method == 'complete':
            # Maximum distance between any pair
            return np.max([np.linalg.norm(X[p1] - X[p2]) 
                          for p1 in clusters[i] for p2 in clusters[j]])
        elif method == 'average':
            # Average distance between all pairs
            dists = [np.linalg.norm(X[p1] - X[p2]) 
                    for p1 in clusters[i] for p2 in clusters[j]]
            return np.mean(dists)
        else:  # ward
            # Ward's minimum variance method
            # Formula: sqrt(2 * n_i * n_j / (n_i + n_j)) * ||center_i - center_j||
            center_i = np.mean(cluster_data[i], axis=0)
            center_j = np.mean(cluster_data[j], axis=0)
            n_i = len(clusters[i])
            n_j = len(clusters[j])
            # Squared Euclidean distance between centroids
            squared_dist = np.sum((center_i - center_j) ** 2)
            # Ward's distance
            ward_dist = np.sqrt(2.0 * n_i * n_j / (n_i + n_j) * squared_dist)
            return ward_dist
    
    # Merge clusters iteratively
    while len(active) > 1:
        # Find pair with minimum distance
        min_dist = np.inf
        merge_i, merge_j = None, None
        
        active_list = sorted(active)
        for i in range(len(active_list)):
            for j in range(i + 1, len(active_list)):
                ci, cj = active_list[i], active_list[j]
                dist = compute_distance(ci, cj)
                if dist < min_dist:
                    min_dist = dist
                    merge_i, merge_j = ci, cj
        
        # Record merge in linkage matrix
        n_samples_merged = len(clusters[merge_i]) + len(clusters[merge_j])
        Z.append([merge_i, merge_j, min_dist, n_samples_merged])
        
        # Create new cluster
        clusters[next_cluster_id] = clusters[merge_i] + clusters[merge_j]
        cluster_data[next_cluster_id] = np.vstack([cluster_data[merge_i], cluster_data[merge_j]])
        
        # Remove old clusters from active set
        active.remove(merge_i)
        active.remove(merge_j)
        active.add(next_cluster_id)
        
        next_cluster_id += 1
    
    return np.array(Z)


def dendrogram(Z, no_plot=False):
    """
    Create dendrogram data structure compatible with scipy format
    
    Returns:
        dict with 'icoord' and 'dcoord' for plotting
    """
    Z = np.array(Z, dtype=np.float64)
    n = int(Z[-1, 3])  # Total number of original samples
    
    # Dictionary to store horizontal position for each node
    # Leaves (original samples) get positions 5, 15, 25, ...
    positions = {i: (i * 10) + 5 for i in range(n)}
    
    # Dictionary to store the height (y-coordinate) of each cluster
    heights = {i: 0.0 for i in range(n)}
    
    icoord = []
    dcoord = []
    
    # Process each merge in the linkage matrix
    for i, row in enumerate(Z):
        # Get the two clusters being merged
        left_idx = int(row[0])
        right_idx = int(row[1])
        dist = row[2]
        
        # Get horizontal positions
        left_x = positions[left_idx]
        right_x = positions[right_idx]
        
        # Get heights (y-coordinates) of the two clusters
        left_y = heights[left_idx]
        right_y = heights[right_idx]
        
        # New cluster ID
        new_cluster_id = n + i
        
        # Compute center position for new cluster
        center_x = (left_x + right_x) / 2.0
        positions[new_cluster_id] = center_x
        heights[new_cluster_id] = dist
        
        # Create the "U" shape for this merge
        # Format: [x_left, x_left, x_right, x_right]
        icoord.append([left_x, left_x, right_x, right_x])
        # Format: [y_left, y_merge, y_merge, y_right]
        dcoord.append([left_y, dist, dist, right_y])
    
    return {
        'icoord': icoord,
        'dcoord': dcoord,
        'ivl': [str(i) for i in range(n)],
        'leaves': list(range(n)),
        'color_list': ['C0'] * len(Z)
    }


def fcluster(Z, t, criterion='inconsistent'):
    """
    Form flat clusters from hierarchical clustering
    Simplified version - only supports distance criterion
    
    Args:
        Z: linkage matrix
        t: threshold (distance for 'distance' criterion)
        criterion: 'distance' or 'maxclust'
    
    Returns:
        array of cluster labels
    """
    n_samples = int(Z[-1, 3])
    
    if criterion == 'maxclust':
        # Cut to get specified number of clusters
        n_clusters = int(t)
        cut_height = Z[-(n_clusters - 1), 2] if n_clusters > 1 else Z[-1, 2] + 1
    else:  # distance
        cut_height = t
    
    # Initialize labels
    labels = np.arange(n_samples)
    
    # Process merges below threshold
    for i, merge in enumerate(Z):
        idx1, idx2, dist, count = int(merge[0]), int(merge[1]), merge[2], int(merge[3])
        
        if dist <= cut_height:
            # Merge clusters
            # Find all points with label idx1 or idx2 and assign them the same label
            label_to_use = min(idx1, idx2) if idx1 < n_samples and idx2 < n_samples else n_samples + i
            
            if idx1 < n_samples:
                mask1 = labels == idx1
            else:
                # Find points merged at step idx1 - n_samples
                mask1 = labels == (n_samples + (idx1 - n_samples))
            
            if idx2 < n_samples:
                mask2 = labels == idx2
            else:
                mask2 = labels == (n_samples + (idx2 - n_samples))
            
            labels[mask1 | mask2] = label_to_use
    
    # Relabel to consecutive integers starting from 1
    unique_labels = np.unique(labels)
    label_map = {old: new + 1 for new, old in enumerate(unique_labels)}
    labels = np.array([label_map[label] for label in labels])
    
    return labels
