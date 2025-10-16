"""
Configuration Module - App settings and constants
"""

# Page Configuration
PAGE_CONFIG = {
    'page_title': "Study Xplorer",
    'page_icon': "🧾",
    'layout': "wide",
    'initial_sidebar_state': "expanded"
}

# Scaling Methods
SCALING_METHODS = ['standard', 'minmax', 'robust']

# Dimensionality Reduction Methods
DR_METHODS = ['PCA', 'T-SNE', 'PCA -> T-SNE']

# Clustering Methods
CLUSTERING_METHODS = ['kmeans', 'hierarchical', 'dbscan']

# Default Parameters
DEFAULT_PARAMS = {
    'max_k': 8,
    'n_clusters_default': 3,
    'perplexity_default': 30,
    'learning_rate_default': 200,
    'eps_default': 0.5,
    'min_samples_default': 5
}
