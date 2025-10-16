"""
Preprocessing Module - Data preprocessing and dimensionality reduction
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class ClusteringPreprocessor:
    """Handle data preprocessing and dimensionality reduction"""
    
    def __init__(self):
        self.scaler = None
        self.reducer = None
        self.dr_info = None
        
    def preprocess_data(self, data, method='standard', ipk_mapping=True):
        """Preprocessing data dengan berbagai metode"""
        df = data.copy()
        
        # Mapping IPK jika diperlukan
        if ipk_mapping:
            ipk_map = {'> 3.5': 5, '3.00 - 3.50': 4, '2.51 - 3.00': 3, 
                      '2.01 - 2.50': 2, '< 2.00': 1}
            if 'Indeks Prestasi Kumulatif (IPK) Terakhir' in df.columns:
                df['IPK_Skala'] = df['Indeks Prestasi Kumulatif (IPK) Terakhir'].map(ipk_map)
        
        # Pilih kolom fitur (skip kolom 0-4 yang biasanya ID, Nama, dll)
        feature_columns = df.columns[5:].tolist()
        if ipk_mapping and 'IPK_Skala' in df.columns:
            # Replace 'Indeks Prestasi Kumulatif (IPK) Terakhir' dengan 'IPK_Skala'
            if 'Indeks Prestasi Kumulatif (IPK) Terakhir' in feature_columns:
                feature_columns.remove('Indeks Prestasi Kumulatif (IPK) Terakhir')
            if 'IPK_Skala' not in feature_columns:
                feature_columns.append('IPK_Skala')
        
        X = df[feature_columns].copy()
        self.feature_names = X.columns.tolist()  # Store to class attribute
        
        # Scaling
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
            
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, self.feature_names
    
    def apply_dimensionality_reduction(self, X_scaled, method='PCA', n_components=None, 
                                      variance_threshold=0.95, **kwargs):
        """Apply dimensionality reduction dengan berbagai metode"""
        if n_components is None:
            # Auto select based on variance threshold untuk PCA
            if method == 'PCA':
                pca = PCA()
                pca.fit(X_scaled)
                cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
                n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
            else:
                n_components = 2  # Default untuk metode lain
        
        if method == 'PCA':
            self.reducer = PCA(n_components=n_components)
            X_reduced = self.reducer.fit_transform(X_scaled)
            
            # Store PCA info
            self.dr_info = {
                'method': 'PCA',
                'n_components': n_components,
                'explained_variance': self.reducer.explained_variance_ratio_,
                'cumulative_variance': np.cumsum(self.reducer.explained_variance_ratio_)
            }
            
        elif method == 'T-SNE':
            perplexity = kwargs.get('perplexity', 30)
            learning_rate = kwargs.get('learning_rate', 200)
            
            self.reducer = TSNE(n_components=n_components, 
                          perplexity=perplexity,
                          learning_rate=learning_rate,
                          random_state=42)
            X_reduced = self.reducer.fit_transform(X_scaled)
            
            self.dr_info = {
                'method': 'T-SNE',
                'n_components': n_components,
                'explained_variance': None
            }
            
        elif method == 'PCA -> T-SNE':
            # Step 1: PCA untuk reduksi awal
            pca_components = kwargs.get('pca_components', min(50, X_scaled.shape[1]))
            pca = PCA(n_components=pca_components)
            X_pca = pca.fit_transform(X_scaled)
            
            # Step 2: t-SNE untuk visualisasi
            perplexity = kwargs.get('perplexity', 30)
            tsne = TSNE(n_components=n_components, 
                       perplexity=perplexity,
                       random_state=42)
            X_reduced = tsne.fit_transform(X_pca)
            
            self.dr_info = {
                'method': 'PCA -> T-SNE',
                'n_components': n_components,
                'explained_variance': pca.explained_variance_ratio_
            }
        
        return X_reduced
