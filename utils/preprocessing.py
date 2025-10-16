import numpy as np
from utils.custom_algorithms import StandardScaler, MinMaxScaler, RobustScaler, PCA

class ClusteringPreprocessor:
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
        """Apply dimensionality reduction dengan PCA"""
        if n_components is None:
            # Auto select based on variance threshold untuk PCA
            pca = PCA()
            pca.fit(X_scaled)
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        
        # PCA only
        self.reducer = PCA(n_components=n_components)
        X_reduced = self.reducer.fit_transform(X_scaled)
        
        # Store PCA info
        self.dr_info = {
            'method': 'PCA',
            'n_components': n_components,
            'explained_variance': self.reducer.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(self.reducer.explained_variance_ratio_)
        }
        
        return X_reduced
