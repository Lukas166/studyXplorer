import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi page
st.set_page_config(
    page_title="Clustering Analysis Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .cluster-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

class ClusteringAnalysis:
    def __init__(self, data):
        self.data = data
        self.X = None
        self.X_scaled = None
        self.X_reduced = None
        self.feature_names = None
        self.results = {}
        
    def preprocess_data(self, method='standard', ipk_mapping=True):
        """Preprocessing data dengan berbagai metode"""
        df = self.data.copy()
        
        # Mapping IPK jika diperlukan
        if ipk_mapping:
            ipk_map = {'> 3.5': 5, '3.00 - 3.50': 4, '2.51 - 3.00': 3, 
                      '2.01 - 2.50': 2, '< 2.00': 1}
            df['IPK_Skala'] = df['Indeks Prestasi Kumulatif (IPK) Terakhir'].map(ipk_map)
        
        # Pilih kolom fitur
        feature_columns = df.columns[5:].tolist()
        if ipk_mapping:
            feature_columns = feature_columns[:-1] + ['IPK_Skala']
        
        self.X = df[feature_columns].copy()
        self.feature_names = self.X.columns.tolist()
        
        # Scaling
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
            
        self.X_scaled = scaler.fit_transform(self.X)
        
        return self.X_scaled
    
    def apply_dimensionality_reduction(self, method='PCA', n_components=None, variance_threshold=0.95, **kwargs):
        """Apply dimensionality reduction dengan berbagai metode"""
        if n_components is None:
            # Auto select based on variance threshold untuk PCA
            if method == 'PCA':
                pca = PCA()
                pca.fit(self.X_scaled)
                cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
                n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
            else:
                n_components = 2  # Default untuk metode lain
        
        if method == 'PCA':
            reducer = PCA(n_components=n_components)
            self.X_reduced = reducer.fit_transform(self.X_scaled)
            
            # Store PCA info
            self.dr_info = {
                'method': 'PCA',
                'n_components': n_components,
                'explained_variance': reducer.explained_variance_ratio_,
                'cumulative_variance': np.cumsum(reducer.explained_variance_ratio_)
            }
            
        elif method == 'T-SNE':
            perplexity = kwargs.get('perplexity', 30)
            learning_rate = kwargs.get('learning_rate', 200)
            
            reducer = TSNE(n_components=n_components, 
                          perplexity=perplexity,
                          learning_rate=learning_rate,
                          random_state=42)
            self.X_reduced = reducer.fit_transform(self.X_scaled)
            
            self.dr_info = {
                'method': 'T-SNE',
                'n_components': n_components,
                'explained_variance': None
            }
            
        elif method == 'PCA -> T-SNE':
            # Step 1: PCA untuk reduksi awal
            pca_components = kwargs.get('pca_components', min(50, self.X_scaled.shape[1]))
            pca = PCA(n_components=pca_components)
            X_pca = pca.fit_transform(self.X_scaled)
            
            # Step 2: t-SNE untuk visualisasi
            perplexity = kwargs.get('perplexity', 30)
            tsne = TSNE(n_components=n_components, 
                       perplexity=perplexity,
                       random_state=42)
            self.X_reduced = tsne.fit_transform(X_pca)
            
            self.dr_info = {
                'method': 'PCA -> T-SNE',
                'n_components': n_components,
                'explained_variance': pca.explained_variance_ratio_
            }
        
        return self.X_reduced
    
    def find_optimal_k(self, max_k=10):
        """Cari nilai k optimal untuk KMeans"""
        wcss = []
        silhouette_scores = []
        db_scores = []
        
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.X_reduced)
            
            wcss.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.X_reduced, labels))
            db_scores.append(davies_bouldin_score(self.X_reduced, labels))
        
        # Find optimal k based on silhouette score
        optimal_k_silhouette = np.argmax(silhouette_scores) + 2
        optimal_k_elbow = self._find_elbow_point(wcss) + 2
        
        self.k_analysis = {
            'k_range': list(range(2, max_k + 1)),
            'wcss': wcss,
            'silhouette_scores': silhouette_scores,
            'db_scores': db_scores,
            'optimal_k_silhouette': optimal_k_silhouette,
            'optimal_k_elbow': optimal_k_elbow
        }
        
        return self.k_analysis
    
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
    
    def perform_clustering(self, method='kmeans', n_clusters=3, **kwargs):
        """Perform clustering dengan berbagai metode"""
        if method == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == 'hierarchical':
            model = AgglomerativeClustering(n_clusters=n_clusters)
        elif method == 'dbscan':
            eps = kwargs.get('eps', 0.5)
            min_samples = kwargs.get('min_samples', 5)
            model = DBSCAN(eps=eps, min_samples=min_samples)
        else:
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        labels = model.fit_predict(self.X_reduced)
        
        # Calculate metrics
        n_clusters_found = len(np.unique(labels[labels != -1])) if method == 'dbscan' else n_clusters
        noise_points = np.sum(labels == -1) if method == 'dbscan' else 0
        
        if n_clusters_found > 1:
            silhouette = silhouette_score(self.X_reduced, labels)
            db_index = davies_bouldin_score(self.X_reduced, labels)
            calinski = calinski_harabasz_score(self.X_reduced, labels)
        else:
            silhouette = db_index = calinski = -1
        
        self.clustering_results = {
            'method': method,
            'labels': labels,
            'n_clusters': n_clusters_found,
            'noise_points': noise_points,
            'silhouette_score': silhouette,
            'davies_bouldin_score': db_index,
            'calinski_harabasz_score': calinski,
            'model': model
        }
        
        return self.clustering_results
    
    def analyze_cluster_profiles(self):
        """Analisis profil setiap cluster"""
        if not hasattr(self, 'clustering_results'):
            return None
        
        df_analysis = self.data.copy()
        df_analysis['Cluster'] = self.clustering_results['labels']
        
        # Add IPK if not exists
        if 'IPK_Skala' not in df_analysis.columns:
            ipk_map = {'> 3.5': 4, '3.00 - 3.50': 3, '2.51 - 3.00': 2, 
                      '2.01 - 2.50': 1, '< 2.00': 0}
            df_analysis['IPK_Skala'] = df_analysis['Indeks Prestasi Kumulatif (IPK) Terakhir'].map(ipk_map)
        
        # Profile analysis
        profile_cols = self.feature_names + ['IPK_Skala']
        cluster_profiles = df_analysis.groupby('Cluster')[profile_cols].mean()
        
        # Interpretation
        interpretations = []
        for cluster in cluster_profiles.index:
            profile = cluster_profiles.loc[cluster]
            
            # Learning style analysis
            # Deep Learning Indicators
            deep_learning_score = (
                profile.get('Saya sering mencari sumber belajar (jurnal, buku, video) di luar materi yang diberikan oleh dosen.', 0) +
                profile.get('Saya suka mencari tahu hubungan antara materi kuliah dengan isu-isu di dunia nyata atau bidang ilmu lain.', 0) +
                profile.get('Saya baru merasa benar-benar paham suatu topik jika saya bisa menjelaskannya kepada orang lain, bukan hanya bisa mengerjakan soalnya.', 0) +
                profile.get('Jika menemukan konsep yang rumit, saya tertantang untuk memahaminya lebih dalam, meskipun itu memakan banyak waktu.', 0)
            ) / 4
            
            # Surface Learning Indicators
            surface_learning_score = (
                profile.get('Tujuan utama saya belajar adalah untuk mendapatkan nilai ujian yang tinggi.', 0) +
                profile.get('Bagi saya, membaca ulang slide presentasi dan catatan dari dosen sudah lebih dari cukup untuk persiapan ujian.', 0)
            ) / 2
            
            # Strategic Learning Indicators
            strategic_learning_score = (
                profile.get('Saya merasa adanya tekanan deadline membuat saya lebih fokus dan termotivasi.', 0) +
                profile.get('Saya cenderung mempelajari materi secara bertahap sepanjang semester, bukan hanya saat mendekati ujian (Sistem Kebut Semalam).', 0) +
                profile.get('Saya secara aktif menggunakan umpan balik (feedback) dari dosen pada tugas untuk memperbaiki pemahaman konseptual saya, bukan hanya untuk memperbaiki nilai.', 0)
            ) / 3
            
            # Determine dominant learning style
            learning_styles = {
                'Deep Learner': deep_learning_score,
                'Surface Learner': surface_learning_score,
                'Strategic Learner': strategic_learning_score
            }
            dominant_style = max(learning_styles, key=learning_styles.get)
            
            interpretation = {
                'cluster': cluster,
                'size': len(df_analysis[df_analysis['Cluster'] == cluster]),
                'ipk': profile.get('IPK_Skala', 0),
                'deep_learning': deep_learning_score,
                'surface_learning': surface_learning_score,
                'strategic_learning': strategic_learning_score,
                'dominant_style': dominant_style,
                'recommendations': self._generate_recommendations(dominant_style, profile)
            }
            interpretations.append(interpretation)
        
        return {
            'profiles': cluster_profiles,
            'interpretations': interpretations,
            'data_with_clusters': df_analysis
        }

    def _generate_recommendations(self, learning_style, profile):
        """Generate recommendations based on learning style"""
        recommendations = []
        
        if learning_style == 'Deep Learner':
            recommendations.extend([
                "✅ Pertahankan gaya belajar mendalam Anda",
                "✅ Terus eksplorasi materi di luar kurikulum",
                "✅ Manfaatkan kemampuan menjelaskan konsep untuk membantu teman",
                "⚡ Coba aplikasikan pengetahuan dalam proyek nyata"
            ])
        elif learning_style == 'Surface Learner':
            recommendations.extend([
                "📚 Coba fokus pada pemahaman konsep daripada sekadar nilai",
                "🔍 Eksplorasi hubungan antara materi kuliah dengan aplikasi nyata",
                "💡 Bergabung dengan kelompok diskusi untuk memperdalam pemahaman",
                "🎯 Setel tujuan belajar yang lebih dari sekadar nilai ujian"
            ])
        else:  # Strategic Learner
            recommendations.extend([
                "⏰ Manfaatkan kemampuan manajemen waktu yang baik",
                "📊 Kombinasikan dengan pendekatan deep learning untuk hasil optimal",
                "🔄 Terus gunakan feedback untuk improvement",
                "🎯 Eksplorasi metode belajar yang lebih variatif"
            ])
        
        # Additional recommendations based on specific scores
        if profile.get('Saya sering mencari sumber belajar (jurnal, buku, video) di luar materi yang diberikan oleh dosen.', 0) < 3:
            recommendations.append("🌐 Coba eksplorasi sumber belajar tambahan di luar materi kuliah")
        
        if profile.get('Saya cenderung mempelajari materi secara bertahap sepanjang semester, bukan hanya saat mendekati ujian (Sistem Kebut Semalam).', 0) < 3:
            recommendations.append("📅 Kembangkan kebiasaan belajar konsisten sepanjang semester")
        
        return recommendations

def main():
    # Header
    st.markdown('<h1 class="main-header">🎓 Clustering Analysis Tool</h1>', unsafe_allow_html=True)
    st.markdown("Analisis clustering untuk data mahasiswa dengan berbagai metode preprocessing dan algoritma")
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload dataset CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return
    else:
        # Use sample data
        st.info("📁 Upload file CSV atau gunakan sample data untuk memulai analisis")
        return
    
    # Preprocessing options
    st.sidebar.subheader("🔧 Preprocessing")
    scaling_method = st.sidebar.selectbox(
        "Scaling Method",
        ['standard', 'minmax', 'robust'],
        index=0,
        help="Pilih metode scaling untuk preprocessing data"
    )
    
    # Dimensionality Reduction options - MODIFIED SECTION
    st.sidebar.subheader("📊 Dimensionality Reduction")
    dr_method = st.sidebar.selectbox(
        "Reduction Method",
        ['PCA', 'T-SNE', 'PCA -> T-SNE'],
        index=0,
        help="Pilih metode reduksi dimensi"
    )

    if dr_method in ['PCA', 'PCA -> T-SNE']:
        dr_components = st.sidebar.radio(
            "Components Selection",
            ['Auto (95% variance)', 'Manual'],
            index=0
        )
        if dr_components == 'Manual':
            n_components = st.sidebar.slider("Number of Components", 2, 10, 3)
        else:
            n_components = None
    
    # Parameters khusus untuk t-SNE based methods
    if dr_method in ['T-SNE', 'PCA -> T-SNE']:
        st.sidebar.subheader("t-SNE Parameters")
        perplexity = st.sidebar.slider("Perplexity", 5, 50, 30)
        learning_rate = st.sidebar.slider("Learning Rate", 10, 1000, 200)
        dr_params = {'perplexity': perplexity, 'learning_rate': learning_rate}
    else:
        dr_params = {}
    
    # Clustering options
    st.sidebar.subheader("🎯 Clustering Algorithm")
    clustering_method = st.sidebar.selectbox(
        "Method",
        ['kmeans', 'hierarchical', 'dbscan'],
        index=0
    )
    
    if clustering_method == 'kmeans':
        n_clusters = st.sidebar.slider("Number of Clusters", 2, 8, 3)
    elif clustering_method == 'hierarchical':
        n_clusters = st.sidebar.slider("Number of Clusters", 2, 8, 3)
    else:  # dbscan
        eps = st.sidebar.slider("EPS", 0.1, 2.0, 0.5, 0.1)
        min_samples = st.sidebar.slider("Min Samples", 2, 10, 5)
    
    # Initialize analyzer
    analyzer = ClusteringAnalysis(df)
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Data Overview", "🔍 Preprocessing", "🎯 Clustering", "📊 Results"])
    
    with tab1:
        st.subheader("Data Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dataset Preview:**")
            st.dataframe(df.head(), use_container_width=True)
        
        with col2:
            st.write("**Dataset Info:**")
            st.write(f"Shape: {df.shape}")
            st.write(f"Columns: {len(df.columns)}")
            
            # Show basic statistics
            st.write("**Basic Statistics:**")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    
    with tab2:
        st.subheader("Data Preprocessing")
        
        # Perform preprocessing
        with st.spinner("Processing data..."):
            X_scaled = analyzer.preprocess_data(method=scaling_method)
            
            # Apply Dimensionality Reduction - MODIFIED
            if dr_components == 'Auto (95% variance)':
                X_reduced = analyzer.apply_dimensionality_reduction(
                    method=dr_method, 
                    variance_threshold=0.95,
                    **dr_params
                )
            else:
                X_reduced = analyzer.apply_dimensionality_reduction(
                    method=dr_method,
                    n_components=n_components,
                    **dr_params
                )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Preprocessing Summary:**")
            st.metric("Original Features", len(analyzer.feature_names))
            st.metric("Scaled Data Shape", f"{X_scaled.shape}")
            st.metric(f"{analyzer.dr_info['method']} Components", analyzer.dr_info['n_components'])
            
            if analyzer.dr_info['explained_variance'] is not None:
                cumulative_variance = np.cumsum(analyzer.dr_info['explained_variance'])
                st.metric("Variance Explained", f"{cumulative_variance[-1]:.1%}")
        
        with col2:
            # Variance plot untuk metode yang support explained variance
            if analyzer.dr_info['explained_variance'] is not None:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[f'Comp{i+1}' for i in range(analyzer.dr_info['n_components'])],
                    y=analyzer.dr_info['explained_variance'],
                    name='Explained Variance'
                ))
                fig.add_trace(go.Scatter(
                    x=[f'Comp{i+1}' for i in range(analyzer.dr_info['n_components'])],
                    y=np.cumsum(analyzer.dr_info['explained_variance']),
                    name='Cumulative Variance',
                    yaxis='y2'
                ))
                fig.update_layout(
                    title=f'{analyzer.dr_info["method"]} Variance Explained',
                    xaxis_title='Components',
                    yaxis_title='Explained Variance',
                    yaxis2=dict(title='Cumulative Variance', overlaying='y', side='right'),
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"**{analyzer.dr_info['method']}** tidak menyediakan explained variance ratio")
    
    with tab3:
        st.subheader("Clustering Analysis")
        
        # Find optimal k untuk KMeans
        if clustering_method == 'kmeans':
            with st.spinner("Finding optimal k..."):
                k_analysis = analyzer.find_optimal_k(max_k=8)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Optimal K Recommendation:**")
                
                # Metrics for optimal k
                optimal_k_sil = k_analysis['optimal_k_silhouette']
                sil_score = k_analysis['silhouette_scores'][optimal_k_sil-2]
                db_score = k_analysis['db_scores'][optimal_k_sil-2]
                
                st.metric("Recommended K (Silhouette)", optimal_k_sil)
                st.metric("Silhouette Score", f"{sil_score:.3f}")
                st.metric("Davies-Bouldin Score", f"{db_score:.3f}")
                
                st.info(f"🎯 Recommended number of clusters: **{optimal_k_sil}** based on silhouette score")
            
            with col2:
                # Plot metrics
                fig = make_subplots(rows=2, cols=1, subplot_titles=('Elbow Method', 'Silhouette Score'))
                
                # Elbow plot
                fig.add_trace(
                    go.Scatter(x=k_analysis['k_range'], y=k_analysis['wcss'], 
                              mode='lines+markers', name='WCSS'),
                    row=1, col=1
                )
                
                # Silhouette plot
                fig.add_trace(
                    go.Scatter(x=k_analysis['k_range'], y=k_analysis['silhouette_scores'],
                              mode='lines+markers', name='Silhouette', line=dict(color='green')),
                    row=2, col=1
                )
                
                fig.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        # Perform clustering
        st.write("**Clustering Execution:**")
        if clustering_method == 'dbscan':
            clustering_params = {'eps': eps, 'min_samples': min_samples}
        else:
            clustering_params = {}
        
        if st.button("Run Clustering", type="primary"):
            with st.spinner("Performing clustering..."):
                results = analyzer.perform_clustering(
                    method=clustering_method,
                    n_clusters=n_clusters if clustering_method != 'dbscan' else None,
                    **clustering_params
                )
            
            # Display clustering results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Clusters Found", results['n_clusters'])
            with col2:
                st.metric("Silhouette Score", f"{results['silhouette_score']:.3f}")
            with col3:
                st.metric("Davies-Bouldin", f"{results['davies_bouldin_score']:.3f}")
            
            if results['noise_points'] > 0:
                st.warning(f"⚠️ {results['noise_points']} points classified as noise")
    
    with tab4:
        st.subheader("Clustering Results & Interpretation")
        
        if not hasattr(analyzer, 'clustering_results'):
            st.info("🚀 Run clustering first to see results!")
        else:
            # Visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # 2D Scatter plot
                method_name = analyzer.dr_info['method']
                comp1_label = f'{method_name} Component 1'
                comp2_label = f'{method_name} Component 2'
                
                fig = px.scatter(
                    x=analyzer.X_reduced[:, 0],
                    y=analyzer.X_reduced[:, 1],
                    color=analyzer.clustering_results['labels'].astype(str),
                    title=f"Cluster Visualization ({method_name})",
                    labels={'x': comp1_label, 'y': comp2_label},
                    color_discrete_sequence=px.colors.qualitative.Set1
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # 3D Scatter plot if enough components
                if analyzer.X_reduced.shape[1] >= 3:
                    fig = px.scatter_3d(
                        x=analyzer.X_reduced[:, 0],
                        y=analyzer.X_reduced[:, 1],
                        z=analyzer.X_reduced[:, 2],
                        color=analyzer.clustering_results['labels'].astype(str),
                        title=f"3D Cluster Visualization ({method_name})",
                        labels={'x': f'{method_name} 1', 'y': f'{method_name} 2', 'z': f'{method_name} 3'},
                        color_discrete_sequence=px.colors.qualitative.Set1
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Cluster profiles and interpretations
            st.subheader("📋 Cluster Profiles & Recommendations")
            
            with st.spinner("Analyzing cluster profiles..."):
                profiles = analyzer.analyze_cluster_profiles()
            
            if profiles:
                for interpretation in profiles['interpretations']:
                    if interpretation['cluster'] != -1:  # Skip noise cluster
                        with st.expander(f"🎯 Cluster {interpretation['cluster']} - {interpretation['dominant_style']} "f"({interpretation['size']} students, IPK: {float(np.mean(interpretation['ipk'])):.2f})"):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.write("**Learning Style Scores:**")
                                learning_data = {
                                    'Deep Learning': interpretation['deep_learning'],
                                    'Surface Learning': interpretation['surface_learning'],
                                    'Strategic Learning': interpretation['strategic_learning']
                                }
                                
                                fig = go.Figure(go.Bar(
                                    x=list(learning_data.keys()),
                                    y=list(learning_data.values()),
                                    marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']
                                ))
                                fig.update_layout(
                                    title=f"Learning Style Profile - Cluster {interpretation['cluster']}",
                                    yaxis_title="Score",
                                    height=300
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                st.write("**Recommendations:**")
                                for rec in interpretation['recommendations']:
                                    st.write(rec)
            
            # Download results
            st.subheader("📥 Download Results")
            if profiles:
                results_df = profiles['data_with_clusters']
                csv = results_df.to_csv(index=False)
                
                st.download_button(
                    label="Download Clustering Results (CSV)",
                    data=csv,
                    file_name="clustering_results.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()