import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import configurations
from config.settings import PAGE_CONFIG, SCALING_METHODS, DR_METHODS, CLUSTERING_METHODS, DEFAULT_PARAMS

# Import utilities
from utils import ClusteringPreprocessor, ClusteringEngine, ClusterAnalyzer

# Import UI components
from components import render_data_overview, render_preprocessing, render_clustering, render_results
from components.ui_components import (
    render_header, render_step_indicator, render_metric_grid,
    render_success, render_info, render_warning, render_section_divider
)

# Import styling
from styles.custom_styles import CUSTOM_CSS


# Configure page
PAGE_CONFIG['layout'] = 'centered'  # Force centered layout
st.set_page_config(**PAGE_CONFIG)

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def main():
    """Main application logic - Modern Single-Page Workflow"""
    
    # Render header
    render_header()
    
    # Initialize session state
    if 'current_step' not in st.session_state:
        st.session_state['current_step'] = 1
    if 'df' not in st.session_state:
        st.session_state['df'] = None
    
    # Render step indicator
    render_step_indicator(st.session_state['current_step'])
    
    render_section_divider()
    
    # ===== STEP 1: DATA UPLOAD =====
    st.markdown("""
        <div class="custom-card animate-fade-in">
            <div class="card-header">
                <div class="card-icon">📁</div>
                <span>Upload Dataset</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Download Template and Load Sample Data buttons
    col1, col2, col3 = st.columns([1.2, 1.2, 1.6])
    with col1:
        # Download template button
        with open("template-pertanyaan.csv", "rb") as template_file:
            st.download_button(
                label="Download Template",
                data=template_file,
                file_name="template-pertanyaan.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_template_btn"
            )
    
    with col2:
        # Load sample data button
        if st.button("Load Sample Data", use_container_width=True, key="load_sample_btn"):
            try:
                df = pd.read_csv("dataset.csv")
                st.session_state['df'] = df
                st.session_state['uploaded_file_name'] = 'dataset.csv'
                if st.session_state['current_step'] == 1:
                    st.session_state['current_step'] = 2
                st.rerun()
            except Exception as e:
                st.error(f"Error loading sample data: {e}")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV dataset", type=['csv'], label_visibility="collapsed")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Only save to session if it's a new file or first upload
            if st.session_state['df'] is None or 'uploaded_file_name' not in st.session_state or st.session_state.get('uploaded_file_name') != uploaded_file.name:
                st.session_state['df'] = df
                st.session_state['uploaded_file_name'] = uploaded_file.name
                # Mark step 1 as completed (move to step 2)
                if st.session_state['current_step'] == 1:
                    st.session_state['current_step'] = 2
                    render_success(f"Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
                    st.rerun()  # Rerun to update step indicator
            render_success(f"Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
        except Exception as e:
            st.error(f"Error: {e}")
            return
    
    if st.session_state['df'] is None:
        render_info("Upload file CSV untuk memulai analisis clustering")
        return
    
    df = st.session_state['df']
    
    # Data Overview
    with st.expander("View Data Overview", expanded=False):
        render_data_overview(df)
    
    render_section_divider()
    
    # ===== STEP 2: PREPROCESSING CONFIGURATION =====
    st.markdown("""
        <div class="custom-card animate-fade-in">
            <div class="card-header">
                <div class="card-icon">🔧</div>
                <span>Preprocessing & Dimensionality Reduction</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Scaling Method**")
        scaling_method = st.selectbox(
            "scaling",
            SCALING_METHODS,
            index=0,
            label_visibility="collapsed",
            help="Pilih metode scaling untuk normalisasi data"
        )
    
    with col2:
        st.markdown("**Dimensionality Reduction**")
        dr_method = st.selectbox(
            "dr_method",
            DR_METHODS,
            index=0,
            label_visibility="collapsed",
            help="Pilih metode reduksi dimensi"
        )
    
    with col3:
        st.markdown("**Components**")
        dr_components = st.radio(
            "components",
            ['Auto (95% variance)', 'Manual'],
            index=0,
            label_visibility="collapsed"
        )
        if dr_components == 'Manual':
            # Hitung jumlah fitur numerik (kolom 5 ke atas + IPK_Skala)
            num_features = len(df.columns[5:])  # Fitur mulai dari kolom ke-5
            # Maksimum komponen tidak boleh melebihi jumlah fitur atau sampel
            max_components = min(num_features, df.shape[0] - 1)
            default_components = min(3, max_components)
            n_components = st.slider(
                "Number of Components", 
                2, 
                max_components, 
                default_components,
                help=f"Max: {max_components} komponen (dari {num_features} fitur numerik)"
            )
        else:
            n_components = None
    
    # No additional parameters needed for PCA
    dr_params = {}
    
    # Run Preprocessing Button
    if st.button("Run Preprocessing", key="preprocess_btn", use_container_width=True):
        preprocessor = ClusteringPreprocessor()
        with st.spinner("Processing data..."):
            X_scaled, feature_names = preprocessor.preprocess_data(df, method=scaling_method)
            
            if dr_components == 'Auto (95% variance)':
                X_reduced = preprocessor.apply_dimensionality_reduction(
                    X_scaled, method=dr_method, variance_threshold=0.95, **dr_params
                )
            else:
                X_reduced = preprocessor.apply_dimensionality_reduction(
                    X_scaled, method=dr_method, n_components=n_components, **dr_params
                )
            
            st.session_state['X_scaled'] = X_scaled
            st.session_state['X_reduced'] = X_reduced
            st.session_state['feature_names'] = feature_names
            st.session_state['dr_info'] = preprocessor.dr_info
            st.session_state['preprocessor'] = preprocessor
            # Mark step 2 as completed (move to step 3)
            if st.session_state['current_step'] == 2:
                st.session_state['current_step'] = 3
            
            render_success("Preprocessing completed successfully!")
            st.rerun()
    
    # Show preprocessing results if available
    if 'X_reduced' in st.session_state:
        with st.expander("View Preprocessing Results", expanded=True):
            render_preprocessing(
                st.session_state['preprocessor'],
                df,
                scaling_method,
                dr_method,
                dr_components,
                n_components,
                dr_params
            )
    
    if 'X_reduced' not in st.session_state:
        return
    
    render_section_divider()
    
    # ===== STEP 3: CLUSTERING =====
    st.markdown("""
        <div class="custom-card animate-fade-in">
            <div class="card-header">
                <div class="card-icon">✨</div>
                <span>Clustering Configuration</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Clustering Algorithm**")
        clustering_method = st.selectbox(
            "clustering",
            CLUSTERING_METHODS,
            index=0,
            label_visibility="collapsed"
        )
    
    with col2:
        if clustering_method == 'kmeans':
            st.markdown("**Cluster Selection Mode**")
            use_optimal_k = st.checkbox("Use Optimal K", value=False, key="use_optimal_k")
            
            if not use_optimal_k:
                st.markdown("**Number of Clusters**")
                n_clusters = st.slider("n_clusters", 2, 8, DEFAULT_PARAMS['n_clusters_default'], label_visibility="collapsed")
            else:
                n_clusters = None  # Will be determined by optimal k analysis
        elif clustering_method == 'hierarchical':
            # No parameters needed for hierarchical
            n_clusters = None
            use_optimal_k = False
        else:
            n_clusters = None
            use_optimal_k = False
    
    with col3:
        if clustering_method == 'dbscan':
            st.markdown("**DBSCAN Parameters**")
            # Simple controls without recommendation/graph
            eps = st.slider(
                "EPS (Epsilon)",
                0.1,
                10.0,
                float(DEFAULT_PARAMS['eps_default']),
                help="Maximum distance between two samples to be considered neighbors"
            )
            min_samples = st.slider(
                "Min Samples",
                2,
                20,
                int(DEFAULT_PARAMS['min_samples_default']),
                help="Minimum number of samples in a neighborhood to form a core point"
            )
        else:
            eps = None
            min_samples = None
    
    # Run Clustering Button
    if st.button("✨ Run Clustering Analysis", key="cluster_btn", use_container_width=True):
        clustering_engine = ClusteringEngine()
        
        # Clear previous session state to avoid UI persistence bugs
        # Bug Fix 1: Clear k_analysis when switching methods or not using optimal K
        if clustering_method != 'kmeans' or not use_optimal_k:
            if 'k_analysis' in st.session_state:
                del st.session_state['k_analysis']
        
        # Bug Fix 2: Reset cluster navigation index when re-running clustering
        st.session_state['current_cluster_index'] = 0
        
        with st.spinner("Performing clustering analysis..."):
            results = render_clustering(
                clustering_engine,
                st.session_state['X_reduced'],
                clustering_method,
                n_clusters if clustering_method != 'dbscan' else DEFAULT_PARAMS['n_clusters_default'],
                eps,
                min_samples,
                use_optimal_k=use_optimal_k if clustering_method == 'kmeans' else False
            )
            
            if results:
                st.session_state['clustering_results'] = results
                st.session_state['clustering_engine'] = clustering_engine
                # Mark step 3 as completed (move to step 4)
                if st.session_state['current_step'] == 3:
                    st.session_state['current_step'] = 4
                render_success("Clustering analysis completed!")
                st.rerun()
    
    if 'clustering_results' not in st.session_state:
        return
    
    render_section_divider()
    
    # ===== STEP 4: RESULTS ===== 
    analyzer = ClusterAnalyzer()
    render_results(
        st.session_state['clustering_engine'],
        analyzer,
        st.session_state['X_reduced'],
        st.session_state['dr_info'],
        df,
        st.session_state['feature_names']
    )


if __name__ == "__main__":
    main()
