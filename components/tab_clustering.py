"""
Clustering Component - Modern UI
"""

import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from components.ui_components import render_metric_grid, render_info, render_warning


def render_clustering(clustering_engine, X_reduced, clustering_method, n_clusters, 
                     eps=None, min_samples=None, use_optimal_k=False):
    """Render clustering analysis with modern styling"""
    
    # Find optimal k for KMeans (show analysis if user wants to use optimal k)
    if clustering_method == 'kmeans' and use_optimal_k:
        st.markdown("### Optimal Cluster Analysis")
        
        with st.spinner("Analyzing optimal number of clusters..."):
            k_analysis = clustering_engine.find_optimal_k(X_reduced, max_k=8)
        
        optimal_k_sil = k_analysis['optimal_k_silhouette']
        sil_score = k_analysis['silhouette_scores'][optimal_k_sil-2]
        db_score = k_analysis['db_scores'][optimal_k_sil-2]
        
        # Store k_analysis in session state for later display
        st.session_state['k_analysis'] = k_analysis
        
        # Metrics
        metrics = [
            {"label": "Recommended K", "value": optimal_k_sil, "icon": "🎯"},
            {"label": "Silhouette Score", "value": f"{sil_score:.3f}", "icon": "📊"},
            {"label": "Davies-Bouldin", "value": f"{db_score:.3f}", "icon": "📈"}
        ]
        render_metric_grid(metrics)
        
        render_info(f"Recommended number of clusters: **{optimal_k_sil}** (based on silhouette score)")
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Elbow Method
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=k_analysis['k_range'],
                y=k_analysis['wcss'],
                mode='lines+markers',
                name='WCSS',
                line=dict(color='rgb(102, 126, 234)', width=3),
                marker=dict(size=10)
            ))
            fig1.update_layout(
                title='Elbow Method',
                xaxis_title='Number of Clusters (k)',
                yaxis_title='Within-Cluster Sum of Squares',
                template='plotly_white',
                height=350
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Silhouette Score
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=k_analysis['k_range'],
                y=k_analysis['silhouette_scores'],
                mode='lines+markers',
                name='Silhouette',
                line=dict(color='rgb(16, 185, 129)', width=3),
                marker=dict(size=10)
            ))
            fig2.update_layout(
                title='Silhouette Score',
                xaxis_title='Number of Clusters (k)',
                yaxis_title='Silhouette Score',
                template='plotly_white',
                height=350
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Use optimal k instead of user selection
        n_clusters = optimal_k_sil
        render_info(f"✅ Using optimal K = **{n_clusters}** for clustering")
    
    # Perform clustering
    if clustering_method == 'dbscan':
        clustering_params = {'eps': eps, 'min_samples': min_samples}
    else:
        clustering_params = {}
    
    with st.spinner("Performing clustering analysis..."):
        results = clustering_engine.perform_clustering(
            X_reduced,
            method=clustering_method,
            n_clusters=n_clusters if clustering_method != 'dbscan' else None,
            **clustering_params
        )
    
    # Display results
    st.markdown("### 📊 Clustering Results")
    
    metrics = [
        {"label": "Clusters Found", "value": results['n_clusters'], "icon": "🎯"},
        {"label": "Silhouette Score", "value": f"{results['silhouette_score']:.3f}", "icon": "📊"},
        {"label": "Davies-Bouldin", "value": f"{results['davies_bouldin_score']:.3f}", "icon": "📈"},
        {"label": "Calinski-Harabasz", "value": f"{results['calinski_harabasz_score']:.1f}", "icon": "🔢"}
    ]
    
    render_metric_grid(metrics)
    
    if results['noise_points'] > 0:
        render_warning(f"{results['noise_points']} data points classified as noise (outliers)")
    
    return results
