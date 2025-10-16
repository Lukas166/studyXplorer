"""
Results Component - Modern UI
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from components.ui_components import render_metric_grid, render_cluster_badge, render_info


def render_results(clustering_engine, analyzer, X_reduced, dr_info, df, feature_names):
    """Render results with modern styling"""
    
    if clustering_engine.clustering_results is None:
        st.info("🚀 Run clustering first to see results!")
        return
    
    labels = clustering_engine.clustering_results['labels']
    method_name = dr_info['method']
    
    # Show K Analysis if available (from optimal k selection)
    if 'k_analysis' in st.session_state:
        st.markdown("### Optimal Cluster Analysis")
        k_analysis = st.session_state['k_analysis']
        
        optimal_k_sil = k_analysis['optimal_k_silhouette']
        sil_score = k_analysis['silhouette_scores'][optimal_k_sil-2]
        db_score = k_analysis['db_scores'][optimal_k_sil-2]
        
        # Metrics
        metrics = [
            {"label": "Recommended K", "value": optimal_k_sil, "icon": "🎯"},
            {"label": "Silhouette Score", "value": f"{sil_score:.3f}", "icon": "📊"},
            {"label": "Davies-Bouldin", "value": f"{db_score:.3f}", "icon": "📈"}
        ]
        render_metric_grid(metrics)
        
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
        
        st.markdown("<br>", unsafe_allow_html=True)
    
    # Summary Metrics
    st.markdown("### Clustering Summary")
    unique_labels = np.unique(labels[labels != -1])
    n_clusters = len(unique_labels)
    
    metrics = []
    for cluster_id in unique_labels:
        cluster_size = np.sum(labels == cluster_id)
        metrics.append({
            "label": f"Cluster {cluster_id}",
            "value": f"{cluster_size}",
            "icon": "🎯"
        })
    
    render_metric_grid(metrics)
    
    # Visualizations
    st.markdown("### Cluster Visualizations")
    
    # Color palette - colorful
    colors = px.colors.qualitative.Set2
    
    # Radio button untuk memilih jenis plot
    plot_type = st.radio(
        "Select visualization type:",
        ["2D Plot", "3D Plot"] if X_reduced.shape[1] >= 3 else ["2D Plot"],
        horizontal=True,
        key="plot_type_radio"
    )
    
    if plot_type == "2D Plot":
        # 2D Visualization - Static Image
        fig_2d = go.Figure()
        
        for cluster_id in unique_labels:
            mask = labels == cluster_id
            fig_2d.add_trace(go.Scatter(
                x=X_reduced[mask, 0],
                y=X_reduced[mask, 1],
                mode='markers',
                name=f'Cluster {cluster_id}',
                marker=dict(
                    size=10,
                    color=colors[cluster_id % len(colors)],
                    line=dict(width=1, color='white')
                )
            ))
        
        # Add noise points if any
        if np.any(labels == -1):
            noise_mask = labels == -1
            fig_2d.add_trace(go.Scatter(
                x=X_reduced[noise_mask, 0],
                y=X_reduced[noise_mask, 1],
                mode='markers',
                name='Noise',
                marker=dict(size=8, color='#cccccc', symbol='x')
            ))
        
        fig_2d.update_layout(
            title=dict(
                text=f'2D Cluster Visualization ({method_name})',
                font=dict(color='#000000', size=16)
            ),
            xaxis=dict(
                title=dict(
                    text=f'{method_name} Component 1',
                    font=dict(color='#000000')
                ),
                tickfont=dict(color='#000000'),
                gridcolor='#cccccc',
                zerolinecolor='#000000',
                linecolor='#000000'
            ),
            yaxis=dict(
                title=dict(
                    text=f'{method_name} Component 2',
                    font=dict(color='#000000')
                ),
                tickfont=dict(color='#000000'),
                gridcolor='#cccccc',
                zerolinecolor='#000000',
                linecolor='#000000'
            ),
            template='plotly_white',
            height=600,
            showlegend=True,
            legend=dict(
                font=dict(color='#000000')
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#000000')
        )
        
        # Make it static (no interaction)
        st.plotly_chart(fig_2d, use_container_width=True, config={'staticPlot': True})
        
    else:  # 3D Plot
        if X_reduced.shape[1] >= 3:
            fig_3d = go.Figure()
            
            for cluster_id in unique_labels:
                mask = labels == cluster_id
                fig_3d.add_trace(go.Scatter3d(
                    x=X_reduced[mask, 0],
                    y=X_reduced[mask, 1],
                    z=X_reduced[mask, 2],
                    mode='markers',
                    name=f'Cluster {cluster_id}',
                    marker=dict(
                        size=6,
                        color=colors[cluster_id % len(colors)],
                        line=dict(width=0.5, color='white')
                    )
                ))
            
            if np.any(labels == -1):
                noise_mask = labels == -1
                fig_3d.add_trace(go.Scatter3d(
                    x=X_reduced[noise_mask, 0],
                    y=X_reduced[noise_mask, 1],
                    z=X_reduced[noise_mask, 2],
                    mode='markers',
                    name='Noise',
                    marker=dict(size=4, color='#cccccc', symbol='x')
                ))
            
            fig_3d.update_layout(
                title=dict(
                    text=f'3D Cluster Visualization ({method_name})',
                    font=dict(color='#000000', size=16)
                ),
                scene=dict(
                    xaxis=dict(
                        title=dict(
                            text=f'{method_name} 1',
                            font=dict(color='#000000')
                        ),
                        tickfont=dict(color='#000000'),
                        gridcolor='#cccccc',
                        zerolinecolor='#000000',
                        linecolor='#000000',
                        backgroundcolor='white'
                    ),
                    yaxis=dict(
                        title=dict(
                            text=f'{method_name} 2',
                            font=dict(color='#000000')
                        ),
                        tickfont=dict(color='#000000'),
                        gridcolor='#cccccc',
                        zerolinecolor='#000000',
                        linecolor='#000000',
                        backgroundcolor='white'
                    ),
                    zaxis=dict(
                        title=dict(
                            text=f'{method_name} 3',
                            font=dict(color='#000000')
                        ),
                        tickfont=dict(color='#000000'),
                        gridcolor='#cccccc',
                        zerolinecolor='#000000',
                        linecolor='#000000',
                        backgroundcolor='white'
                    ),
                    bgcolor='white'
                ),
                template='plotly_white',
                height=600,
                showlegend=True,
                legend=dict(
                    font=dict(color='#000000')
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='#000000')
            )
            
            # Static 3D plot
            st.plotly_chart(fig_3d, use_container_width=True, config={'staticPlot': False})
    
    # Cluster Profiles
    st.markdown("### Detailed Cluster Profiles")
    
    with st.spinner("Analyzing cluster characteristics..."):
        profiles = analyzer.analyze_cluster_profiles(df, labels, feature_names)
    
    if profiles:
        # Initialize session state for cluster navigation
        if 'current_cluster_index' not in st.session_state:
            st.session_state['current_cluster_index'] = 0
        
        # Filter out noise cluster
        valid_interpretations = [i for i in profiles['interpretations'] if i['cluster'] != -1]
        total_clusters = len(valid_interpretations)
        
        # Pre-render all cluster figures to cache them
        if 'cluster_figures' not in st.session_state or len(st.session_state.get('cluster_figures', [])) != total_clusters:
            cluster_figures = []
            for interpretation in valid_interpretations:
                cluster_id = interpretation['cluster']
                categories = ['Deep Learning', 'Surface Learning', 'Strategic Learning']
                values = [
                    interpretation['deep_learning'],
                    interpretation['surface_learning'],
                    interpretation['strategic_learning']
                ]
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=f'Cluster {cluster_id}',
                    line=dict(color=colors[cluster_id % len(colors)], width=2),
                    fillcolor=colors[cluster_id % len(colors)],
                    opacity=0.6
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True, 
                            range=[0, max(values) * 1.2],
                            tickfont=dict(color='#000000'),
                            gridcolor='#cccccc'
                        ),
                        angularaxis=dict(
                            tickfont=dict(color='#000000'),
                            gridcolor='#cccccc'
                        ),
                        bgcolor='white'
                    ),
                    title=dict(
                        text=f'Learning Style Profile - Cluster {cluster_id}',
                        font=dict(color='#000000', size=14)
                    ),
                    template='plotly_white',
                    height=350,
                    font=dict(color='#000000'),
                    paper_bgcolor='white',
                    plot_bgcolor='white'
                )
                cluster_figures.append(fig)
            
            st.session_state['cluster_figures'] = cluster_figures
        
        if total_clusters > 0:
            # Simple navigation with < >
            nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])
            
            with nav_col1:
                if st.session_state['current_cluster_index'] > 0:
                    if st.button("◀", key="prev_cluster", use_container_width=True):
                        st.session_state['current_cluster_index'] -= 1
                        st.rerun()
            
            with nav_col2:
                st.markdown(f"""
                    <div style='text-align: center; padding: 0.5rem;'>
                        <span style='color: #000000; font-size: 1.1rem; font-weight: 600;'>
                            Cluster {st.session_state['current_cluster_index'] + 1} of {total_clusters}
                        </span>
                    </div>
                """, unsafe_allow_html=True)
            
            with nav_col3:
                if st.session_state['current_cluster_index'] < total_clusters - 1:
                    if st.button("▶", key="next_cluster", use_container_width=True):
                        st.session_state['current_cluster_index'] += 1
                        st.rerun()
            
            # Display current cluster
            interpretation = valid_interpretations[st.session_state['current_cluster_index']]
            cluster_id = interpretation['cluster']
            
            # Cluster header
            st.markdown(f"""
                <div class="custom-card animate-fade-in">
                    <div class="card-header">
                        <div class="card-icon"></div>
                        <span>Cluster {cluster_id} - {interpretation['dominant_style']}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Use pre-rendered figure from cache
                fig = st.session_state['cluster_figures'][st.session_state['current_cluster_index']]
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Key metrics
                st.markdown("**Key Metrics**")
                cluster_metrics = [
                    {"label": "Size", "value": interpretation['size'], "icon": "👥"},
                    {"label": "Avg IPK", "value": f"{float(np.mean(interpretation['ipk'])):.2f}", "icon": "📊"}
                ]
                render_metric_grid(cluster_metrics)
                
                st.markdown("**Recommendations**")
                for rec in interpretation['recommendations']:
                    st.markdown(f"- {rec}")
            
            st.markdown("<br>", unsafe_allow_html=True)
    
    # Download Section
    st.markdown("### Export Results")
    
    if profiles:
        results_df = profiles['data_with_clusters']
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name="clustering_results.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Show preview of results
            with st.expander("👁️ Preview Results Data"):
                st.dataframe(results_df.head(10), use_container_width=True)
