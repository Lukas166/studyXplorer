"""
Preprocessing Component - Modern UI
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from components.ui_components import render_metric_grid


def render_preprocessing(preprocessor, df, scaling_method, dr_method, dr_components, 
                         n_components, dr_params):
    """Render preprocessing results with modern styling"""
    
    # Preprocessing metrics
    dr_info = preprocessor.dr_info
    
    # Get feature names from the preprocessor's stored data
    if hasattr(preprocessor, 'feature_names') and preprocessor.feature_names is not None:
        n_features = len(preprocessor.feature_names)
    else:
        # Count numeric columns from original dataframe
        n_features = len(df.select_dtypes(include=['number']).columns)
    
    metrics = [
        {"label": "Original Features", "value": n_features, "icon": "📊"},
        {"label": "Scaling Method", "value": scaling_method, "icon": "⚖️"},
        {"label": f"{dr_info['method']} Components", "value": dr_info['n_components'], "icon": "🔢"}
    ]
    
    if dr_info['explained_variance'] is not None:
        # Use explained_variance_ratio (already normalized to sum to 1)
        cumulative_variance = dr_info['cumulative_variance']
        metrics.append({
            "label": "Variance Explained",
            "value": f"{cumulative_variance[-1]:.1%}",
            "icon": "📈"
        })
    
    render_metric_grid(metrics)
    
    # Variance plot for methods that support it
    if dr_info['explained_variance'] is not None:
        st.markdown("### Variance Analysis")
        
        fig = go.Figure()
        
        # Individual variance
        fig.add_trace(go.Bar(
            x=[f'PC{i+1}' for i in range(dr_info['n_components'])],
            y=dr_info['explained_variance'],
            name='Individual Variance',
            marker_color='rgb(102, 126, 234)',
            text=[f"{v:.1%}" for v in dr_info['explained_variance']],
            textposition='outside'
        ))
        
        # Cumulative variance
        fig.add_trace(go.Scatter(
            x=[f'PC{i+1}' for i in range(dr_info['n_components'])],
            y=dr_info['cumulative_variance'],
            name='Cumulative Variance',
            yaxis='y2',
            line=dict(color='rgb(118, 75, 162)', width=3),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title=f'{dr_info["method"]} - Explained Variance Ratio',
            xaxis_title='Principal Components',
            yaxis_title='Individual Variance',
            yaxis2=dict(
                title='Cumulative Variance',
                overlaying='y',
                side='right',
                range=[0, 1]
            ),
            showlegend=True,
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"**{dr_info['method']}** does not provide explained variance information")
    
    return None, None, None  # Not needed anymore as we store in session_state
