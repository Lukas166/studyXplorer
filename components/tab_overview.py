"""
Data Overview Component - Modern UI
"""

import streamlit as st
import numpy as np
from components.ui_components import render_metric_grid


def render_data_overview(df):
    """Render data overview with modern styling"""
    
    # Dataset metrics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    metrics = [
        {"label": "Total Rows", "value": f"{df.shape[0]:,}", "icon": "📋"},
        {"label": "Total Columns", "value": df.shape[1], "icon": "📋"},
        {"label": "Numeric Features", "value": len(numeric_cols), "icon": "🔢"},
        {"label": "Missing Values", "value": df.isnull().sum().sum(), "icon": "❓"}
    ]
    
    render_metric_grid(metrics)
    
    # Dataset preview
    st.markdown("### Dataset Preview")
    
    # Anonymize name column for preview
    df_preview = df.head(10).copy()
    # Find column that contains "nama" (case insensitive)
    name_cols = [col for col in df_preview.columns if 'nama' in col.lower()]
    for col in name_cols:
        df_preview[col] = 'Anonim'
    
    st.dataframe(df_preview, use_container_width=True, height=300)
    
    # Basic statistics
    if len(numeric_cols) > 0:
        st.markdown("### Statistical Summary")
        st.dataframe(df[numeric_cols].describe(), use_container_width=True)
