"""
Modern UI Components Helper
"""

import streamlit as st


def render_header():
    """Render modern app header"""
    st.markdown("""
        <div class="app-header animate-fade-in">
            <h1>Study Xplorer - Clustering Analysis</h1>
            <p>Analisis clustering profesional untuk data mahasiswa dengan visualisasi interaktif</p>
        </div>
    """, unsafe_allow_html=True)


def render_step_indicator(current_step):
    """
    Render step indicator as sidebar
    current_step: 1 (Upload), 2 (Preprocess), 3 (Cluster), 4 (Results)
    """
    steps = [
        {"num": 1, "label": "Upload", "icon": "📁"},
        {"num": 2, "label": "Preprocess", "icon": "🔧"},
        {"num": 3, "label": "Cluster", "icon": "✨"},
        {"num": 4, "label": "Results", "icon": "📄"}
    ]
    
    html = '<div class="step-sidebar">'
    
    for i, step in enumerate(steps):
        # Determine state
        if step["num"] < current_step:
            state = "completed"
        elif step["num"] == current_step:
            state = "active"
        else:
            state = "pending"
        
        label_class = "pending" if state == "pending" else ""
        
        # Step item with vertical connector - one line, no extra whitespace
        html += f'<div class="step-item-sidebar {state}"><div class="step-circle {state}"><span class="step-icon">{step["icon"]}</span></div><div class="step-label {label_class}">{step["label"]}</div></div>'
    
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_card(title, icon, content_func):
    """
    Render a custom card
    content_func: function to render card content
    """
    st.markdown(f"""
        <div class="custom-card animate-fade-in">
            <div class="card-header">
                <div class="card-icon">{icon}</div>
                <span>{title}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    content_func()


def render_metric_card(label, value, icon="📊"):
    """Render a single metric card"""
    return f"""<div class="metric-card">
        <div class="metric-label">{icon} {label}</div>
        <div class="metric-value">{value}</div>
    </div>"""


def render_metric_grid(metrics):
    """
    Render metrics in a grid
    metrics: list of dicts with 'label', 'value', 'icon'
    """
    html = '<div class="metric-grid">'
    for metric in metrics:
        html += render_metric_card(
            metric.get('label', ''),
            metric.get('value', ''),
            metric.get('icon', '📊')
        )
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_success(message):
    """Render success message"""
    st.markdown(f"""
        <div class="success-box animate-fade-in">
            ✓ {message}
        </div>
    """, unsafe_allow_html=True)


def render_info(message):
    """Render info message"""
    st.markdown(f"""
        <div class="info-box animate-fade-in">
            {message}
        </div>
    """, unsafe_allow_html=True)


def render_warning(message):
    """Render warning message"""
    st.markdown(f"""
        <div class="warning-box animate-fade-in">
            ⚠️ {message}
        </div>
    """, unsafe_allow_html=True)


def render_section_divider():
    """Render a visual section divider"""
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


def render_cluster_badge(cluster_num, count=None):
    """Render a cluster badge"""
    text = f"Cluster {cluster_num}"
    if count:
        text += f" ({count} samples)"
    return f'<span class="cluster-badge">{text}</span>'
