"""
Custom CSS Styling for Clustering Analysis Tool - Clean Black & White UI
"""

CUSTOM_CSS = """
<style>
    /* Import Modern Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Hide Streamlit branding and default styles */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Global Styles */
    .stApp {
        background: #ffffff;
        font-family: 'Inter', sans-serif;
        color: #000000;
    }
    
    /* Main Container */
    .main .block-container {
        max-width: 1200px;
        padding: 2rem 2rem 2rem 2rem;
        background: #ffffff;
        margin: 0 auto;
        position: relative;
        padding-left: 180px;
    }
    
    /* Header Styling */
    .app-header {
        text-align: center;
        padding: 2rem 0 3rem 0;
        margin-bottom: 2rem;
    }
    
    .app-header h1 {
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -1px;
        color: #000000;
    }
    
    .app-header p {
        font-size: 1.1rem;
        color: #666666;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Card Styling */
    .custom-card {
        background: transparent;
        padding: 0;
        margin: 2rem 0 1rem 0;
    }
    
    .card-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #000000;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .card-icon {
        font-size: 1.4rem;
    }
    
    /* Step Indicator - Sidebar Style */
    .step-sidebar {
        position: fixed;
        left: 20px;
        top: 120px;
        width: 140px;
        z-index: 1000;
    }
    
    .step-item-sidebar {
        position: relative;
        margin-bottom: 3rem;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
    }
    
    .step-circle {
        width: 44px;
        height: 44px;
        min-width: 44px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        transition: all 0.3s ease;
        position: relative;
        z-index: 2;
        border: 2px solid #000000;
        background: rgba(0, 0, 0, 1);
    }
    
    .step-circle.active {
        background: rgba(0, 0, 0, 0.9);
        border-color: #000000;
    }
    
    .step-circle.completed {
        background: rgba(0, 0, 0, 0.8);
        border-color: #000000;
    }
    
    .step-circle.completed .step-icon {
        display: none;
    }
    
    .step-circle.completed::after {
        content: "✓";
        position: absolute;
        font-size: 1.2rem;
        color: #ffffff;
        font-weight: bold;
    }
    
    .step-circle.pending {
        background: rgba(0, 0, 0, 0.3);
        border-color: #cccccc;
    }
    
    .step-circle.pending .step-icon {
        opacity: 0.3;
    }
    
    .step-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #000000;
        margin-top: 0.6rem;
    }
    
    .step-label.pending {
        color: #999999;
    }
    
    /* Vertical Line Connector */
    .step-item-sidebar:not(:last-child)::before {
        content: "";
        position: absolute;
        left: 21px;
        top: 44px;
        width: 2px;
        height: calc(100% + 3rem - 44px);
        background: #e0e0e0;
        z-index: 1;
    }
    
    .step-item-sidebar.completed:not(:last-child)::before {
        background: #000000;
    }
    
    .step-item-sidebar.active:not(:last-child)::before {
        background: linear-gradient(to bottom, #000000 0%, #e0e0e0 100%);
    }
    
    /* Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .metric-card {
        background: #ffffff;
        color: #000000;
        padding: 1.25rem;
        border: 2px solid #000000;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        background: #f9f9f9;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .metric-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #666666;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 0.5rem;
        color: #000000;
    }
    
    /* Button Styling */
    .stButton > button {
        background: #000000 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px;
        padding: 0.875rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: #1a1a1a !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
    }
    
    .stButton > button:active {
        background: #000000 !important;
    }
    
    .stButton > button:focus:not(:active) {
        background: #000000 !important;
        color: #ffffff !important;
        border: none !important;
    }
    
    /* Navigation buttons (< >) - Simple Arrow Only */
    button[key="prev_cluster"],
    button[key="next_cluster"] {
        font-size: 2rem !important;
        padding: 0.25rem 0.5rem !important;
        min-width: 40px !important;
        background: transparent !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.2s ease !important;
        font-weight: 300 !important;
    }
    
    button[key="prev_cluster"]:hover,
    button[key="next_cluster"]:hover {
        background: transparent !important;
        color: #333333 !important;
        transform: scale(1.2) !important;
    }
    
    .stDownloadButton > button {
        background: #000000 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px;
        padding: 0.875rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    
    .stDownloadButton > button:hover {
        background: #1a1a1a !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Specific styling for template download and sample data buttons */
    button[key="download_template_btn"],
    button[key="load_sample_btn"],
    button[key="download_results_csv_btn"] {
        background: #000000 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px;
        padding: 0.875rem 1.5rem !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    
    button[key="download_template_btn"]:hover,
    button[key="load_sample_btn"]:hover,
    button[key="download_results_csv_btn"]:hover {
        background: #1a1a1a !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Input Styling */
    .stSelectbox, .stSlider, .stRadio, .stFileUploader {
        margin: 1rem 0;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        border-radius: 6px;
        border: 2px solid #000000;
        background: #ffffff;
        color: #000000 !important;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #000000;
        box-shadow: 0 0 0 3px rgba(0, 0, 0, 0.1);
    }
    
    .stSelectbox label {
        color: #000000 !important;
        font-weight: 600;
    }
    
    .stSelectbox [data-baseweb="select"] {
        color: #000000 !important;
    }
    
    .stSelectbox div[data-baseweb="select"] > div {
        color: #000000 !important;
    }
    
    /* Radio buttons */
    .stRadio > label {
        color: #000000 !important;
        font-weight: 600;
    }
    
    .stRadio > div {
        color: #000000 !important;
    }
    
    .stRadio > div > label {
        color: #000000 !important;
    }
    
    .stRadio > div > label > div {
        color: #000000 !important;
    }
    
    .stRadio > div > label > div[data-testid="stMarkdownContainer"] {
        color: #000000 !important;
    }
    
    .stRadio > div > label > div[data-testid="stMarkdownContainer"] p {
        color: #000000 !important;
    }
    
    .stRadio label span {
        color: #000000 !important;
    }
    
    /* Radio button circle */
    .stRadio [role="radio"] {
        border-color: #000000 !important;
    }
    
    .stRadio [role="radio"][aria-checked="true"]::before {
        background-color: #000000 !important;
    }
    
    /* Checkbox */
    .stCheckbox > label {
        color: #000000 !important;
        font-weight: 600;
    }
    
    .stCheckbox label {
        color: #000000 !important;
    }
    
    .stCheckbox label span {
        color: #000000 !important;
    }
    
    .stCheckbox [data-testid="stMarkdownContainer"] {
        color: #000000 !important;
    }
    
    .stCheckbox [data-testid="stMarkdownContainer"] p {
        color: #000000 !important;
    }
    
    /* Checkbox box */
    .stCheckbox [role="checkbox"] {
        border-color: #000000 !important;
    }
    
    .stCheckbox [role="checkbox"][aria-checked="true"] {
        background-color: #000000 !important;
        border-color: #000000 !important;
    }
    
    /* Slider */
    .stSlider > label {
        color: #000000 !important;
        font-weight: 600;
    }
    
    .stSlider [data-baseweb="slider"] {
        background: transparent;
    }
    
    /* Slider track (garis) */
    .stSlider [data-baseweb="slider"] > div > div {
        background: #000000 !important;
    }
    
    /* Slider thumb (bulatan) */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #000000 !important;
        border: 2px solid #000000 !important;
    }
    
    .stSlider .st-emotion-cache-1inwz65 {
        background: #000000 !important;
    }
    
    /* Override any other slider colors */
    .stSlider div[role="slider"]::before {
        background-color: #000000 !important;
    }
    
    /* Slider value display - no background */
    .stSlider [data-testid="stTickBar"] {
        background: transparent !important;
    }
    
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] {
        background: transparent !important;
        color: #000000 !important;
    }
    
    /* Slider value text */
    .stSlider [data-baseweb="slider"] > div:last-child {
        background: transparent !important;
    }
    
    .stSlider [data-baseweb="slider"] > div:last-child > div {
        background: transparent !important;
        color: #000000 !important;
    }
    
    /* File Uploader */
    .stFileUploader > label {
        color: #000000 !important;
        font-weight: 600;
    }
    
    .stFileUploader [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #cccccc !important;
        border-radius: 8px;
        background: #f5f5f5 !important;
    }
    
    .stFileUploader [data-testid="stFileUploadDropzone"]:hover {
        border-color: #999999 !important;
        background: #eeeeee !important;
    }
    
    .stFileUploader [data-testid="stFileUploadDropzone"] section {
        border: none !important;
        background: transparent !important;
    }
    
    .stFileUploader [data-testid="stFileUploadDropzone"] small {
        color: #666666 !important;
    }
    
    /* Tombol "Browse files" di dalam file uploader */
    .stFileUploader [data-testid="stFileUploadDropzone"] button {
        background-color: #000000 !important;
        color: #ffffff !important; /* Teks menjadi putih */
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
    }
    
    .stFileUploader [data-testid="stFileUploadDropzone"] button:hover {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }

    /* Tombol 'X' (hapus file) kembali ke normal */
    .stFileUploader [data-testid="stFileUploaderDeleteBtn"] {
        background-color: transparent !important;
        color: #31333F !important; /* Warna ikon abu-abu default */
        border: none !important;
        box-shadow: none !important;
    }

    .stFileUploader [data-testid="stFileUploaderDeleteBtn"]:hover {
        background-color: transparent !important;
        color: #FF4B4B !important; /* Warna ikon merah saat hover (standar Streamlit) */
    }

    /* Memastikan ikon SVG di dalam tombol X juga mengikuti warna */
    .stFileUploader [data-testid="stFileUploaderDeleteBtn"] svg {
        fill: currentColor !important;
    }
    
    /* File uploader file name area */
    .stFileUploader [data-testid="stFileUploaderFile"] {
        background: transparent !important;
    }
    
    .stFileUploader [data-testid="stFileUploaderFileName"] {
        color: #000000 !important;
    }
    
    /* Alert Boxes */
    .stAlert {
        border-radius: 8px;
        border: 2px solid #000000;
        padding: 1rem 1.5rem;
        background: #ffffff;
        color: #000000;
    }
    
    /* Success Message */
    .success-box {
        background: #ffffff;
        color: #000000;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 2px solid #000000;
        font-weight: 500;
    }
    
    /* Info Box */
    .info-box {
        background: #f9f9f9;
        color: #000000;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 2px solid #000000;
        font-weight: 500;
    }
    
    /* Warning Box */
    .warning-box {
        background: #ffffff;
        color: #000000;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 2px solid #000000;
        font-weight: 600;
    }
    
    /* DataFrames */
    .dataframe {
        border: 2px solid #000000;
        border-radius: 8px;
        overflow: hidden;
    }
    
    [data-testid="stDataFrame"] {
        border: 2px solid #000000;
        border-radius: 8px;
    }
    
    /* Expander */
    /* Header */
    .streamlit-expanderHeader {
        background: #ffffff;
        border: 2px solid #000000;
        border-radius: 8px;
        padding: 1rem;
        font-weight: 600;
        color: #000000;
    }
    .streamlit-expanderHeader:hover {
        background: #f9f9f9;
    }
    /* Ensure header doesn't turn dark on focus/active in dark themes */
    .streamlit-expanderHeader:focus,
    .streamlit-expanderHeader:active {
        background: #ffffff !important;
        color: #000000 !important;
    }

    /* Expander container/content (fix black background when open on some themes) */
    [data-testid="stExpander"],
    [data-testid="stExpander"] > details,
    [data-testid="stExpander"] > details[open],
    [data-testid="stExpander"] > details > summary,
    [data-testid="stExpander"] .streamlit-expanderContent {
        background: #ffffff !important;
        color: #000000 !important;
    }
    /* Content area wrapper */
    [data-testid="stExpander"] .streamlit-expanderContent > div {
        background: #ffffff !important;
    }
    
    /* Tab Styling - Hide default tabs, use custom */
    .stTabs {
        display: none;
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: #000000;
        border-radius: 4px;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #000000 !important;
    }
    
    /* Section Divider */
    .section-divider {
        height: 1px;
        background: #e0e0e0;
        margin: 3rem 0;
    }
    
    /* Cluster Badge */
    .cluster-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 600;
        margin: 0.25rem;
        background: #000000;
        color: #ffffff;
        border: 2px solid #000000;
    }
    
    /* Results Grid */
    .results-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    /* Plotly Chart Container */
    .js-plotly-plot {
        border: 2px solid #000000;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Markdown text */
    .stMarkdown {
        color: #000000;
    }
    
    /* Labels */
    label {
        color: #000000 !important;
        font-weight: 600;
    }
    
    /* Text inputs */
    .stTextInput > div > div > input {
        border: 2px solid #000000;
        border-radius: 6px;
        color: #000000;
    }
    
    /* Text areas */
    .stTextArea > div > div > textarea {
        border: 2px solid #000000;
        border-radius: 6px;
        color: #000000;
    }
    
    /* Smooth Animations */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f0f0f0;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #000000;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #333333;
    }
    
    /* Remove all system colors */
    .stAlert[data-baseweb="notification"] {
        background-color: #ffffff !important;
        border: 2px solid #000000 !important;
    }
    
    /* Spinner color */
    .stSpinner > div > div {
        border-top-color: #000000 !important;
    }
    
    /* Override any blue/red/green system colors */
    [class*="st-"] [class*="primary"] {
        color: #000000 !important;
    }
    
    /* Info/Success/Warning/Error messages */
    .stSuccess, .stInfo, .stWarning, .stError {
        background-color: #ffffff !important;
        border: 2px solid #000000 !important;
        color: #000000 !important;
    }
    
    /* Remove focus outline colors */
    *:focus {
        outline-color: #000000 !important;
    }
    
    /* Input focus state */
    input:focus, textarea:focus, select:focus {
        border-color: #000000 !important;
        box-shadow: 0 0 0 2px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Remove any remaining colored elements */
    [class*="accent"], [class*="primary"], [class*="secondary"] {
        color: #000000 !important;
    }
    
    /* Tombol "Browse files" di dalam file uploader */
    .stFileUploader [data-testid="stFileUploadDropzone"] button {
        background-color: #000000 !important;
        color: #ffffff !important; /* Pastikan teks berwarna putih */
        border: none !important;
    }
    
    .stFileUploader [data-testid="stFileUploadDropzone"] button:hover {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }

    /* Mengembalikan tombol 'X' (hapus file) ke keadaan normal (transparan) tanpa efek hover */
    .stFileUploader [data-testid="stFileUploaderDeleteBtn"] {
        background-color: transparent !important;
        color: #000000 !important; /* Warna ikon */
    }

    .stFileUploader [data-testid="stFileUploaderDeleteBtn"]:hover {
        background-color: transparent !important; /* Tidak ada perubahan background saat hover */
    }
</style>
"""
