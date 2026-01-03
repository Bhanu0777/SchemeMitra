"""
🏛️ SchemeMitra - AI Government Scheme Finder
A modern web application to help Indians discover government schemes they are eligible for.
Built with Streamlit, Azure OpenAI, and Azure Text Analytics.
"""

import streamlit as st
import json
import os
from datetime import datetime
from dotenv import load_dotenv
import requests
from typing import List, Dict, Tuple

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="SchemeMitra - AI Government Scheme Finder",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'bookmarked_schemes' not in st.session_state:
    st.session_state.bookmarked_schemes = []

if 'language' not in st.session_state:
    st.session_state.language = 'en'

if 'accessibility_mode' not in st.session_state:
    st.session_state.accessibility_mode = False

if 'search_history' not in st.session_state:
    st.session_state.search_history = []

if 'expanded_schemes' not in st.session_state:
    st.session_state.expanded_schemes = []

# ============================================================================
# AZURE AI SERVICES CONFIGURATION
# ============================================================================

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-35-turbo")
AZURE_TEXTANALYTICS_KEY = os.getenv("AZURE_TEXTANALYTICS_KEY")
AZURE_TEXTANALYTICS_ENDPOINT = os.getenv("AZURE_TEXTANALYTICS_ENDPOINT")

# API version for Azure OpenAI
AZURE_OPENAI_API_VERSION = "2023-05-15"

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data
def load_schemes() -> List[Dict]:
    """Load schemes from JSON file."""
    try:
        with open('schemes.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('schemes', [])
    except FileNotFoundError:
        st.error("❌ schemes.json not found. Please ensure the file exists in the project directory.")
        return []

# Load schemes
SCHEMES = load_schemes()

# Categories mapping
CATEGORIES = {
    'Farmers': '🌾',
    'Women': '👩‍💼',
    'Youth': '👨‍🎓',
    'MSME': '🏭',
    'Education': '📚',
    'Senior Citizens': '👴'
}

CATEGORY_NAMES = list(CATEGORIES.keys())

# ============================================================================
# AZURE AI FUNCTIONS
# ============================================================================

def call_azure_openai(prompt: str, max_tokens: int = 200) -> str:
    """
    Call Azure OpenAI API to generate responses.
    Used for eligibility explanations and scheme matching.
    """
    if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT]):
        return "⚠️ Azure OpenAI not configured. Please set your API credentials in .env file."
    
    try:
        headers = {
            "Content-Type": "application/json",
            "api-key": AZURE_OPENAI_API_KEY
        }
        
        data = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that explains Indian government schemes in simple, non-legal language. Be concise and clear."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": max_tokens,
            "top_p": 0.95
        }
        
        url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
        
        response = requests.post(url, json=data, headers=headers, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    
    except requests.exceptions.RequestException as e:
        return f"⚠️ Error calling Azure OpenAI: {str(e)}"
    except Exception as e:
        return f"⚠️ Unexpected error: {str(e)}"

def analyze_text_azure(text: str) -> Dict:
    """
    Analyze user input using Azure Text Analytics.
    Extracts key entities and sentiment.
    """
    if not all([AZURE_TEXTANALYTICS_KEY, AZURE_TEXTANALYTICS_ENDPOINT]):
        return {"error": "Azure Text Analytics not configured"}
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": AZURE_TEXTANALYTICS_KEY
        }
        
        data = {
            "documents": [
                {
                    "id": "1",
                    "language": "en",
                    "text": text
                }
            ]
        }
        
        url = f"{AZURE_TEXTANALYTICS_ENDPOINT}/text/analytics/v3.1/entities/recognition/general"
        
        response = requests.post(url, json=data, headers=headers, timeout=10)
        response.raise_for_status()
        
        return response.json()
    
    except requests.exceptions.RequestException as e:
        return {"error": f"Error calling Azure Text Analytics: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

def generate_eligibility_explanation(scheme: Dict, user_profile: str) -> Tuple[str, int]:
    """
    Generate AI-powered eligibility explanation and match score.
    Returns: (explanation_text, match_score_percentage)
    """
    prompt = f"""
    Scheme Name: {scheme['name']}
    Ministry: {scheme['ministry']}
    Beneficiary Type: {scheme['beneficiary']}
    Benefit: {scheme['benefit']}
    
    User Profile: {user_profile}
    
    Based on the scheme details and user profile:
    1. Briefly explain (2-3 sentences) why this user MIGHT be eligible
    2. Mention any potential eligibility gaps
    3. Suggest next steps
    
    Keep language simple and non-legal.
    """
    
    explanation = call_azure_openai(prompt, max_tokens=150)
    
    # Calculate match score (in production, this would be more sophisticated)
    # For now, based on keyword matching
    match_score = calculate_match_score(scheme, user_profile)
    
    return explanation, match_score

def calculate_match_score(scheme: Dict, user_profile: str) -> int:
    """Calculate match percentage based on keyword matching."""
    scheme_text = f"{scheme['name']} {scheme['beneficiary']} {scheme['category']}".lower()
    profile_text = user_profile.lower()
    
    keywords = [
        'farmer', 'women', 'youth', 'student', 'senior', 'elder', 'msme', 'business',
        'entrepreneur', 'girl', 'female', 'young', 'old', 'small', 'enterprise'
    ]
    
    matches = sum(1 for keyword in keywords if keyword in scheme_text and keyword in profile_text)
    
    # Base score + keyword matches
    base_score = 50
    additional_score = matches * 5
    
    return min(95, base_score + additional_score)  # Cap at 95%

# ============================================================================
# UI STYLING & EMBEDDED CSS
# ============================================================================

def inject_css():
    """Inject custom CSS for modern UI with smooth animations."""
    css = """
    <style>
    /* ============================================================================
       GLOBAL STYLES
       ============================================================================ */
    
    :root {
        --primary-color: #1a3a52;
        --accent-color: #ff6b35;
        --accent-light: #f7931e;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --background: #f5f7fa;
        --card-bg: #ffffff;
        --text-dark: #333333;
        --text-light: #666666;
        --border-light: #e5e7eb;
        --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.1);
    }
    
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    html {
        scroll-behavior: smooth;
    }
    
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: var(--background);
        color: var(--text-dark);
        line-height: 1.6;
    }
    
    /* ============================================================================
       ACCESSIBILITY MODE
       ============================================================================ */
    
    .accessibility-mode {
        font-size: 18px !important;
    }
    
    .high-contrast {
        --primary-color: #000000;
        --accent-color: #ffff00;
        --background: #000000;
        --card-bg: #ffffff;
        --text-dark: #000000;
    }
    
    /* ============================================================================
       NAVIGATION BAR
       ============================================================================ */
    
    .navbar-container {
        background: linear-gradient(135deg, var(--primary-color) 0%, #2c5282 100%);
        padding: 1.5rem 2rem;
        border-bottom: 3px solid var(--accent-color);
        margin-bottom: 2rem;
        position: sticky;
        top: 0;
        z-index: 1000;
        transition: box-shadow 0.3s ease;
    }
    
    .navbar-container:hover {
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
    }
    
    .navbar-title {
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff;
        text-align: center;
        letter-spacing: 0.5px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .navbar-subtitle {
        font-size: 0.9rem;
        color: #e0e7ff;
        text-align: center;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* ============================================================================
       SEARCH SECTION
       ============================================================================ */
    
    .search-container {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: var(--shadow-md);
        margin-bottom: 2rem;
    }
    
    .search-input-wrapper {
        position: relative;
        margin-bottom: 1rem;
    }
    
    .search-input-wrapper input {
        width: 100%;
        padding: 14px 20px;
        font-size: 1rem;
        border: 2px solid var(--border-light);
        border-radius: 8px;
        outline: none;
        transition: all 0.3s ease;
    }
    
    .search-input-wrapper input:focus {
        border-color: var(--accent-color);
        box-shadow: 0 0 0 3px rgba(255, 107, 53, 0.15);
        transform: translateY(-2px);
    }
    
    .search-input-wrapper input::placeholder {
        color: var(--text-light);
    }
    
    /* ============================================================================
       BUTTONS WITH GRADIENT SWEEP
       ============================================================================ */
    
    .button-group {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        justify-content: center;
    }
    
    .btn {
        padding: 12px 24px;
        font-size: 0.95rem;
        font-weight: 600;
        border: none;
        border-radius: 6px;
        cursor: pointer;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .btn-primary {
        background: linear-gradient(90deg, var(--accent-color) 0%, var(--accent-light) 100%);
        color: white;
    }
    
    .btn-primary::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: rgba(255, 255, 255, 0.2);
        transition: left 0.5s ease;
    }
    
    .btn-primary:hover::before {
        left: 100%;
    }
    
    .btn-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(255, 107, 53, 0.3);
    }
    
    .btn-primary:active {
        transform: translateY(0);
    }
    
    .btn-secondary {
        background-color: var(--border-light);
        color: var(--text-dark);
    }
    
    .btn-secondary:hover {
        background-color: var(--text-light);
        color: white;
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }
    
    .btn-success {
        background-color: var(--success-color);
        color: white;
    }
    
    .btn-success:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(16, 185, 129, 0.3);
    }
    
    /* ============================================================================
       CATEGORY ICONS (WITH ROTATE & SCALE)
       ============================================================================ */
    
    .category-row {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .category-item {
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .category-icon {
        width: 70px;
        height: 70px;
        background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        margin: 0 auto 0.5rem;
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
        position: relative;
    }
    
    .category-icon::before {
        content: '';
        position: absolute;
        inset: -4px;
        background: radial-gradient(circle, rgba(255, 107, 53, 0.3), transparent);
        border-radius: 50%;
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .category-item:hover .category-icon {
        transform: scale(1.15) rotate(360deg);
        box-shadow: 0 8px 20px rgba(255, 107, 53, 0.4);
    }
    
    .category-item:hover .category-icon::before {
        opacity: 1;
    }
    
    .category-label {
        font-weight: 600;
        font-size: 0.85rem;
        color: var(--text-dark);
        transition: color 0.3s ease;
    }
    
    .category-item:hover .category-label {
        color: var(--accent-color);
    }
    
    /* ============================================================================
       FILTER PANEL
       ============================================================================ */
    
    .filter-panel {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: var(--shadow-sm);
        margin-bottom: 2rem;
    }
    
    .filter-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: var(--primary-color);
        margin-bottom: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .filter-group {
        margin-bottom: 1.2rem;
    }
    
    .filter-group:last-child {
        margin-bottom: 0;
    }
    
    select {
        width: 100%;
        padding: 10px 14px;
        border: 2px solid var(--border-light);
        border-radius: 6px;
        background-color: white;
        font-size: 0.95rem;
        color: var(--text-dark);
        cursor: pointer;
        transition: all 0.3s ease;
        appearance: none;
        padding-right: 30px;
        background-image: url("data:image/svg+xml;charset=UTF-8,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23333333' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3e%3cpolyline points='6 9 12 15 18 9'%3e%3c/polyline%3e%3c/svg%3e");
        background-repeat: no-repeat;
        background-position: right 10px center;
        background-size: 20px;
    }
    
    select:focus {
        border-color: var(--accent-color);
        box-shadow: 0 0 0 3px rgba(255, 107, 53, 0.1);
    }
    
    select:hover {
        border-color: var(--accent-color);
    }
    
    /* ============================================================================
       SCHEME CARDS (MAIN - WITH SHADOW LIFT & ZOOM)
       ============================================================================ */
    
    .scheme-card {
        background: white;
        border-left: 4px solid var(--accent-color);
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-md);
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
        position: relative;
    }
    
    .scheme-card:hover {
        transform: translateY(-6px) scale(1.02);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.15);
    }
    
    .scheme-header {
        display: flex;
        justify-content: space-between;
        align-items: start;
        margin-bottom: 1rem;
    }
    
    .scheme-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--primary-color);
        flex: 1;
        margin-right: 1rem;
    }
    
    .scheme-status {
        display: inline-block;
        padding: 6px 12px;
        background-color: var(--success-color);
        color: white;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        white-space: nowrap;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    .scheme-meta {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .meta-item {
        font-size: 0.9rem;
        color: var(--text-light);
    }
    
    .meta-label {
        font-weight: 600;
        color: var(--primary-color);
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    
    .meta-value {
        margin-top: 0.3rem;
        color: var(--text-dark);
    }
    
    .scheme-benefit {
        background: linear-gradient(135deg, rgba(255, 107, 53, 0.1), rgba(247, 147, 30, 0.1));
        padding: 1rem;
        border-left: 3px solid var(--accent-color);
        border-radius: 6px;
        margin: 1rem 0;
        font-weight: 500;
        color: var(--text-dark);
    }
    
    .match-score {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-top: 1rem;
        padding: 0.8rem;
        background: #f0f9ff;
        border-radius: 6px;
        font-weight: 600;
    }
    
    .match-score-bar {
        flex: 1;
        height: 8px;
        background: var(--border-light);
        border-radius: 4px;
        overflow: hidden;
    }
    
    .match-score-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--success-color), var(--accent-color));
        animation: fillBar 0.6s ease;
    }
    
    @keyframes fillBar {
        from { width: 0; }
        to { width: var(--match-percentage); }
    }
    
    .match-label {
        font-size: 0.8rem;
        color: var(--text-light);
    }
    
    /* ============================================================================
       EXPANDABLE SECTION (ELIGIBILITY)
       ============================================================================ */
    
    .expandable-section {
        margin-top: 1rem;
        border-top: 1px solid var(--border-light);
        padding-top: 1rem;
    }
    
    .expand-button {
        background: none;
        border: none;
        color: var(--accent-color);
        cursor: pointer;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        transition: all 0.3s ease;
    }
    
    .expand-button:hover {
        color: var(--accent-light);
        gap: 1rem;
    }
    
    .expand-icon {
        transition: transform 0.3s ease;
        display: inline-block;
    }
    
    .expand-icon.open {
        transform: rotate(180deg);
    }
    
    .eligibility-content {
        margin-top: 1rem;
        padding: 1rem;
        background: #f9fafb;
        border-radius: 6px;
        border-left: 3px solid var(--accent-color);
        animation: slideDown 0.3s ease;
    }
    
    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateY(-10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* ============================================================================
       BUTTON GROUPS IN CARDS
       ============================================================================ */
    
    .card-actions {
        display: flex;
        gap: 0.8rem;
        margin-top: 1.2rem;
        flex-wrap: wrap;
    }
    
    .btn-small {
        padding: 8px 16px;
        font-size: 0.85rem;
    }
    
    .btn-outline {
        background: transparent;
        border: 2px solid var(--accent-color);
        color: var(--accent-color);
        font-weight: 600;
    }
    
    .btn-outline:hover {
        background: var(--accent-color);
        color: white;
        transform: translateY(-2px);
    }
    
    .btn-bookmark {
        background: #fef3c7;
        color: #92400e;
        border: 2px solid #fcd34d;
    }
    
    .btn-bookmark.bookmarked {
        background: var(--accent-color);
        color: white;
        border-color: var(--accent-color);
    }
    
    .btn-source {
        background: linear-gradient(135deg, var(--primary-color), #2c5282);
        color: white;
    }
    
    .btn-source:hover {
        box-shadow: 0 6px 12px rgba(26, 58, 82, 0.3);
    }
    
    /* ============================================================================
       EMPTY STATE
       ============================================================================ */
    
    .empty-state {
        text-align: center;
        padding: 3rem 2rem;
        background: white;
        border-radius: 10px;
        box-shadow: var(--shadow-sm);
    }
    
    .empty-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .empty-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--primary-color);
        margin-bottom: 0.5rem;
    }
    
    .empty-message {
        color: var(--text-light);
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    
    /* ============================================================================
       DISCLAIMER & FOOTER
       ============================================================================ */
    
    .disclaimer {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1.2rem;
        border-radius: 6px;
        margin-bottom: 2rem;
        font-size: 0.9rem;
        color: #92400e;
    }
    
    .disclaimer-title {
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: var(--text-light);
        font-size: 0.9rem;
        border-top: 1px solid var(--border-light);
        margin-top: 3rem;
    }
    
    /* ============================================================================
       UNDERLINE REVEAL (using ::after)
       ============================================================================ */
    
    .underline-reveal {
        position: relative;
        display: inline-block;
    }
    
    .underline-reveal::after {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 0;
        width: 0;
        height: 2px;
        background-color: var(--accent-color);
        transition: width 0.3s ease;
    }
    
    .underline-reveal:hover::after {
        width: 100%;
    }
    
    /* ============================================================================
       RESPONSIVE DESIGN
       ============================================================================ */
    
    @media (max-width: 768px) {
        .navbar-title {
            font-size: 1.5rem;
        }
        
        .navbar-subtitle {
            font-size: 0.8rem;
        }
        
        .scheme-meta {
            grid-template-columns: 1fr;
        }
        
        .category-row {
            grid-template-columns: repeat(3, 1fr);
        }
        
        .scheme-header {
            flex-direction: column;
        }
        
        .scheme-status {
            margin-top: 0.5rem;
        }
        
        .search-container {
            padding: 1.5rem 1rem;
        }
    }
    
    @media (max-width: 480px) {
        .navbar-container {
            padding: 1rem;
        }
        
        .navbar-title {
            font-size: 1.3rem;
        }
        
        .button-group {
            gap: 0.5rem;
        }
        
        .btn {
            padding: 10px 16px;
            font-size: 0.85rem;
            flex: 1;
        }
        
        .category-row {
            grid-template-columns: repeat(2, 1fr);
        }
        
        .category-icon {
            width: 60px;
            height: 60px;
            font-size: 1.5rem;
        }
    }
    
    /* ============================================================================
       UTILITY CLASSES
       ============================================================================ */
    
    .text-center {
        text-align: center;
    }
    
    .mb-2 {
        margin-bottom: 1rem;
    }
    
    .mt-2 {
        margin-top: 1rem;
    }
    
    .hidden {
        display: none;
    }
    
    .flex-center {
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .badge {
        display: inline-block;
        padding: 4px 8px;
        background: var(--accent-color);
        color: white;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    
    .verified-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        color: var(--success-color);
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_navbar():
    """Render the top navigation bar."""
    navbar_html = """
    <div class="navbar-container">
        <div class="navbar-title">🏛️ SchemeMitra</div>
        <div class="navbar-subtitle">AI-Powered Government Scheme Finder</div>
    </div>
    """
    st.markdown(navbar_html, unsafe_allow_html=True)

def render_disclaimer():
    """Render the important disclaimer."""
    disclaimer_html = """
    <div class="disclaimer">
        <div class="disclaimer-title">⚠️ Important Disclaimer</div>
        <div>
            SchemeMitra is an independent application and is <strong>NOT an official government portal</strong>.
            This platform provides guidance only. Always verify information on official government portals.
            We are not responsible for inaccuracies. Consult official government offices for clarification.
        </div>
    </div>
    """
    st.markdown(disclaimer_html, unsafe_allow_html=True)

def render_category_selector():
    """Render category selector with circular icons."""
    col1, col2, col3 = st.columns([1, 1, 1])
    
    category_cols = {
        'Farmers': col1,
        'Women': col2,
        'Youth': col3,
        'MSME': col1,
        'Education': col2,
        'Senior Citizens': col3
    }
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <p style="font-weight: 700; color: #1a3a52; margin-bottom: 1rem;">Browse by Category</p>
    </div>
    """, unsafe_allow_html=True)
    
    for idx, category in enumerate(CATEGORY_NAMES):
        if idx % 3 == 0:
            col1, col2, col3 = st.columns([1, 1, 1])
        
        cols = [col1, col2, col3]
        with cols[idx % 3]:
            if st.button(f"{CATEGORIES[category]}\n{category}", key=f"cat_{category}", use_container_width=True):
                st.session_state.selected_category = category

def render_search_section():
    """Render the search section with input and filters."""
    st.markdown("""
    <div class="search-container">
        <p style="font-size: 1.1rem; font-weight: 700; color: #1a3a52; margin-bottom: 1rem;">
            🔍 Find Your Perfect Scheme
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        search_query = st.text_input(
            "Search by scheme name, ministry, or keyword...",
            key="search_input",
            placeholder="e.g., Farmer support, Women entrepreneur, Student scholarship..."
        )
    
    with col2:
        search_button = st.button("🔍 Search", use_container_width=True)
    
    return search_query, search_button

def render_filters():
    """Render filter panel."""
    st.markdown("""
    <div style="margin-bottom: 2rem; font-weight: 700; color: #1a3a52; font-size: 1.1rem;">
        ⚙️ Refine Your Search
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_ministry = st.selectbox(
            "Ministry / Department",
            options=["All Ministries"] + sorted(set(s['ministry'] for s in SCHEMES)),
            key="filter_ministry"
        )
    
    with col2:
        selected_beneficiary = st.selectbox(
            "Beneficiary Type",
            options=["All Types"] + sorted(set(s['beneficiary'] for s in SCHEMES)),
            key="filter_beneficiary"
        )
    
    with col3:
        selected_category = st.selectbox(
            "Category",
            options=["All Categories"] + CATEGORY_NAMES,
            key="filter_category"
        )
    
    return selected_ministry, selected_beneficiary, selected_category

def render_scheme_card(scheme: Dict, idx: int):
    """Render a single scheme card with all features."""
    is_bookmarked = scheme['id'] in st.session_state.bookmarked_schemes
    is_expanded = scheme['id'] in st.session_state.expanded_schemes
    
    # Create unique key for expand button
    expand_key = f"expand_{scheme['id']}"
    
    # Determine match score
    user_profile = st.session_state.get('last_user_profile', 'General user')
    _, match_score = generate_eligibility_explanation(scheme, user_profile)
    
    card_html = f"""
    <div class="scheme-card" id="scheme_{scheme['id']}">
        <div class="scheme-header">
            <div class="scheme-title">{scheme['name']}</div>
            <div class="scheme-status">✓ Active</div>
        </div>
        
        <div class="scheme-meta">
            <div class="meta-item">
                <div class="meta-label">📍 Ministry</div>
                <div class="meta-value">{scheme['ministry']}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">👥 Beneficiary</div>
                <div class="meta-value">{scheme['beneficiary']}</div>
            </div>
        </div>
        
        <div class="scheme-benefit">
            💰 <strong>Benefit:</strong> {scheme['benefit']}
        </div>
        
        <div class="match-score">
            <span class="match-label">Eligibility Match:</span>
            <div class="match-score-bar" style="--match-percentage: {match_score}%">
                <div class="match-score-fill"></div>
            </div>
            <span style="font-weight: 700; color: #10b981; min-width: 45px;">{match_score}%</span>
        </div>
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)
    
    # Expandable eligibility section
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("💡 Why I'm Eligible", key=f"expand_{scheme['id']}", use_container_width=True):
            if scheme['id'] in st.session_state.expanded_schemes:
                st.session_state.expanded_schemes.remove(scheme['id'])
            else:
                st.session_state.expanded_schemes.append(scheme['id'])
            st.rerun()
    
    with col2:
        if st.button(f"{'⭐ Bookmarked' if is_bookmarked else '☆ Bookmark'}", 
                     key=f"bookmark_{scheme['id']}", use_container_width=True):
            if is_bookmarked:
                st.session_state.bookmarked_schemes.remove(scheme['id'])
            else:
                st.session_state.bookmarked_schemes.append(scheme['id'])
            st.rerun()
    
    with col3:
        st.markdown(f"""
        <a href="{scheme['source_url']}" target="_blank" style="text-decoration: none; width: 100%; display: block;">
            <button class="btn btn-small btn-source" style="width: 100%; padding: 8px 16px; font-size: 0.85rem; font-weight: 600; border: none; border-radius: 6px; cursor: pointer; background: linear-gradient(135deg, #1a3a52, #2c5282); color: white; transition: all 0.3s ease;">
                🔗 Official Source
            </button>
        </a>
        """, unsafe_allow_html=True)
    
    # Show eligibility explanation if expanded
    if scheme['id'] in st.session_state.expanded_schemes:
        with st.spinner("✨ Generating AI explanation..."):
            explanation, _ = generate_eligibility_explanation(scheme, user_profile)
            
            eligibility_html = f"""
            <div class="eligibility-content">
                <strong>Why you might be eligible:</strong><br>
                {explanation}
            </div>
            """
            st.markdown(eligibility_html, unsafe_allow_html=True)
    
    st.divider()

def render_bookmarked_schemes():
    """Render bookmarked schemes section."""
    if st.session_state.bookmarked_schemes:
        st.markdown("""
        <div style="margin-bottom: 2rem; margin-top: 2rem;">
            <h2 style="color: #1a3a52; border-bottom: 3px solid #ff6b35; padding-bottom: 0.5rem;">
                ⭐ My Bookmarked Schemes
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        bookmarked = [s for s in SCHEMES if s['id'] in st.session_state.bookmarked_schemes]
        
        for scheme in bookmarked:
            render_scheme_card(scheme, 0)

def render_feedback_section():
    """Render feedback section."""
    st.markdown("""
    <div style="background: #f0f9ff; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #3b82f6; margin-top: 2rem;">
        <p style="font-weight: 700; color: #1a3a52; margin-bottom: 0.5rem;">📝 Was this helpful?</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("👍 Yes, very helpful!", key="feedback_yes", use_container_width=True):
            st.success("Thank you for your feedback! It helps us improve SchemeMitra.")
    
    with col2:
        if st.button("😐 Somewhat helpful", key="feedback_neutral", use_container_width=True):
            st.info("Thank you! We'll work on improving the experience.")
    
    with col3:
        if st.button("👎 Not helpful", key="feedback_no", use_container_width=True):
            st.warning("We'd love to hear your suggestions. Please reach out!")

def render_footer():
    """Render footer."""
    footer_html = """
    <div class="footer">
        <p><strong>🏛️ SchemeMitra</strong> © 2026 - AI Government Scheme Finder</p>
        <p>Built with ❤️ for the Imagine Cup | Data from official government portals</p>
        <p style="font-size: 0.8rem; margin-top: 1rem; color: #999;">
            This is an educational MVP and not officially affiliated with the Government of India.
        </p>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

# ============================================================================
# FILTERING & SEARCH LOGIC
# ============================================================================

def filter_schemes(schemes: List[Dict], 
                  search_query: str = "",
                  ministry_filter: str = "All Ministries",
                  beneficiary_filter: str = "All Types",
                  category_filter: str = "All Categories") -> List[Dict]:
    """
    Filter schemes based on search query and filters.
    """
    filtered = schemes.copy()
    
    # Search filter
    if search_query:
        search_lower = search_query.lower()
        filtered = [
            s for s in filtered
            if (search_lower in s['name'].lower() or
                search_lower in s['description'].lower() or
                search_lower in s['ministry'].lower() or
                search_lower in s['beneficiary'].lower())
        ]
    
    # Ministry filter
    if ministry_filter != "All Ministries":
        filtered = [s for s in filtered if s['ministry'] == ministry_filter]
    
    # Beneficiary filter
    if beneficiary_filter != "All Types":
        filtered = [s for s in filtered if s['beneficiary'] == beneficiary_filter]
    
    # Category filter
    if category_filter != "All Categories":
        filtered = [s for s in filtered if s['category'] == category_filter]
    
    return filtered

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    
    # Inject custom CSS
    inject_css()
    
    # Render navbar
    render_navbar()
    
    # Render disclaimer
    render_disclaimer()
    
    # Sidebar for settings
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # Language selector
        language = st.radio(
            "Language",
            options=["🇬🇧 English", "🇮🇳 Hindi"],
            key="language_select"
        )
        st.session_state.language = "en" if "English" in language else "hi"
        
        # Accessibility mode
        accessibility = st.checkbox(
            "♿ Accessibility Mode (Large Text + High Contrast)",
            value=st.session_state.accessibility_mode
        )
        st.session_state.accessibility_mode = accessibility
        
        st.divider()
        
        st.markdown("""
        ### 📱 About SchemeMitra
        
        AI-powered scheme discovery using:
        - **Azure OpenAI** for intelligent matching
        - **Text Analytics** for input analysis
        - **Real government data** from official portals
        
        ### 🔐 Privacy
        - No login required
        - No personal data stored
        - Session-based bookmarks only
        - No external tracking
        
        ### 📚 Need Help?
        - Check official scheme sources
        - Contact government offices directly
        - Review application disclaimers
        """)
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <p style="font-size: 1.1rem; color: #666666; font-weight: 500;">
                🎯 Discover government schemes tailored to your profile
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Search section
    search_query, search_button = render_search_section()
    
    # User profile input for AI analysis
    with st.expander("📋 Tell us about yourself (Optional - for better matching)", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=0, max_value=100, value=30)
        
        with col2:
            category = st.selectbox("Select your category", CATEGORY_NAMES + ["Other"])
        
        skills = st.text_input("Your skills/profession (optional)")
        
        # Create user profile for AI
        user_profile = f"{age} years old, {category} category"
        if skills:
            user_profile += f", skills: {skills}"
        
        st.session_state.last_user_profile = user_profile
    
    st.divider()
    
    # Category selector
    render_category_selector()
    
    # Filters
    selected_ministry, selected_beneficiary, selected_category = render_filters()
    
    st.divider()
    
    # Get selected category from button clicks
    selected_category = st.session_state.get('selected_category', selected_category)
    
    # Filter schemes
    filtered_schemes = filter_schemes(
        SCHEMES,
        search_query=search_query if search_button or search_query else "",
        ministry_filter=selected_ministry,
        beneficiary_filter=selected_beneficiary,
        category_filter=selected_category
    )
    
    # Display results
    st.markdown("""
    <div style="margin-bottom: 2rem; margin-top: 2rem;">
        <h2 style="color: #1a3a52; border-bottom: 3px solid #ff6b35; padding-bottom: 0.5rem;">
            🎯 Available Schemes
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    if filtered_schemes:
        st.markdown(f"""
        <div style="padding: 0.8rem; background: #f0f9ff; border-radius: 6px; margin-bottom: 1.5rem; text-align: center; font-weight: 600; color: #0369a1;">
            Found {len(filtered_schemes)} scheme(s) matching your criteria
        </div>
        """, unsafe_allow_html=True)
        
        for idx, scheme in enumerate(filtered_schemes, 1):
            render_scheme_card(scheme, idx)
    
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🔍</div>
            <div class="empty-title">No Schemes Found</div>
            <div class="empty-message">
                Try adjusting your filters or search terms. You can also:<br>
                • Remove filters to see all schemes<br>
                • Try different keywords<br>
                • Browse by category above
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Bookmarked schemes section
    if st.session_state.bookmarked_schemes:
        render_bookmarked_schemes()
    
    st.divider()
    
    # Feedback section
    render_feedback_section()
    
    st.divider()
    
    # Footer
    render_footer()

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
