import streamlit as st
import streamlit.components.v1 as components
import joblib
import numpy as np
import re
import textwrap

# ==========================================
# 1. PAGE CONFIGURATION & ULTRA-MODERN 3D CSS
# ==========================================
st.set_page_config(page_title="AutoJudge Pro", page_icon="⚡", layout="centered")

st.markdown("""
<style>
    /* --- 1. Animated Deep Space Background --- */
    .stApp {
        background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #1e3c72);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: #e0e0e0;
    }
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* --- 2. Glassmorphism 3D Container Styles --- */
    /* This targets the container wrapping the inputs */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        background: rgba(255, 255, 255, 0.1); /* Semi-transparent white */
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3); /* Deep shadow for floating effect */
        backdrop-filter: blur(12px); /* FROSTED GLASS EFFECT */
        -webkit-backdrop-filter: blur(12px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.18); /* Subtle edge */
        padding: 25px;
        transition: all 0.4s ease;
    }
    
    /* Hover Lift Effect */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"]:hover {
        transform: translateY(-5px) scale(1.01);
        box-shadow: 0 15px 45px 0 rgba(0, 0, 0, 0.4);
    }

    /* --- 3. Input Field Styling (Glass Insets) --- */
    .stTextInput > label, .stTextArea > label {
        color: #ffffff !important;
        font-weight: 600;
        letter-spacing: 1px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Making inputs look like inset glass */
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: rgba(0, 0, 0, 0.2) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px;
    }
    .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
        border: 1px solid rgba(255, 126, 95, 0.8) !important;
        box-shadow: inset 0 0 10px rgba(255, 126, 95, 0.3);
    }

    /* --- 4. Tactile 3D Button --- */
    .stButton > button {
        background: linear-gradient(to right, #ff512f, #dd2476);
        color: white;
        font-size: 20px;
        font-weight: bold;
        border: none;
        border-radius: 12px;
        width: 100%;
        padding: 15px;
        /* The 3D "thickness" */
        border-bottom: 4px solid #9e1a4d;
        box-shadow: 0 10px 20px -10px rgba(221, 36, 118, 0.5);
        transition: all 0.1s ease-in-out;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    /* 3D Press Animation */
    .stButton > button:active {
        transform: translateY(3px);
        border-bottom: 1px solid #9e1a4d; /* Shrink the border */
        box-shadow: 0 5px 10px -10px rgba(221, 36, 118, 0.5);
    }
    
    /* Main Titles */
    h1 {
        text-shadow: 0 4px 8px rgba(0,0,0,0.5);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD TRAINED MODELS
# ==========================================
@st.cache_resource
def load_engine():
    try:
        # Matches the filenames from your train.py
        clf = joblib.load('classifier.pkl')
        reg = joblib.load('regressor.pkl')
        tfidf = joblib.load('tfidf.pkl')
        scaler = joblib.load('scaler.pkl')
        return clf, reg, tfidf, scaler
    except FileNotFoundError:
        return None, None, None, None

classifier, regressor, tfidf, scaler = load_engine()

if not classifier:
    st.error("⚠️ Critical Error: Model files not found. Please run 'train_model.py' first.")
    st.stop()

# ==========================================
# 3. PREPROCESSING
# ==========================================
def sanitize_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    return text

def extract_features(title, desc, inp, out):
    full_text = f"{title} {desc} {inp} {out}"
    clean_text = sanitize_text(full_text)
    
    text_vector = tfidf.transform([clean_text]).toarray()
    word_count = len(clean_text.split())
    len_vector = scaler.transform([[word_count]])
    
    return np.hstack((text_vector, len_vector))

# ==========================================
# 4. UI LAYOUT
# ==========================================
st.markdown(
    "<h1 style='text-align:center;'>⚡ AutoJudge Pro</h1>"
    "<p style='text-align:center; color:#a0a0a0; font-size: 1.2rem;'>AI-Powered CP Problem Classifier</p>",
    unsafe_allow_html=True
)

# The CSS automatically applies the Glassmorphism effect to this container block
with st.container():
    p_title = st.text_input("Problem Title", placeholder="e.g. Dijkstra's Shortest Path")
    p_desc = st.text_area("Problem Statement", height=150, placeholder="Paste description here...")
    
    col1, col2 = st.columns(2)
    with col1:
        p_in = st.text_area("Input Spec", height=80)
    with col2:
        p_out = st.text_area("Output Spec", height=80)

st.write("") # Spacer

# ==========================================
# 5. ACTION LOGIC
# ==========================================
if st.button("🚀 Analyze Difficulty"):
    
    if not p_desc:
        st.warning("Please provide a problem description.")
        st.stop()
        
    safe_title = p_title if p_title else "Untitled Problem"
    
    # 1. Inference
    features = extract_features(safe_title, p_desc, p_in, p_out)
    pred_class = classifier.predict(features)[0]
    pred_score = int(np.clip(regressor.predict(features)[0], 0, 3500))
    
    # 2. Display Logic
    display_class = pred_class.title()
    
    # Brighter colors to pop against the dark glass background
    color_map = {
        "Easy": "#00ff88",   # Neon Green
        "Medium": "#ffaa00", # Neon Orange
        "Hard": "#ff0055"    # Neon Red
    }
    accent_color = color_map.get(display_class, "#ffffff")

    # 3. Gauge Math
    percentage = pred_score / 3500
    fill_angle = int(percentage * 180)

    # 4. Render HTML Card (Updated for Glassmorphism)
    result_html = f"""
    <div style="
        background: rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 30px; 
        text-align:center; 
        font-family:'Helvetica Neue', sans-serif;
        color: white;
        transition: transform 0.3s ease;
        "
        onmouseover="this.style.transform='translateY(-5px) scale(1.01)'"
        onmouseout="this.style.transform='translateY(0px) scale(1)'"
        >
        
        <h3 style="color:#b0b0b0; margin:0; text-transform:uppercase; letter-spacing:1px; font-size:0.9rem;">Analysis Report</h3>
        <h2 style="color:white; margin-top:10px; text-shadow: 0 2px 4px rgba(0,0,0,0.5);">{safe_title}</h2>
        <hr style="border-color: rgba(255,255,255,0.2);">
        
        <div style="display:flex; justify-content:space-around; align-items:center; margin-top:25px;">
            
            <div>
                <span style="color:#b0b0b0; font-weight:600;">DIFFICULTY</span><br>
                <span style="font-size:3.5rem; font-weight:800; color:{accent_color}; text-shadow: 0 0 15px {accent_color}80;">
                    {display_class}
                </span>
            </div>
            
            <div style="height:80px; border-left:2px solid rgba(255,255,255,0.1);"></div>
            
            <div style="width:160px;">
                <span style="color:#b0b0b0; font-weight:600;">RATING</span>
                
                <div style="position:relative; width:160px; height:80px; overflow:hidden; margin:15px auto 0;">
                    
                    <div style="position:absolute; width:160px; height:160px; 
                                border-radius:50%; background:rgba(255,255,255,0.1);"></div>
                                
                    <div style="position:absolute; width:160px; height:160px; 
                                border-radius:50%;
                                background:conic-gradient(from 270deg, 
                                    {accent_color} 0deg, 
                                    {accent_color} {fill_angle}deg, 
                                    transparent {fill_angle}deg
                                ); 
                                box-shadow: 0 0 20px {accent_color}60;"></div>
                    
                    <div style="position:absolute; width:120px; height:120px; 
                                background:rgba(0,0,0,0.4); /* Darker inner circle */
                                backdrop-filter: blur(5px);
                                border-radius:50%; 
                                top:20px; left:20px;
                                border: 1px solid rgba(255,255,255,0.1);"></div>
                    
                    <div style="position:absolute; width:100%; top:45px; 
                                text-align:center; font-size:1.8rem; 
                                font-weight:bold; color:white;
                                text-shadow: 0 2px 4px rgba(0,0,0,0.5);">
                        {pred_score}
                    </div>
                </div>
            </div>
        </div>
    </div>
    """
    
    components.html(textwrap.dedent(result_html), height=380)