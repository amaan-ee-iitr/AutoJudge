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

# Inject comprehensive 3D CSS directly
st.markdown("""
<style>
    /* === ANIMATED 3D BACKGROUND === */
    .stApp {
        background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #1e3c72) !important;
        background-size: 400% 400% !important;
        animation: gradientBG 15s ease infinite !important;
        color: #e0e0e0 !important;
        perspective: 1000px !important;
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes float3D {
        0%, 100% { transform: translateY(0px) rotateX(0deg); }
        50% { transform: translateY(-8px) rotateX(2deg); }
    }
    
    @keyframes glowPulse {
        0%, 100% { box-shadow: 0 0 20px rgba(255, 126, 95, 0.4); }
        50% { box-shadow: 0 0 30px rgba(255, 126, 95, 0.6); }
    }
    
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    @keyframes textGlow {
        0%, 100% { text-shadow: 0 4px 8px rgba(0,0,0,0.5), 0 0 20px rgba(255, 126, 95, 0.3); }
        50% { text-shadow: 0 4px 8px rgba(0,0,0,0.5), 0 0 30px rgba(255, 126, 95, 0.5); }
    }
    
    /* === MAIN CONTAINER 3D EFFECT === */
    .main .block-container {
        transform-style: preserve-3d !important;
        perspective: 1000px !important;
    }
    
    /* === GLASSMORPHISM CONTAINER - Target Streamlit's structure === */
    .main .block-container > div:first-of-type {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(20px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
        border-radius: 24px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        padding: 30px !important;
        margin: 20px 0 !important;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.4),
            inset 0 1px 1px rgba(255, 255, 255, 0.2) !important;
        animation: float3D 6s ease-in-out infinite !important;
        transition: all 0.5s ease !important;
        position: relative !important;
    }
    
    .main .block-container > div:first-of-type::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        height: 2px !important;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.5), transparent) !important;
        border-radius: 24px 24px 0 0 !important;
    }
    
    /* === INPUT FIELDS 3D STYLING === */
    .stTextInput label,
    .stTextArea label {
        color: #ffffff !important;
        font-weight: 700 !important;
        letter-spacing: 1.5px !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5) !important;
    }
    
    .stTextInput input,
    .stTextArea textarea {
        background: rgba(0, 0, 0, 0.3) !important;
        color: white !important;
        border: 2px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px !important;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput input:focus,
    .stTextArea textarea:focus {
        border: 2px solid rgba(255, 126, 95, 0.8) !important;
        box-shadow: 
            inset 0 2px 8px rgba(255, 126, 95, 0.3),
            0 0 20px rgba(255, 126, 95, 0.4) !important;
        outline: none !important;
    }
    
    /* === 3D BUTTON === */
    .stButton > button {
        background: linear-gradient(135deg, #ff512f 0%, #dd2476 50%, #ff512f 100%) !important;
        background-size: 200% 200% !important;
        animation: gradientShift 3s ease infinite !important;
        color: white !important;
        font-size: 20px !important;
        font-weight: bold !important;
        border: none !important;
        border-radius: 16px !important;
        width: 100% !important;
        padding: 18px !important;
        border-bottom: 5px solid #9e1a4d !important;
        box-shadow: 
            0 10px 30px rgba(221, 36, 118, 0.6),
            inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        transition: all 0.2s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stButton > button::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: -100% !important;
        width: 100% !important;
        height: 100% !important;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent) !important;
        transition: left 0.5s !important;
    }
    
    .stButton > button:hover::before {
        left: 100% !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 
            0 15px 40px rgba(221, 36, 118, 0.8),
            0 0 30px rgba(255, 126, 95, 0.4) !important;
        animation: glowPulse 2s ease-in-out infinite !important;
    }
    
    .stButton > button:active {
        transform: translateY(4px) scale(0.98) !important;
        border-bottom: 2px solid #9e1a4d !important;
    }
    
    /* === 3D TEXT EFFECTS === */
    h1 {
        text-shadow: 0 4px 8px rgba(0,0,0,0.5), 0 0 20px rgba(255, 126, 95, 0.3) !important;
        animation: textGlow 3s ease-in-out infinite !important;
    }
    
    /* === 3D SCROLLBAR === */
    ::-webkit-scrollbar {
        width: 10px !important;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.2) !important;
        border-radius: 10px !important;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, rgba(255, 126, 95, 0.6), rgba(221, 36, 118, 0.6)) !important;
        border-radius: 10px !important;
        box-shadow: inset 0 0 6px rgba(0, 0, 0, 0.3) !important;
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
    """
    <div style="text-align:center; margin-bottom:30px; transform-style: preserve-3d;">
        <h1 style='
            text-align:center;
            background: linear-gradient(135deg, #ffffff 0%, #ff7e5f 50%, #ffffff 100%);
            background-size: 200% 200%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: titleGradient 3s ease infinite;
            transform: translateZ(30px);
            margin-bottom:10px;
            letter-spacing: 2px;
        '>⚡ AutoJudge Pro</h1>
        <p style='
            text-align:center; 
            color:#a0a0a0; 
            font-size: 1.2rem;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
            transform: translateZ(20px);
            letter-spacing: 1px;
        '>AI-Powered CP Problem Classifier</p>
    </div>
    <style>
        @keyframes titleGradient {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Input fields with 3D glass styling
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
    pred_score = round(np.clip(regressor.predict(features)[0], 0, 3500) / 100) * 100
    
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

    # 4. Render HTML Card with Advanced 3D Effects
    result_html = f"""
    <div id="resultCard" style="
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
        box-shadow: 
            0 20px 60px 0 rgba(0, 0, 0, 0.4),
            inset 0 1px 1px rgba(255, 255, 255, 0.2),
            0 0 0 1px rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 35px; 
        text-align:center; 
        font-family:'Helvetica Neue', sans-serif;
        color: white;
        transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        transform-style: preserve-3d;
        position: relative;
        overflow: hidden;
        "
        onmouseover="this.style.transform='translateY(-10px) translateZ(30px) rotateX(2deg) rotateY(-1deg) scale(1.02)'; this.style.boxShadow='0 30px 80px 0 rgba(0, 0, 0, 0.5), inset 0 1px 1px rgba(255, 255, 255, 0.3), 0 0 40px {accent_color}40';"
        onmouseout="this.style.transform='translateY(0px) translateZ(0px) rotateX(0deg) rotateY(0deg) scale(1)'; this.style.boxShadow='0 20px 60px 0 rgba(0, 0, 0, 0.4), inset 0 1px 1px rgba(255, 255, 255, 0.2), 0 0 0 1px rgba(255, 255, 255, 0.1)';"
        >
        
        <!-- Animated Top Border -->
        <div style="position:absolute; top:0; left:0; right:0; height:2px; 
                    background: linear-gradient(90deg, transparent, {accent_color}, transparent);
                    animation: borderGlow 3s ease-in-out infinite;"></div>
        
        <h3 style="color:#b0b0b0; margin:0; text-transform:uppercase; letter-spacing:2px; 
                    font-size:0.9rem; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">Analysis Report</h3>
        <h2 style="color:white; margin-top:15px; text-shadow: 
                    0 4px 8px rgba(0,0,0,0.5),
                    0 0 20px {accent_color}40;
                    transform: translateZ(10px);">{safe_title}</h2>
        <hr style="border:none; height:1px; background:linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent); margin:20px 0;">
        
        <div style="display:flex; justify-content:space-around; align-items:center; margin-top:30px; transform-style: preserve-3d;">
            
            <div style="transform: translateZ(20px); transition: transform 0.3s ease;">
                <span style="color:#b0b0b0; font-weight:700; letter-spacing:1.5px; font-size:0.85rem;">DIFFICULTY</span><br>
                <span id="difficultyText" style="font-size:3.5rem; font-weight:900; color:{accent_color}; 
                    text-shadow: 
                        0 0 20px {accent_color}80,
                        0 0 40px {accent_color}60,
                        0 4px 8px rgba(0,0,0,0.5);
                    display:inline-block;
                    transform: translateZ(30px);
                    animation: textPulse 2s ease-in-out infinite;">
                    {display_class}
                </span>
            </div>
            
            <div style="height:100px; border-left:2px solid rgba(255,255,255,0.15); 
                        box-shadow: 0 0 10px rgba(255,255,255,0.1);"></div>
            
            <div style="width:180px; transform: translateZ(20px);">
                <span style="color:#b0b0b0; font-weight:700; letter-spacing:1.5px; font-size:0.85rem;">RATING</span>
                
                <div style="position:relative; width:180px; height:90px; overflow:hidden; margin:20px auto 0;
                            transform-style: preserve-3d;">
                    
                    <!-- Outer Glow Ring -->
                    <div style="position:absolute; width:180px; height:180px; 
                                border-radius:50%; 
                                background:radial-gradient(circle, {accent_color}20, transparent 70%);
                                animation: ringPulse 2s ease-in-out infinite;"></div>
                    
                    <!-- Base Circle -->
                    <div style="position:absolute; width:180px; height:180px; 
                                border-radius:50%; 
                                background:linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
                                box-shadow: 
                                    inset 0 2px 4px rgba(0,0,0,0.3),
                                    0 0 0 1px rgba(255,255,255,0.1);"></div>
                                
                    <!-- Progress Arc with 3D Effect -->
                    <div style="position:absolute; width:180px; height:180px; 
                                border-radius:50%;
                                background:conic-gradient(from 270deg, 
                                    {accent_color} 0deg, 
                                    {accent_color} {fill_angle}deg, 
                                    transparent {fill_angle}deg
                                ); 
                                box-shadow: 
                                    0 0 30px {accent_color}80,
                                    inset 0 0 20px {accent_color}40;
                                filter: blur(1px);
                                animation: arcGlow 2s ease-in-out infinite;"></div>
                    
                    <!-- Inner Glass Circle -->
                    <div style="position:absolute; width:140px; height:140px; 
                                background:linear-gradient(135deg, rgba(0,0,0,0.5), rgba(0,0,0,0.3));
                                backdrop-filter: blur(10px);
                                -webkit-backdrop-filter: blur(10px);
                                border-radius:50%; 
                                top:20px; left:20px;
                                border: 2px solid rgba(255,255,255,0.15);
                                box-shadow: 
                                    inset 0 2px 4px rgba(0,0,0,0.4),
                                    0 0 20px rgba(0,0,0,0.3);"></div>
                    
                    <!-- Rating Number with 3D Effect -->
                    <div style="position:absolute; width:100%; top:50px; 
                                text-align:center; font-size:2rem; 
                                font-weight:900; color:white;
                                text-shadow: 
                                    0 0 20px {accent_color}80,
                                    0 4px 8px rgba(0,0,0,0.5);
                                transform: translateZ(40px);
                                letter-spacing:1px;">
                        {pred_score}
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <style>
        @keyframes borderGlow {{
            0%, 100% {{ opacity: 0.5; }}
            50% {{ opacity: 1; }}
        }}
        @keyframes textPulse {{
            0%, 100% {{ transform: translateZ(30px) scale(1); }}
            50% {{ transform: translateZ(30px) scale(1.05); }}
        }}
        @keyframes ringPulse {{
            0%, 100% {{ transform: scale(1); opacity: 0.5; }}
            50% {{ transform: scale(1.1); opacity: 0.8; }}
        }}
        @keyframes arcGlow {{
            0%, 100% {{ filter: blur(1px) brightness(1); }}
            50% {{ filter: blur(2px) brightness(1.2); }}
        }}
    </style>
    """
    
    components.html(textwrap.dedent(result_html), height=380)