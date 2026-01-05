import streamlit as st
import joblib
import numpy as np
import re
import pandas as pd

# ==========================================
# 1. SETUP & LOADING
# ==========================================
st.set_page_config(page_title="AutoJudge AI", page_icon="⚖️", layout="centered")

# --- 3D CSS STYLES ---
st.markdown("""
<style>
    /* Background Gradient */
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    @keyframes gradient {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    /* Text Styling */
    div.stTextArea > label, div.stTextInput > label {
        color: #333 !important; /* Dark text for inside white cards */
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    h1, h2, h3 {
        font-family: 'Helvetica', sans-serif;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }

    /* 3D Card Class */
    .card-3d {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.19), 0 6px 6px rgba(0,0,0,0.23);
        transition: all 0.3s cubic-bezier(.25,.8,.25,1);
        margin-bottom: 20px;
    }
    .card-3d:hover {
        transform: translate3d(0, -5px, 0);
        box-shadow: 0 14px 28px rgba(0,0,0,0.25), 0 10px 10px rgba(0,0,0,0.22);
    }

    /* 3D Button */
    div.stButton > button {
        background: linear-gradient(to bottom, #ff4b1f, #ff9068);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: bold;
        font-size: 18px;
        box-shadow: 0 5px #c0392b;
        transition: all 0.1s;
        width: 100%;
    }
    div.stButton > button:active {
        box-shadow: 0 2px #c0392b;
        transform: translateY(3px);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    try:
        clf = joblib.load('model_classifier.pkl')
        reg = joblib.load('model_regressor.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        scaler = joblib.load('scaler.pkl')
        return clf, reg, vectorizer, scaler
    except FileNotFoundError:
        return None, None, None, None

clf, reg, vectorizer, scaler = load_artifacts()

if clf is None:
    st.error("❌ Error: Model files not found. Please run 'train.py' first.")
    st.stop()

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text) 
    return text

def preprocess_input(description, input_desc, output_desc):
    # Note: We do NOT use the title for prediction to maintain consistency 
    # with how the model was trained (which only used desc + in + out).
    combined_text = f"{description} {input_desc} {output_desc}"
    cleaned_text = clean_text(combined_text)
    
    X_text = vectorizer.transform([cleaned_text]).toarray()
    text_len = len(cleaned_text.split())
    X_len = scaler.transform([[text_len]])
    
    return np.hstack((X_text, X_len))

# ==========================================
# 3. USER INTERFACE
# ==========================================
st.markdown("<h1 style='color: white; text-align: center;'>🤖 AutoJudge AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: white; text-align: center; margin-bottom: 30px;'>Predict Competitive Programming Problem Difficulty</p>", unsafe_allow_html=True)

# -- INPUT CARD --
with st.container():
    st.markdown('<div class="card-3d">', unsafe_allow_html=True)
    
    # NEW: Problem Title Input
    title = st.text_input("Problem Title", placeholder="e.g., Dijkstra's Shortest Path")
    
    desc = st.text_area("Problem Description", height=150, placeholder="e.g., Given a graph with N nodes...")
    
    col1, col2 = st.columns(2)
    with col1:
        inp_desc = st.text_area("Input Format", height=80, placeholder="e.g., First line T...")
    with col2:
        out_desc = st.text_area("Output Format", height=80, placeholder="e.g., Print the integer...")
        
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 4. PREDICTION LOGIC
# ==========================================
st.write("") # Spacing

if st.button("🚀 Analyze Problem"):
    if not desc:
        st.warning("Please enter at least a problem description.")
    else:
        # Run Prediction
        X_input = preprocess_input(desc, inp_desc, out_desc)
        pred_class = clf.predict(X_input)[0]
        pred_score = reg.predict(X_input)[0]
        
        display_class = pred_class.title()
        
        # Determine Color
        if display_class == "Easy":
            res_color = "#2ecc71"
        elif display_class == "Medium":
            res_color = "#f39c12"
        else:
            res_color = "#e74c3c"
            
        # Handle empty title for display
        display_title = title if title else "Untitled Problem"

        # -- RESULT CARD --
        st.markdown(f"""
        <div class="card-3d" style="text-align: center;">
            <h3 style='color: #555; margin: 0;'>Analysis Report for:</h3>
            <h2 style='color: #333; margin-top: 5px;'>{display_title}</h2>
            <hr style="border: 1px solid #eee;">
            
            <div style="display: flex; justify-content: space-around; align-items: center; margin-top: 20px;">
                <div>
                    <h4 style="color: #777; margin-bottom: 5px;">Class</h4>
                    <h1 style="color: {res_color}; font-size: 3rem; margin: 0; text-shadow: 1px 1px 2px #ccc;">{display_class}</h1>
                </div>
                <div style="height: 60px; border-left: 2px solid #eee;"></div>
                <div>
                    <h4 style="color: #777; margin-bottom: 5px;">Rating</h4>
                    <h1 style="color: #333; font-size: 3rem; margin: 0; text-shadow: 1px 1px 2px #ccc;">{int(pred_score)}</h1>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Progress Bar
        st.write(f"**Difficulty Meter**")
        progress_val = min(max(pred_score / 3500, 0.0), 1.0)
        st.progress(progress_val)