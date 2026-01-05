import streamlit as st
import pickle
import plotly.graph_objects as go

# --- 1. Load Models ---
try:
    with open('model_data.pkl', 'rb') as f:
        bundle = pickle.load(f)
    classifier = bundle["classifier"]
    regressor = bundle["regressor"]
    vectorizer = bundle["vectorizer"]
except FileNotFoundError:
    st.error("⚠️ Model file not found. Run 'python train.py' first.")
    st.stop()

# --- 2. Logic Helpers ---
def convert_to_cf_rating(score):
    rating = 800 + (score - 1) * 300
    return int(max(800, min(3500, round(rating / 100) * 100)))

def get_rating_color(rating):
    if rating < 1200: return "#CCCCCC" # Newbie (Gray)
    if rating < 1400: return "#77FF77" # Pupil (Light Green)
    if rating < 1600: return "#03A89E" # Specialist (Cyan)
    if rating < 1900: return "#AAAAFF" # Expert (Blue)
    if rating < 2100: return "#FF88FF" # Candidate Master (Purple)
    if rating < 2400: return "#FFCC88" # Master (Orange)
    return "#FF3333"                 # Grandmaster (Red)

# --- 3. Page Config & 3D CSS ---
st.set_page_config(page_title="AutoJudge 3D", page_icon="🧊", layout="centered")

# Custom CSS for 3D Effects (Neumorphism / Glassmorphism)
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background: linear-gradient(135deg, #1e1e2f 0%, #252540 100%);
        color: white;
    }
    
    /* 3D Title */
    h1 {
        text-shadow: 2px 2px 0px #000000;
        font-weight: 800;
    }

    /* Floating Input Cards */
    .stTextArea textarea {
        background-color: #2b2b40;
        color: #ffffff;
        border: 1px solid #444;
        border-radius: 12px;
        box-shadow: 5px 5px 15px rgba(0,0,0,0.5), 
                   -2px -2px 10px rgba(255,255,255,0.05);
        transition: all 0.3s ease;
    }
    .stTextArea textarea:focus {
        transform: translateY(-2px);
        box-shadow: 8px 8px 20px rgba(0,0,0,0.6);
        border-color: #6c5ce7;
    }

    /* 3D Button */
    .stButton>button {
        background: linear-gradient(145deg, #6c5ce7, #4834d4);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 15px 30px;
        font-weight: bold;
        box-shadow: 4px 4px 10px rgba(0,0,0,0.4), 
                   -2px -2px 5px rgba(255,255,255,0.1);
        transition: all 0.2s;
    }
    .stButton>button:active {
        transform: scale(0.98);
        box-shadow: inset 4px 4px 10px rgba(0,0,0,0.4);
    }
    
    /* Result Card */
    .result-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- 4. Main UI ---
st.title("🧊 AutoJudge 3D")
st.caption("AI-Powered Competitive Programming Difficulty Estimator")

with st.container():
    st.markdown("### 📝 Problem Details")
    desc = st.text_area("Paste Problem Description", height=200, placeholder="Once upon a time in Berland...")
    
    col1, col2 = st.columns(2)
    with col1:
        submitted = st.button("🚀 Predict Difficulty", use_container_width=True)

# --- 5. Prediction & Visualization ---
if submitted and desc:
    # Processing
    vec = vectorizer.transform([desc])
    p_class = classifier.predict(vec)[0]
    p_score = regressor.predict(vec)[0]
    
    cf_rating = convert_to_cf_rating(p_score)
    color = get_rating_color(cf_rating)

    st.divider()
    
    # --- RESULT SECTION ---
    c1, c2 = st.columns([1, 1.5])
    
    with c1:
        # 3D Card for Text Result
        st.markdown(f"""
        <div class="result-card">
            <h3 style="margin:0; color: #aaa;">Difficulty Tier</h3>
            <h1 style="font-size: 3em; margin: 10px 0; color: {color}; text-shadow: 0 0 10px {color}88;">
                {p_class.upper()}
            </h1>
            <p>Model Confidence: High</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        # Plotly Gauge Chart (Looks 3D)
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = cf_rating,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Codeforces Rating", 'font': {'size': 20, 'color': "white"}},
            number = {'font': {'color': color, 'weight': 'bold'}},
            gauge = {
                'axis': {'range': [800, 3500], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': color, 'thickness': 0.7}, # The rating bar
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "#333",
                'steps': [
                    {'range': [800, 1200], 'color': '#333'},
                    {'range': [1200, 1900], 'color': '#444'},
                    {'range': [1900, 2400], 'color': '#555'},
                    {'range': [2400, 3500], 'color': '#666'}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': cf_rating
                }
            }
        ))
        
        # Transparent background for graph
        fig.update_layout(paper_bgcolor = "rgba(0,0,0,0)", font = {'color': "white", 'family': "Arial"})
        st.plotly_chart(fig, use_container_width=True)