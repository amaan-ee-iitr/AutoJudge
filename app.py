import streamlit as st
import pickle
import pandas as pd

# 1. Load the saved models from the single bundle
try:
    with open('model_data.pkl', 'rb') as f:
        bundle = pickle.load(f)
    
    classifier = bundle["classifier"]
    regressor = bundle["regressor"]
    vectorizer = bundle["vectorizer"]
except FileNotFoundError:
    st.error("⚠️ Model file 'model_data.pkl' not found. Please run train.py first.")
    st.stop()

# 2. UI Layout
st.set_page_config(page_title="AutoJudge", page_icon="⚖️")

st.title("⚖️ AutoJudge: Problem Difficulty Predictor")
st.markdown("Paste your programming problem details below to predict its difficulty.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        desc = st.text_area("Problem Description", height=200, placeholder="Alice has an array...")
    
    with col2:
        inp_desc = st.text_area("Input Description", height=90, placeholder="First line contains T...")
        out_desc = st.text_area("Output Description", height=90, placeholder="Print the sum...")
        
    submitted = st.form_submit_button("Predict Difficulty")

# 3. Prediction Logic
if submitted:
    if not desc:
        st.warning("Please enter at least a Problem Description.")
    else:
        # Combine inputs
        combined_text = (desc + " " + inp_desc + " " + out_desc).lower()
        
        # --- CRITICAL STEP: Convert text to numbers ---
        # The model cannot read text directly; it needs the vectorizer
        vectorized_text = vectorizer.transform([combined_text])
        
        # Make predictions
        pred_class = classifier.predict(vectorized_text)[0]
        pred_score = regressor.predict(vectorized_text)[0]
        
        # Display Results
        st.divider()
        st.subheader("Prediction Results")
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.info("Predicted Category")
            # Color coding
            if pred_class.lower() == 'easy':
                st.success(f"🟢 {pred_class.upper()}")
            elif pred_class.lower() == 'medium':
                st.warning(f"🟡 {pred_class.upper()}")
            else:
                st.error(f"🔴 {pred_class.upper()}")
                
        with c2:
            st.info("Predicted Score")
            st.metric(label="Numerical Difficulty", value=f"{pred_score:.2f}")