import streamlit as st
import joblib
import pandas as pd

# 1. Load the saved models
try:
    classifier = joblib.load('model_classifier.pkl')
    regressor = joblib.load('model_regressor.pkl')
except:
    st.error("Models not found. Please run train_model.py first.")
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
        # Combine inputs exactly how we did in training
        combined_text = (desc + " " + inp_desc + " " + out_desc).lower()
        
        # Make predictions
        # Note: We pass a list [combined_text] because the model expects an iterable
        pred_class = classifier.predict([combined_text])[0]
        pred_score = regressor.predict([combined_text])[0]
        
        # Display Results
        st.divider()
        st.subheader("Prediction Results")
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.info("Predicted Category")
            # Color coding based on result
            if pred_class.lower() == 'easy':
                st.success(f"🟢 {pred_class}")
            elif pred_class.lower() == 'medium':
                st.warning(f"🟡 {pred_class}")
            else:
                st.error(f"🔴 {pred_class}")
                
        with c2:
            st.info("Predicted Score")
            st.metric(label="Numerical Difficulty", value=f"{pred_score:.2f}")