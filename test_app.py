"""Quick test script to verify the app can load"""
import sys
import os

print("Testing AutoJudge App Setup...")
print("=" * 50)

# Test imports
print("\n1. Testing imports...")
try:
    import streamlit as st
    print("   OK: streamlit imported")
except ImportError as e:
    print(f"   ERROR: streamlit not found: {e}")
    print("   Please install: pip install streamlit")
    sys.exit(1)

try:
    import joblib
    print("   OK: joblib imported")
except ImportError as e:
    print(f"   ERROR: joblib not found: {e}")

try:
    import numpy as np
    print("   OK: numpy imported")
except ImportError as e:
    print(f"   ERROR: numpy not found: {e}")

try:
    import pandas as pd
    print("   OK: pandas imported")
except ImportError as e:
    print(f"   ERROR: pandas not found: {e}")

# Test model files
print("\n2. Checking model files...")
model_files = ['classifier.pkl', 'regressor.pkl', 'tfidf.pkl', 'scaler.pkl']
all_exist = True
for model_file in model_files:
    if os.path.exists(model_file):
        print(f"   OK: {model_file} exists")
    else:
        print(f"   ERROR: {model_file} NOT FOUND")
        all_exist = False

if not all_exist:
    print("\n   WARNING: Some model files are missing!")
    print("   Run 'python train_model.py' to generate them.")

# Test app.py syntax
print("\n3. Testing app.py syntax...")
try:
    with open('app.py', 'r', encoding='utf-8') as f:
        code = f.read()
    compile(code, 'app.py', 'exec')
    print("   OK: app.py syntax is valid")
except SyntaxError as e:
    print(f"   ERROR: Syntax error in app.py: {e}")
    sys.exit(1)
except Exception as e:
    print(f"   ERROR: Error reading app.py: {e}")
    sys.exit(1)

print("\n" + "=" * 50)
print("Test complete!")
if all_exist:
    print("SUCCESS: Everything looks good! Run 'streamlit run app.py' to start the app.")
else:
    print("WARNING: Fix the issues above before running the app.")
