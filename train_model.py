import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import classification_report, mean_squared_error, r2_score

# 1. Load Data
def load_data(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except: pass
    return pd.DataFrame(data)

print("Loading dataset...")
df = load_data('problems.jsonl')
print(f"Loaded {len(df)} problems.")

# 2. Preprocessing
# Combine text fields to give the model maximum context
print("Preprocessing text...")
df['full_text'] = (
    df['title'].fillna('') + " " + 
    df['description'].fillna('') + " " + 
    df['input_description'].fillna('') + " " + 
    df['output_description'].fillna('')
)

# Features (X) and Targets (y)
X_text = df['full_text']
y_class = df['problem_class']  # Target for Classification (Easy/Medium/Hard)
y_score = df['problem_score']  # Target for Regression (1.0 - 10.0)

# 3. Vectorization (Convert text to numbers)
# Using TF-IDF (Term Frequency-Inverse Document Frequency)
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(X_text)

# 4. Split Data
# We use the same split for both tasks to compare them fairly
X_train, X_test, y_class_train, y_class_test, y_score_train, y_score_test = train_test_split(
    X, y_class, y_score, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# TASK 1: CLASSIFICATION (Easy vs Medium vs Hard)
# ---------------------------------------------------------
print("\n--- Training Classifier (Easy/Medium/Hard) ---")
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_class_train)

# Evaluate Classifier
y_class_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_class_test, y_class_pred))

# ---------------------------------------------------------
# TASK 2: REGRESSION (Predicting Rating/Score)
# ---------------------------------------------------------
print("\n--- Training Regressor (Predicting Numeric Rating) ---")
# Using Ridge Regression for speed and stability with text data, 
# but RandomForestRegressor works well too.
reg = Ridge(alpha=1.0) 
reg.fit(X_train, y_score_train)

# Evaluate Regressor
y_score_pred = reg.predict(X_test)
mse = mean_squared_error(y_score_test, y_score_pred)
r2 = r2_score(y_score_test, y_score_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R^2 Score (Accuracy of fit): {r2:.4f}")

# ---------------------------------------------------------
# DEMO: Show predictions for a few test samples
# ---------------------------------------------------------
print("\n--- Sample Predictions ---")
test_indices = np.random.choice(X_test.shape[0], 5, replace=False)

for i in test_indices:
    # Get original text (re-fetching from dataframe via index alignment is tricky with sparse matrices, 
    # so we just show the predicted vs actual values here)
    actual_cls = y_class_test.iloc[i]
    pred_cls = y_class_pred[i]
    
    actual_scr = y_score_test.iloc[i]
    pred_scr = y_score_pred[i]
    
    print(f"Actual: [{actual_cls.upper()} | {actual_scr}]  -->  Predicted: [{pred_cls.upper()} | {pred_scr:.2f}]")