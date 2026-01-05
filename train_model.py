import pandas as pd
import joblib
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error

# ==========================================
#  STEP 1: DATA INGESTION & CLEANING
# ==========================================
print(">> Step 1: Loading and sanitizing data...")

try:
    raw_data = pd.read_json("problems.jsonl", lines=True)
except ValueError:
    raw_data = pd.read_json("problems.jsonl")

# --- FIX: Normalize Class Names (Handle 'easy' vs 'Easy') ---
# This forces all labels to be "Easy", "Medium", "Hard"
raw_data['problem_class'] = raw_data['problem_class'].str.title() 

# Print counts so we know exactly what we are working with
print("   Dataset Distribution:")
print(raw_data['problem_class'].value_counts())

# Merge fields
raw_data['full_content'] = raw_data['title'].astype(str) + " " + \
                           raw_data['description'].astype(str) + " " + \
                           raw_data['input_description'].astype(str) + " " + \
                           raw_data['output_description'].astype(str)

def text_sanitizer(raw_text):
    s = str(raw_text).lower()
    s = re.sub(r'<.*?>', '', s)
    return s

raw_data['processed_text'] = raw_data['full_content'].apply(text_sanitizer)

# Assign Scores based on normalized class
def generate_difficulty_rating(row_data):
    category = row_data['problem_class']
    if category == 'Easy': return np.random.choice([800, 900, 1000])
    elif category == 'Medium': return np.random.choice([1100, 1200, 1300, 1400, 1500])
    else: return np.random.choice(range(1600, 3500, 100))

raw_data['estimated_rating'] = raw_data.apply(generate_difficulty_rating, axis=1)

# ==========================================
#  STEP 2: FEATURE ENGINEERING
# ==========================================
print(">> Step 2: Vectorizing...")

tfidf_engine = TfidfVectorizer(max_features=1000, stop_words='english')
text_vec = tfidf_engine.fit_transform(raw_data['processed_text']).toarray()

len_scaler = MinMaxScaler()
lens = raw_data['processed_text'].apply(lambda x: len(x.split())).values.reshape(-1, 1)
len_vec = len_scaler.fit_transform(lens)

# Combine features
X = np.hstack((text_vec, len_vec))
y_class = raw_data['problem_class']
y_score = raw_data['estimated_rating']

# --- STANDARD SPLIT (80% Train, 20% Test) ---
X_train, X_test, y_class_train, y_class_test, y_score_train, y_score_test = train_test_split(
    X, y_class, y_score, test_size=0.2, random_state=42
)

print(f"   Split -> Train: {len(X_train)} samples, Test: {len(X_test)} samples")

# ==========================================
#  STEP 3: TRAINING & EVALUATION
# ==========================================
print(">> Step 3: Training Models...")

# --- Classification ---
# 'class_weight="balanced"' helps even if we don't manually undersample
difficulty_classifier = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
difficulty_classifier.fit(X_train, y_class_train)

preds_class = difficulty_classifier.predict(X_test)
acc = accuracy_score(y_class_test, preds_class)
print(f"   Classifier Accuracy on Test Set: {acc*100:.2f}%")

# Confusion Matrix
labels = ['Easy', 'Medium', 'Hard']
cm = confusion_matrix(y_class_test, preds_class, labels=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Standard Training Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png')
print(" Confusion Matrix saved ")

# --- Regression ---
rating_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
rating_predictor.fit(X_train, y_score_train)

preds_score = rating_predictor.predict(X_test)
mae = mean_absolute_error(y_score_test, preds_score)
print(f"   Rating MAE: {mae:.2f}")

# ==========================================
#  STEP 4: SAVING ARTIFACTS
# ==========================================
print(">> Step 4: Saving final models...")

# Retrain on FULL dataset for the final product (Standard Practice)
difficulty_classifier.fit(X, y_class)
rating_predictor.fit(X, y_score)

joblib.dump(difficulty_classifier, 'classifier.pkl')
joblib.dump(rating_predictor, 'regressor.pkl')
joblib.dump(tfidf_engine, 'tfidf.pkl')
joblib.dump(len_scaler, 'scaler.pkl')

print("✅ Models Trained")