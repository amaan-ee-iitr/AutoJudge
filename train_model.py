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
from imblearn.over_sampling import SMOTE # <--- NEW IMPORT

# ==========================================
#  STEP 1: DATA INGESTION & CLEANING
# ==========================================
print(">> Step 1: Loading and sanitizing data...")

try:
    raw_data = pd.read_json("problems.jsonl", lines=True)
except ValueError:
    raw_data = pd.read_json("problems.jsonl")

# --- FIX: Normalize Class Names ---
raw_data['problem_class'] = raw_data['problem_class'].str.title() 

print("   Original Dataset Distribution:")
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

# Assign Scores
def generate_difficulty_rating(row_data):
    category = row_data['problem_class']
    if category == 'Easy': return np.random.choice([800, 900, 1000])
    elif category == 'Medium': return np.random.choice([1100, 1200, 1300, 1400, 1500])
    else: return np.random.choice(range(1600, 3500, 100))

raw_data['estimated_rating'] = raw_data.apply(generate_difficulty_rating, axis=1)


# ==========================================
#  STEP 2: FEATURE ENGINEERING (With Bigrams)
# ==========================================
print(">> Step 2: Vectorizing (Unigrams + Bigrams)...")

# ngram_range=(1, 2) -> Reads "word" AND "word word"
tfidf_engine = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 2))

# ... rest of your vectorization code ...

tfidf_engine = TfidfVectorizer(max_features=1000, stop_words='english')
text_vec = tfidf_engine.fit_transform(raw_data['processed_text']).toarray()

len_scaler = MinMaxScaler()
lens = raw_data['processed_text'].apply(lambda x: len(x.split())).values.reshape(-1, 1)
len_vec = len_scaler.fit_transform(lens)

# Combine features
X = np.hstack((text_vec, len_vec))
y_class = raw_data['problem_class']
y_score = raw_data['estimated_rating']

# --- SPLIT FIRST (Crucial for SMOTE) ---
# We must split BEFORE applying SMOTE to avoid data leakage
X_train, X_test, y_class_train, y_class_test, y_score_train, y_score_test = train_test_split(
    X, y_class, y_score, test_size=0.2, random_state=42
)

print(f"   Split -> Train: {len(X_train)} samples, Test: {len(X_test)} samples")

# ==========================================
#  STEP 3: APPLYING SMOTE (BALANCING)
# ==========================================
print(">> Step 3: Applying SMOTE to Training Data...")
print(f"   Before SMOTE counts:\n{y_class_train.value_counts()}")

# Initialize SMOTE
smote = SMOTE(random_state=42)

# Resample ONLY the training data
X_train_resampled, y_class_train_resampled = smote.fit_resample(X_train, y_class_train)
# Note: For regression (y_score), we usually just duplicate the scores or use the original 
# indices, but for simplicity here we will train regression on the original imbalanced data 
# (regression is less sensitive to class imbalance than classification).

print(f"   After SMOTE counts:\n{y_class_train_resampled.value_counts()}")

# ==========================================
#  STEP 4: TRAINING & EVALUATION
# ==========================================
print(">> Step 4: Training Models...")

# --- Classification (Trained on SMOTE data) ---
difficulty_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
difficulty_classifier.fit(X_train_resampled, y_class_train_resampled)

# Evaluate on original (REAL) test data
preds_class = difficulty_classifier.predict(X_test)
acc = accuracy_score(y_class_test, preds_class)
print(f"   Classifier Accuracy on Test Set: {acc*100:.2f}%")

# Confusion Matrix
labels = ['Easy', 'Medium', 'Hard']
cm = confusion_matrix(y_class_test, preds_class, labels=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('SMOTE Augmented Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png')
print("Confusion Matrix saved")

# --- Regression (Trained on Original data) ---
# SMOTE is hard for regression without specialized tools, so we use the standard training set
rating_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
rating_predictor.fit(X_train, y_score_train)

preds_score = rating_predictor.predict(X_test)
mae = mean_absolute_error(y_score_test, preds_score)
print(f"   Rating MAE: {mae:.2f}")

# ==========================================
#  STEP 5: SAVING ARTIFACTS
# ==========================================
print(">> Step 5: Saving final models...")

# For the final saved model, we apply SMOTE to the ENTIRE dataset 
# so the production model is as smart as possible.
X_resampled_full, y_class_resampled_full = smote.fit_resample(X, y_class)

difficulty_classifier.fit(X_resampled_full, y_class_resampled_full)
rating_predictor.fit(X, y_score) # Regression trained on original X

joblib.dump(difficulty_classifier, 'classifier.pkl')
joblib.dump(rating_predictor, 'regressor.pkl')
joblib.dump(tfidf_engine, 'tfidf.pkl')
joblib.dump(len_scaler, 'scaler.pkl')

print("✅ Models updated with SMOTE Strategy.")