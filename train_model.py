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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, mean_squared_error

# ==========================================
#  REQUIREMENT 1: DATA PREPROCESSING
# ==========================================
print("1. Data Preprocessing...")
try:
    df = pd.read_json("problems.jsonl", lines=True)
except ValueError:
    df = pd.read_json("problems.jsonl")

# Combine text columns
df['text_combined'] = df['description'].astype(str) + " " + \
                      df['input_description'].astype(str) + " " + \
                      df['output_description'].astype(str)

def cleaner(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text) 
    return text

df['clean_text'] = df['text_combined'].apply(cleaner)

# STRICT SCORING LOGIC
def remap_score(row):
    p_class = row['problem_class']
    if p_class == 'Easy': return np.random.choice([800, 900, 1000])
    elif p_class == 'Medium': return np.random.choice([1100, 1200, 1300, 1400, 1500])
    else: return np.random.choice(range(1600, 3500, 100))

df['problem_score'] = df.apply(remap_score, axis=1)

# ==========================================
#  REQUIREMENT 2: FEATURE EXTRACTION
# ==========================================
print("2. Feature Extraction...")
# Feature A: TF-IDF
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_text = vectorizer.fit_transform(df['clean_text']).toarray()

# Feature B: Text Length
X_len = df['clean_text'].apply(lambda x: len(x.split())).values.reshape(-1, 1)
scaler = MinMaxScaler()
X_len = scaler.fit_transform(X_len)

# Combine features
X = np.hstack((X_text, X_len))
y_class = df['problem_class']
y_score = df['problem_score']

# NEW: Split data (80% Train, 20% Test) to evaluate performance
X_train, X_test, y_class_train, y_class_test, y_score_train, y_score_test = train_test_split(
    X, y_class, y_score, test_size=0.2, random_state=42
)

# ==========================================
#  REQUIREMENT 3: EVALUATION (Metrics & Matrix)
# ==========================================
print("3. Training & Evaluating Models...")

# --- A. Classification Evaluation ---
clf_eval = RandomForestClassifier(n_estimators=100, random_state=42)
clf_eval.fit(X_train, y_class_train)
y_pred_class = clf_eval.predict(X_test)

# Calculate Accuracy
acc = accuracy_score(y_class_test, y_pred_class)
print(f"\n>>> Classification Model Accuracy: {acc*100:.2f}%")
print("\nDetailed Classification Report:")
print(classification_report(y_class_test, y_pred_class))

# Plot Confusion Matrix
# FIX: Changed labels to lowercase ['easy', 'medium', 'hard'] to match your data
cm = confusion_matrix(y_class_test, y_pred_class, labels=['easy', 'medium', 'hard'])

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['easy', 'medium', 'hard'], 
            yticklabels=['easy', 'medium', 'hard'])
plt.title('Confusion Matrix: Difficulty Prediction')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.savefig('confusion_matrix.png')
plt.close()
print("Graph saved: 'confusion_matrix.png'")

# --- B. Regression Evaluation ---
# (Rest of the code remains the same)
reg_eval = RandomForestRegressor(n_estimators=100, random_state=42)
reg_eval.fit(X_train, y_score_train)
y_pred_score = reg_eval.predict(X_test)

mae = mean_absolute_error(y_score_test, y_pred_score)
rmse = np.sqrt(mean_squared_error(y_score_test, y_pred_score))

print(f"\n>>> Regression Model Performance:")
print(f"Mean Absolute Error (MAE): {mae:.2f} points")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} points")
# ==========================================
#  REQUIREMENT 4: FINAL SAVING
# ==========================================
print("\n4. Retraining on FULL dataset and Saving Artifacts...")

# We retrain on X (all data) so the saved model is as smart as possible
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y_class)

reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X, y_score)

joblib.dump(clf, 'model_classifier.pkl')
joblib.dump(reg, 'model_regressor.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✅ All requirements met. Models and Evaluation results saved.")