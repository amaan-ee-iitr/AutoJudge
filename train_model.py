import json
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge

def train_and_save():
    print("1. Loading Data...")
    data = []
    # Ensure this matches your uploaded filename exactly
    filename = 'problems.jsonl' 
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try: data.append(json.loads(line))
                    except: pass
    except FileNotFoundError:
        print(f"Error: '{filename}' not found. Please make sure the file is in this folder.")
        return

    df = pd.DataFrame(data)
    
    # Combined text for training
    df['full_text'] = (
        df['title'].fillna('') + " " + 
        df['description'].fillna('') + " " + 
        df['input_description'].fillna('') + " " + 
        df['output_description'].fillna('')
    )
    
    # 2. Vectorization (The step you were missing!)
    print("2. Learning Vocabulary (Vectorizing)...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(df['full_text'])
    
    y_class = df['problem_class']
    y_score = df['problem_score']

    # 3. Train Models
    print("3. Training Classifier (Easy/Medium/Hard)...")
    clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    clf.fit(X, y_class)

    print("4. Training Regressor (Score prediction)...")
    reg = Ridge(alpha=1.0)
    reg.fit(X, y_score)

    # 5. Save EVERYTHING in one bundle
    print("5. Saving to 'model_data.pkl'...")
    bundle = {
        "classifier": clf,
        "regressor": reg,
        "vectorizer": vectorizer  # <--- CRITICAL: Saving the dictionary
    }
    
    with open('model_data.pkl', 'wb') as f:
        pickle.dump(bundle, f)
    
    print("Done! You now have a working model file.")

if __name__ == "__main__":
    train_and_save()