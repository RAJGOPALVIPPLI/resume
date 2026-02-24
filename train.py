import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from preprocessing import clean_text

def train_model(csv_path="resumes.csv"):
    # 1. DATA HANDLING
    print("Loading data...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found. Please provide a dataset.")
        return

    # 2. PREPROCESSING
    print("Preprocessing text...")
    df['cleaned_text'] = df['resume_text'].apply(clean_text)

    # 3. FEATURE EXTRACTION
    print("Extracting features...")
    tfidf = TfidfVectorizer(max_features=3000, stop_words='english')
    X = tfidf.fit_transform(df['cleaned_text'])
    y = df['label']

    # 4. MODEL SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. TRAIN LOGISTIC REGRESSION
    print("Training Logistic Regression...")
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)

    # 6. EVALUATION (Logistic Regression)
    print("\n--- Logistic Regression Performance ---")
    print(f"Accuracy:  {accuracy_score(y_test, lr_preds):.4f}")
    print(f"Precision: {precision_score(y_test, lr_preds):.4f}")
    print(f"Recall:    {recall_score(y_test, lr_preds):.4f}")
    print(f"F1 Score:  {f1_score(y_test, lr_preds):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, lr_preds))

    # 7. ALTERNATIVE: RANDOM FOREST
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    print("\n--- Random Forest Performance ---")
    print(f"Accuracy:  {accuracy_score(y_test, rf_preds):.4f}")
    print(f"Precision: {precision_score(y_test, rf_preds):.4f}")
    print(f"Recall:    {recall_score(y_test, rf_preds):.4f}")
    print(f"F1 Score:  {f1_score(y_test, rf_preds):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, rf_preds))

    # 8. SAVE MODEL (Using Logistic Regression as primary)
    print("\nSaving models...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(lr_model, f)
    with open('tfidf.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
    print("Done.")

if __name__ == "__main__":
    # Example usage:
    # train_model("data.csv")
    pass
