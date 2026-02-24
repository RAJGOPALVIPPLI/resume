import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import clean_text

# Load the saved models
def load_assets():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('tfidf.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        return model, tfidf
    except FileNotFoundError:
        print("Error: Model or TF-IDF files not found. Run train.py first.")
        return None, None

def predict_resume(text):
    """
    Predict suitability and return probability
    """
    model, tfidf = load_assets()
    if not model or not tfidf:
        return None
    
    # Preprocess
    cleaned = clean_text(text)
    
    # Transform
    vector = tfidf.transform([cleaned])
    
    # Predict
    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0][1]
    
    return {
        "suitable": bool(prediction),
        "score": float(probability)
    }

def calculate_similarity(resume_text, job_description):
    """
    Bonus: Cosine similarity between resume and job description
    """
    _, tfidf = load_assets()
    if not tfidf:
        return 0.0
    
    res_cleaned = clean_text(resume_text)
    jd_cleaned = clean_text(job_description)
    
    vectors = tfidf.transform([res_cleaned, jd_cleaned])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])
    
    return float(similarity[0][0])

if __name__ == "__main__":
    # Example
    sample_resume = "Experienced Software Engineer with 5 years in Python and Machine Learning."
    result = predict_resume(sample_resume)
    if result:
        print(f"Prediction: {'Suitable' if result['suitable'] else 'Not Suitable'}")
        print(f"Confidence: {result['score']:.2%}")
