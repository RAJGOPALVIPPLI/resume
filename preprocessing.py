import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re

# Download necessary NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def clean_text(text):
    """
    Perform text preprocessing:
    - Lowercasing
    - Removing punctuation
    - Removing stopwords
    - Lemmatization
    """
    # Lowercasing
    text = text.lower()
    
    # Removing punctuation and special characters
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenization (simple split for this script)
    words = text.split()
    
    # Stopwords removal
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    
    return " ".join(words)
