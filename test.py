import streamlit
import 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_EMBEDDINGS_BACKEND = "tfidf-fallback"

def embed_text(sentences):
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(sentences)
    norms = np.sqrt((X.multiply(X)).sum(axis=1))
    norms[norms == 0] = 1.0
    return (X.multiply(1.0 / norms)).toarray()
