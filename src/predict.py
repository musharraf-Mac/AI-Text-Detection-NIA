import pickle
import pandas as pd

# Load once
_model = pickle.load(open("../models/baseline_model.pkl", "rb"))
_vectorizer = pickle.load(open("../models/tfidf_vectorizer.pkl", "rb"))

def predict_text(text):
    """Predict a single text: returns 'human' or 'ai'"""
    X = _vectorizer.transform([text])
    return _model.predict(X)[0]

def predict_proba(text):
    """Returns probability/confidence for each class"""
    X = _vectorizer.transform([text])
    proba = _model.predict_proba(X)[0]
    return dict(zip(_model.classes_, proba))

def predict_batch(df, text_col="text"):
    """Predict on a whole dataframe, adds 'predicted' column"""
    X = _vectorizer.transform(df[text_col])
    df = df.copy()
    df["predicted"] = _model.predict(X)
    return df