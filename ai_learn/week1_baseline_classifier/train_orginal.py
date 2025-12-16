import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -----------------------------
# 1) Load data
# -----------------------------
df = pd.read_csv("train.csv")

X = df["text"]
y = df["label"]

# -----------------------------
# 2) Simple preprocessing
# -----------------------------
def clean_text(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    return text.lower()

X = X.apply(clean_text)

# -----------------------------
# 3) Train / test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 4) TF-IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# 5) Logistic Regression
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

print("Training complete")

# Save artifacts
import joblib
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("Saved model and vectorizer")
