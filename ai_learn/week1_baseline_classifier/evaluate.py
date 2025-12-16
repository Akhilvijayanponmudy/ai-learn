import pandas as pd
import joblib
import re

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Load data
df = pd.read_csv("train.csv")

def clean_text(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    return text.lower()

X = df["text"].apply(clean_text)
y = df["label"]

# Load model + vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

X_vec = vectorizer.transform(X)
y_pred = model.predict(X_vec)

print("Accuracy:", accuracy_score(y, y_pred))
print("Precision:", precision_score(y, y_pred))
print("Recall:", recall_score(y, y_pred))
print("F1:", f1_score(y, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y, y_pred))

print("\nClassification Report:")
print(classification_report(y, y_pred))
