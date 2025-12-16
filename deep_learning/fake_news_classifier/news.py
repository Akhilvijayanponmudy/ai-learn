import re
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential

from sklearn.model_selection import train_test_split

# -----------------------------
print("completed imports")
print("-----------------------------")

# -----------------------------
# 1) Load dataset
# -----------------------------
df = pd.read_csv("train.csv")
df = df.fillna("")

TEXT_COL = "title"
LABEL_COL = "label"

X = df[TEXT_COL]
y = df[LABEL_COL].astype(int)

print(f"dataset loaded: {len(df)} rows")
print("-----------------------------")

# -----------------------------
# 2) NLTK setup
# -----------------------------
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

# -----------------------------
# 3) Text preprocessing
# -----------------------------
corpus = []

for text in X:
    review = re.sub("[^a-zA-Z]", " ", text)
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in stop_words]
    corpus.append(" ".join(review))

# -----------------------------
# 4) Tokenizer + padding
# -----------------------------
vocab_size = 10000
sent_length = 30

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(corpus)

sequences = tokenizer.texts_to_sequences(corpus)
X_final = pad_sequences(sequences, padding="pre", maxlen=sent_length)
y_final = np.array(y)

# -----------------------------
# 5) Train / test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
)

# -----------------------------
# 6) Model
# -----------------------------
model = Sequential([
    Embedding(vocab_size, 50),
    LSTM(128),
    Dense(1, activation="sigmoid")
])

model.build(input_shape=(None, sent_length))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

print(model.summary())

# -----------------------------
# 7) Train
# -----------------------------
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32
)

# -----------------------------
# 8) Evaluation
# -----------------------------
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {acc:.4f}")

# -----------------------------
# 9) Prediction function
# -----------------------------
def predict_headline(text):
    review = re.sub("[^a-zA-Z]", " ", text)
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in stop_words]
    review = " ".join(review)

    seq = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(seq, padding="pre", maxlen=sent_length)

    prob = model.predict(padded, verbose=0)[0][0]
    label = "FAKE 🟥" if prob >= 0.5 else "REAL 🟩"
    return label, prob

# -----------------------------
# 10) Demo predictions
# -----------------------------
tests = [
    "Government announces new education reform",
    "Aliens found living under Antarctica ice",
    "Scientists publish climate change warning",
    "Miracle pill cures cancer in one day"
]

for t in tests:
    label, prob = predict_headline(t)
    print(f"\n{t}")
    print(f"{label} (prob_fake={prob:.3f})")

# -----------------------------
# 11) Interactive mode
# -----------------------------
print("\nType headline (Enter to quit):")
while True:
    text = input("> ").strip()
    if not text:
        break
    label, prob = predict_headline(text)
    print(f"{label} (prob_fake={prob:.3f})")
