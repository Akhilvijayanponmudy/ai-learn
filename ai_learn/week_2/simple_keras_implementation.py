import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ----------------------------
# 1. Load data
# ----------------------------
df = pd.read_csv("data.csv")

X = df["text"].astype(str).tolist()
y = df["label"].astype(int).tolist()

num_classes = len(set(y))

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------
# 2. Tokenization
# ----------------------------
MAX_WORDS = 20000
MAX_LEN = 200

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post")
X_val_pad = pad_sequences(X_val_seq, maxlen=MAX_LEN, padding="post")

# Explicitly convert to numpy arrays for Keras 3 compatibility
X_train_pad = np.array(X_train_pad)
y_train = np.array(y_train)
X_val_pad = np.array(X_val_pad)
y_val = np.array(y_val)

# ----------------------------
# 3. Model
# ----------------------------
model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=128), # Removed deprecated input_length
    GlobalAveragePooling1D(),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ----------------------------
# 4. Train
# ----------------------------
history = model.fit(
    X_train_pad, y_train,
    validation_data=(X_val_pad, y_val),
    epochs=10,
    batch_size=32
)

