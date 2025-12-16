import nltk
# nltk.download('punkt')
# nltk.download('punkt_tab')
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

import numpy as np
import tensorflow as tf
import json
import random

with open('intents.json') as json_data:
    intents = json.load(json_data)

# Preprocessing    

words = []
classes = []
documents = []
ignore = ['?']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words if w not in ignore]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

# Training Data

training = []
output = []

output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]

    for w in words:
        bag.append(1 if w in pattern_words else 0)
        
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])

# shuffle
random.shuffle(training)

# build X and Y arrays directly
train_x = np.array([item[0] for item in training])
train_y = np.array([item[1] for item in training])

print(len(training))        # number of samples
print(len(classes))         # number of classes
print(len(words))           # vocabulary size
print(train_x.shape)        # (n_samples, n_words)
print(train_y.shape)        # (n_samples, n_classes)

# ============================

model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(train_x.shape[1],)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=200, batch_size=8, verbose=1)

model.save('chatbot_model.keras')
