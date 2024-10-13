
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras import layers, models
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import Counter
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the Excel files
train_df = pd.read_excel('/content/train_ex2.xlsx')
val_df = pd.read_excel('/content/val_ex2.xlsx')
test_df = pd.read_excel('/content/test_ex2.xlsx')

print(train_df.head())
print(val_df.head())
print(test_df.head())

def preprocess_text(text):
    # Tokenize the text
    word_tokens = word_tokenize(text)

    # Initialize the Porter Stemmer
    stemmer = PorterStemmer()

    # Get the stopwords
    #stop_words = set(stopwords.words('english'))

    # Convert to lowercase and filter out non-alphabetic tokens
    filtered_tokens = [w.lower() for w in word_tokens if w.isalpha()]

    # Stem each word
    stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]

    # Calculate word frequencies
    word_freq = Counter(stemmed_tokens)

    # Determine the top 10 most frequent words
    top_10_words = set([word for word, _ in word_freq.most_common(10)])

    # Determine the 10 most unique words
    unique_words = set([word for word, count in word_freq.items() if count == 1][:10])

    # Remove the top 10 most frequent and 10 most unique words from filtered tokens
    filtered_tokens = [w for w in stemmed_tokens if w not in top_10_words and w not in unique_words]

    # Join the tokens back into a single string
    return ' '.join(filtered_tokens)

# Preprocess the text data
train_df['text'] = train_df['text'].apply(preprocess_text)
val_df['text'] = val_df['text'].apply(preprocess_text)
test_df['text'] = test_df['text'].apply(preprocess_text)

all_text = pd.concat([train_df['text'], val_df['text'], test_df['text']])
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
vectorizer.fit(all_text)

# x_train = vectorizer.transform(train_df['text']).toarray()
# y_train = train_df['label'].values
# x_val = vectorizer.transform(val_df['text']).toarray()
# y_val = val_df['label'].values
# x_test = vectorizer.transform(test_df['text']).toarray()
# Concatenate all text data and vectorize
all_text = pd.concat([train_df['text'], val_df['text']])
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_all = vectorizer.fit_transform(all_text).toarray()

# Split the vectorized data back into train, val, and test
X_train = X_all[:len(train_df)]
X_val = X_all[len(train_df):len(train_df) + len(val_df)]
X_test = vectorizer.transform(test_df['text']).toarray()

y_train = train_df['label'].values
y_val = val_df['label'].values

# Define the autoencoder using the Functional API
input_dim = X_train.shape[1]
encoding_dim = 1024  

# Input layer
input_img = layers.Input(shape=(input_dim,))

# Encoder
encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
encoded = layers.Dense(encoding_dim // 2, activation='relu')(encoded)
encoded = layers.Dense(encoding_dim // 4, activation='relu')(encoded)

# Decoder
decoded = layers.Dense(encoding_dim // 2, activation='relu')(encoded)
decoded = layers.Dense(encoding_dim, activation='relu')(decoded)
decoded = layers.Dense(input_dim, activation='softmax')(decoded)

# Autoencoder model
autoencoder = models.Model(input_img, decoded)

# Encoder model
encoder = models.Model(input_img, encoded)

# Decoder model
encoded_input = layers.Input(shape=(encoding_dim // 4,))
decoder_layer = autoencoder.layers[-3]
decoder = models.Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

history = autoencoder.fit(X_train, X_train, epochs=20, batch_size=256, shuffle=True, validation_data=(X_val, X_val))

# Print the layers of the autoencoder
print("Autoencoder Layers:")
for layer in autoencoder.layers:
    print(layer.name, layer.input_shape, layer.output_shape)

# Print the layers of the encoder
print("\nEncoder Layers:")
for layer in encoder.layers:
    print(layer.name, layer.input_shape, layer.output_shape)

# Print the layers of the decoder
print("\nDecoder Layers:")
for layer in decoder.layers:
    print(layer.name, layer.input_shape, layer.output_shape)

encoder.summary()
decoder.summary()
autoencoder.summary()

# Plot training & validation loss values
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

!pip install tensorflow
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# Encode the data
X_train_encoded = encoder.predict(X_train)
X_val_encoded = encoder.predict(X_val)
X_test_encoded = encoder.predict(X_test)

# Define the fully connected neural network using the Functional API
input_encoded = layers.Input(shape=(encoding_dim // 4,))
x = layers.Dense(128, activation='relu')(input_encoded)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(32, activation='relu')(x)
output = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(input_encoded, output)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train_encoded, y_train, epochs=250, batch_size=150, validation_data=(X_val_encoded, y_val))

# Print the layers of the classification model
print("\nClassification Model Layers:")
for layer in model.layers:
    print(layer.name, layer.input_shape, layer.output_shape)

# Evaluate the model
loss, accuracy = model.evaluate(X_val_encoded, y_val)
print(f"Validation Loss: {loss}, Validation Accuracy: {accuracy}")

# Predict labels for the test set
test_predictions = model.predict(X_test_encoded)
test_predictions = (test_predictions > 0.5).astype(int)

# Save predictions to CSV file
submission_df = pd.DataFrame({'id': test_df['id'], 'label': test_predictions.flatten()})
submission_df.to_csv('submission.csv', index=False)

print("Submission file created successfully.")