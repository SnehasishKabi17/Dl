import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import imdb
import numpy as np

# Load IMDB dataset
num_words = 10000  # Limit to top 10000 words
maxlen = 200       # Pad sequences to a length of 200

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)

# Pad sequences to the same length
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

# Pad sequences to the same length
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Build the Bidirectional LSTM model
model = Sequential()
model.add(Embedding(input_dim=num_words, output_dim=128, input_length=maxlen))  # Embedding layer
model.add(Bidirectional(layers.SimpleRNN(64, return_sequences=False)))  # Bidirectional LSTM layer
model.add(Dropout(0.5))  # Dropout to prevent overfitting
# Sigmoid activation for binary classification
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=64)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy * 100:.2f}%")

# Function to decode the integer-encoded reviews back to words
def decode_review(encoded_review):
    word_index = imdb.get_word_index()
    reverse_word_index = {value: key for key, value in word_index.items()}
    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review if i > 2])  # Offset by 3 for padding
    return decoded_review

# Function to predict sentiment for a given review
def predict_sentiment(review):
    word_index = imdb.get_word_index()
    encoded_review = [word_index.get(word, 0) + 3 for word in review.split()]  # Add offset by 3
    padded_review = pad_sequences([encoded_review], maxlen=maxlen)
    prediction = model.predict(padded_review, verbose=0)  # Disable verbose output
    sentiment = "Positive" if prediction[0][0] >= 0.5 else "Negative"
    return sentiment

# Output sample sentiments from the test set
print("\nSample Sentiments:")
for i in range(5):  # Print 5 sample sentiments
    review = decode_review(X_test[i])
    sentiment = predict_sentiment(review)
    print(f"Review: {review}\nSentiment: {sentiment}\n")
    print("-" *50)
