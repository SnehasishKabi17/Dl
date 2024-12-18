import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
from PIL import Image
import matplotlib.pyplot as plt  # Importing matplotlib for displaying the image

# Load and preprocess the MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

timesteps = X_train.shape[1]
input_dim = X_train.shape[2]

# Build the model with Dropout layers
model = Sequential([
    SimpleRNN(128, input_shape=(timesteps, input_dim), activation='relu', return_sequences=False),
    Dropout(0.2),  # Dropout layer after SimpleRNN to prevent overfitting
    Dense(64, activation='relu'),
    Dropout(0.2),  # Dropout layer after Dense layer
    Dense(10, activation='softmax'),
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Function to load and preprocess a batch of images
def preprocess_images(image_paths):
    img_batch = []
    original_images = []
    
    for image_path in image_paths:
        try:
            img = Image.open(image_path).convert('L')  # Convert to grayscale
            img = img.resize((28, 28))  # Resize to 28x28 pixels
            img_array = np.array(img) / 255.0  # Normalize the image
            img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension (28, 28, 1)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 28, 28, 1)
            img_batch.append(img_array)
            original_images.append(img)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue
    
    # Stack all images into a batch (batch_size, 28, 28, 1)
    return np.vstack(img_batch), original_images

# Function to predict classes for a batch of images
def predict_images(image_paths):
    img_batch, original_images = preprocess_images(image_paths)
    
    # Make predictions on the batch of images
    predictions = model.predict(img_batch)
    predicted_classes = np.argmax(predictions, axis=1)
    confidence_scores = np.max(predictions, axis=1)
    
    # Visualize the results
    for i, img in enumerate(original_images):
        plt.imshow(img, cmap='gray')  # Display the image in grayscale
        plt.title(f"Predicted Class: {predicted_classes[i]} - Confidence: {confidence_scores[i]:.2f}")
        plt.axis('off')  # Hide axis
        plt.show()
    
    return predicted_classes, confidence_scores

# Example usage for batch processing
image_paths = [
    '/content/eight.png',
    
    
]

# Predict the classes for all images in the batch
predicted_classes, confidence_scores = predict_images(image_paths)

# Output the results for each image
for i, path in enumerate(image_paths):
    print(f"Image: {path} - Predicted Class: {predicted_classes[i]} - Confidence: {confidence_scores[i]:.2f}")
