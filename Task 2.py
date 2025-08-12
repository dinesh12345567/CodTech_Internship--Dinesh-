# Task 2: Fashion MNIST Classification

# Step 1: Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# Step 2: Load Dataset
def load_data():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Normalize pixel values
    x_train, x_test = x_train / 255.0, x_test / 255.0

    return x_train, y_train, x_test, y_test

# Step 3: Build Model
def build_model():
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Step 4: Plot Training Curves
def plot_history(history):
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Step 5: Predict and Visualize
def plot_predictions(model, x_test, y_test, class_names, num_images=5):
    predictions = model.predict(x_test)

    plt.figure(figsize=(10, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        true_label, img = y_test[i], x_test[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img, cmap=plt.cm.binary)
        predicted_label = np.argmax(predictions[i])
        color = 'blue' if predicted_label == true_label else 'red'
        plt.xlabel(f"{class_names[predicted_label]} ({100*np.max(predictions[i]):.1f}%)", color=color)

    plt.tight_layout()
    plt.show()

# Step 6: Main Execution
if __name__ == "__main__":
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    x_train, y_train, x_test, y_test = load_data()
    model = build_model()

    # Train model
    history = model.fit(x_train, y_train, epochs=5, validation_split=0.1)

    # Evaluate model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")

    # Plot accuracy/loss curves
    plot_history(history)

    # Show predictions
    plot_predictions(model, x_test, y_test, class_names)
